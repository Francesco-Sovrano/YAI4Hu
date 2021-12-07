import itertools
import logging
from typing import Optional, Dict, Union
import re
from more_itertools import unique_everseen

from quansx.utils.transformers_lib import preprocess_text

import torch
from transformers import(
	AutoModelForSeq2SeqLM, 
	AutoTokenizer,
	PreTrainedModel,
	PreTrainedTokenizer,
)

logger = logging.getLogger(__name__)
	
class QuestionAnswerGenerationPipeline:
	def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, use_cuda: bool):
		self.model = model
		self.tokenizer = tokenizer

		self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
		self.model.to(self.device)

		assert self.model.__class__.__name__ in ["T5ForConditionalGeneration", "BartForConditionalGeneration"]
		
		if "T5ForConditionalGeneration" in self.model.__class__.__name__:
			self.model_type = "t5"
		else:
			self.model_type = "bart"

	def __call__(self, inputs: Union[Dict, str]):
		if inputs.get('answer2question',True):
			if "answer" in inputs:
				return self.generate_question(inputs["answer"], inputs)
			answers = self.extract_answers(inputs)
			return list(map(lambda x: (self.generate_question(x, inputs), x), answers))
		else: # question2answer
			if "question" in inputs:
				return self.answer_question(inputs["question"], inputs)
			questions = self.extract_questions(inputs)
			return list(map(lambda x: (x, self.answer_question(x, inputs)), questions))

	def answer_question(self, question, inputs_dict): 
		return self._extract(
			question, 
			inputs_dict["context"], 
			inputs_dict["key"], 
			self._prepare_inputs_for_answer_gen,
			generate_kwargs=inputs_dict.get("generate_kwargs",None),
		)

	def generate_question(self, answer, inputs_dict): 
		return self._extract(
			answer, 
			inputs_dict["context"], 
			inputs_dict["key"], 
			self._prepare_inputs_for_question_gen,
			generate_kwargs=inputs_dict.get("generate_kwargs",None),
		)
	
	def extract_questions(self, inputs_dict): 
		return self._extract_e2e(
			inputs_dict["context"], 
			inputs_dict["key"], 
			self._prepare_inputs_for_e2e_question_gen, 
			generate_kwargs=inputs_dict.get("e2e_generate_kwargs",None),
			e2e_generator_filter_fn=inputs_dict.get("e2e_generator_filter_fn",None), # lambda x:x.endswith('?')
		)

	def extract_answers(self, inputs_dict): 
		return self._extract_e2e(
			inputs_dict["context"], 
			inputs_dict["key"], 
			self._prepare_inputs_for_e2e_answer_gen,
			generate_kwargs = inputs_dict.get("e2e_generate_kwargs",None),
			e2e_generator_filter_fn=inputs_dict.get("e2e_generator_filter_fn",None), # lambda x:x.endswith('?')
		)

	def _tokenize(self, inputs, max_length):
		inputs = self.tokenizer.batch_encode_plus(
			inputs, 
			max_length=max_length,
			add_special_tokens=True,
			truncation=True,
			padding=True, # Pad to the longest sequence in the batch (or no padding if only a single sequence if provided).
			return_tensors="pt"
		)
		return inputs
	
	def _prepare_inputs_for_question_gen(self, answer, context, key):
		source_text = f"{key} answer: {answer}  {key} context: {context}"
		# if self.model_type == "t5":
		#	 source_text = source_text + " </s>"
		return source_text

	def _prepare_inputs_for_answer_gen(self, question, context, key):
		source_text = f"{key} question: {question}  {key} context: {context}"
		# if self.model_type == "t5":
		#	 source_text = source_text + " </s>"
		return source_text

	def _prepare_inputs_for_e2e_question_gen(self, context, key):
		source_text = f"{key} e2e questions: {context}"
		# if self.model_type == "t5":
		#	 source_text = source_text + " </s>"
		return source_text

	def _prepare_inputs_for_e2e_answer_gen(self, context, key):
		source_text = f"{key} e2e answers: {context}"
		# if self.model_type == "t5":
		#	 source_text = source_text + " </s>"
		return source_text
	
	def _extract(self, source, context, key, input_formatter_fn, generate_kwargs=None):
		source = preprocess_text(source)
		context = preprocess_text(context)
		source_text = input_formatter_fn(source, context, key)
		# print('answer', source_text)
		inputs = self._tokenize([source_text], max_length=generate_kwargs['max_length'])
		if generate_kwargs is None:
			generate_kwargs = {}
		outs = self.model.generate(
			input_ids=inputs['input_ids'].to(self.device), 
			attention_mask=inputs['attention_mask'].to(self.device), 
			**generate_kwargs,
		)
		return self.tokenizer.decode(outs[0], skip_special_tokens=True)

	def _extract_e2e(self, context, key, input_formatter_fn, generate_kwargs=None, e2e_generator_filter_fn=None):
		context = preprocess_text(context)
		source_text = input_formatter_fn(context, key)
		inputs = self._tokenize([source_text], max_length=generate_kwargs['max_length'])
		if generate_kwargs is None:
			generate_kwargs = {}
		outs = self.model.generate(
			input_ids=inputs['input_ids'].to(self.device), 
			attention_mask=inputs['attention_mask'].to(self.device),
			**generate_kwargs,
		)
		prediction_results = []
		for x in outs:
			prediction = self.tokenizer.decode(x, skip_special_tokens=True)
			prediction_results += map(preprocess_text, prediction.split("<sep>"))
		prediction_results = list(unique_everseen(prediction_results))
		if e2e_generator_filter_fn:
			prediction_results = e2e_generator_filter_fn(prediction_results)
		return prediction_results

class QuestionGenerationPipeline(QuestionAnswerGenerationPipeline):
	def __call__(self, inputs: Union[Dict, str]):
		return self.extract_questions(inputs)

class AnswerGenerationPipeline(QuestionAnswerGenerationPipeline):
	def __call__(self, inputs: Union[Dict, str]):
		return self.extract_answers(inputs)

SUPPORTED_TASKS = {
	"question-answer-generation": {
		"impl": QuestionAnswerGenerationPipeline,
		"default": {
			"model": "valhalla/t5-small-qa-qg-hl",
		}
	},
	"question-generation": {
		"impl": QuestionGenerationPipeline,
		"default": {
			"model": "valhalla/t5-small-e2e-qg",
		}
	},
	"answer-generation": {
		"impl": AnswerGenerationPipeline,
		"default": {
			"model": "valhalla/t5-small-qa-qg-hl",
		}
	},
}

def pipeline(
	task: str,
	model: Optional = None,
	tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
	use_cuda: Optional[bool] = True,
	**kwargs,
):
	# Retrieve the task
	if task not in SUPPORTED_TASKS:
		raise KeyError("Unknown task {}, available tasks are {}".format(task, list(SUPPORTED_TASKS.keys())))

	targeted_task = SUPPORTED_TASKS[task]
	task_class = targeted_task["impl"]

	# Use default model/config/tokenizer for the task if no model is provided
	if model is None:
		model = targeted_task["default"]["model"]
	
	# Try to infer tokenizer from model or config name (if provided as str)
	if tokenizer is None:
		if isinstance(model, str):
			tokenizer = model
		else:
			# Impossible to guest what is the right tokenizer here
			raise Exception(
				"Impossible to guess which tokenizer to use. "
				"Please provided a PretrainedTokenizer class or a path/identifier to a pretrained tokenizer."
			)
	
	# Instantiate tokenizer if needed
	if isinstance(tokenizer, (str, tuple)):
		if isinstance(tokenizer, tuple):
			# For tuple we have (tokenizer name, {kwargs})
			tokenizer = AutoTokenizer.from_pretrained(tokenizer[0], **tokenizer[1])
		else:
			tokenizer = AutoTokenizer.from_pretrained(tokenizer)
	
	# Instantiate model if needed
	if isinstance(model, str):
		model = AutoModelForSeq2SeqLM.from_pretrained(model)
	
	return task_class(model=model, tokenizer=tokenizer, use_cuda=use_cuda)
	