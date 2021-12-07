import os
import json
from quansx.model_building.pipeline import pipeline
from quansx.utils.levenshtein_lib import remove_similar_labels
from more_itertools import unique_everseen
from tqdm import tqdm



class QAExtractor:

	def __init__(self, options_dict=None):
		if options_dict is None:
			options_dict = {}
		self.model_type = options_dict.get('model_type', 'distilt5')
		self.model_data = options_dict.get('model_data', 'disco-qaamr')
		self.models_dir = options_dict.get('models_dir', './data/models')
		self.use_cuda = options_dict.get('use_cuda', False)
		model_name = f"{self.model_type}-{self.model_data}-multi"
		self.question_generator = pipeline("question-answer-generation", model=os.path.join(self.models_dir,model_name), use_cuda=self.use_cuda)
		self.generate_kwargs = options_dict.get('generate_kwargs', {
			"max_length": 128,
			"num_beams": 10,
			# "length_penalty": 1.5,
			# "no_repeat_ngram_size": 3, # do not set it when answer2question=False, questions always start with the same ngrams 
			# "early_stopping": True,
			"num_return_sequences": 1,
		})
		self.e2e_generate_kwargs = options_dict.get('e2e_generate_kwargs', {
			"max_length": 128,
			"num_beams": 5,
			# "length_penalty": 1.5,
			# "no_repeat_ngram_size": 3, # do not set it when answer2question=False, questions always start with the same ngrams 
			# "early_stopping": True,
			"num_return_sequences": 5,
		})

	@staticmethod
	def e2e_generator_filter_fn(prediction_results):
		# print('before',prediction_results)
		prediction_results = filter(lambda x:x, prediction_results)
		# prediction_results = list(prediction_results)
		# prediction_results = filter(lambda x: len(next(filter(lambda y: x!=y and y in x, prediction_results),[]))==0, prediction_results)
		prediction_results = sorted(prediction_results, key=len) # from smaller to bigger
		prediction_results = remove_similar_labels(prediction_results)
		# print('after',prediction_results)
		return prediction_results

	def extract_question_answer_dict(self, sentence_list):
		print('Extracting question_answer_dict..')
		question_answer_list = sum(
			(
				list(map(lambda x:(*x,sentence,answer2question,key), self.question_generator({
					'key': key, 
					'context': sentence,
					'answer2question': answer2question,
					'generate_kwargs': self.generate_kwargs,
					'e2e_generate_kwargs': self.e2e_generate_kwargs,
					'e2e_generator_filter_fn': self.e2e_generator_filter_fn,
				})))
				for sentence in tqdm(sentence_list)
				for answer2question in [True,False]
				for key in tuple(filter(lambda x:x in self.model_data, ['disco','qaamr']))
			),[]
		)
		# Clean question-answers
		question_answer_list = list(filter(lambda x:x[1].casefold() not in x[0].casefold(), question_answer_list))# remove questions containing the answer
		# Build question_answer_dict: for retrieving answers by questions, in average constant time using an hash table
		question_answer_dict = {}
		for qa in question_answer_list:
			question = qa[0]
			answer = qa[1:]
			if question not in question_answer_dict:
				question_answer_dict[question] = []
			question_answer_dict[question].append(answer)
		question_answer_dict = {
			q: tuple(unique_everseen(a, key=lambda x: (x[0].casefold(),x[1].casefold())))
			for q,a in question_answer_dict.items()
		}
		return question_answer_dict
