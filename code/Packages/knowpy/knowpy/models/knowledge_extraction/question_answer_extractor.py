import sys
import json
import numpy as np
from more_itertools import unique_everseen

from knowpy.models.model_manager import ModelManager
from quansx.question_answer_extraction import QAExtractor
from knowpy.misc.levenshtein_lib import labels_are_contained
from knowpy.misc.adjacency_matrix import AdjacencyMatrix

from knowpy.misc.jsonld_lib import *
from knowpy.models.reasoning import singlefy

class QuestionAnswerExtractor(ModelManager):
	def __init__(self, model_options):
		super().__init__(model_options)
		self.disable_spacy_component = ["ner", "textcat", "tagger", "lemmatizer"]

	def extract(self, graph):
		# Build adjacency matrix from knowledge graph
		adjacency_matrix = AdjacencyMatrix(
			graph, 
			equivalence_relation_set=set([IS_EQUIVALENT]),
			is_sorted=True,
		)
		content_dict = adjacency_matrix.get_predicate_dict(CONTENT_PREDICATE, singlefy)
		# Build content-to-source dict
		content_to_source_dict = {}
		for source_uri,sentence_list in content_dict.items():
			for sentence in sentence_list:
				source_uri_list = content_to_source_dict.get(sentence,None)
				if source_uri_list is None:
					source_uri_list = content_to_source_dict[sentence] = []
				source_uri_list.append(source_uri)
		print('Extracting QA matrix..')
		# Extract QA dictionary
		sentence_list = list(content_to_source_dict.keys())
		question_answer_matrix = QAExtractor(self.model_options).extract_question_answer_dict(sentence_list)
		# question_answer_matrix = {}
		# step = 10
		# for i in range(int(np.ceil(len(sentence_list)/step))):
		# 	question_answer_matrix.update(load_or_create_cache(
		# 		f'question_answer_matrix_p{i}.pkl', 
		# 		lambda: QAExtractor(self.model_options).extract_question_answer_dict(sentence_list[i*step:(i+1)*step])
		# 	))
		question_answer_matrix = { # Remove badly posed question-answer couples
			question:tuple((
				{
					'answer': answer,
					'sentence': sentence,
					'type': (key,answer2question),
				}
				for answer,sentence,answer2question,key in v_list
				if not labels_are_contained(answer.casefold(),question.casefold())
			))
			for question,v_list in question_answer_matrix.items()
		}
		question_answer_matrix = dict(filter(lambda x:x[1], question_answer_matrix.items()))
		return [
			{
				'abstract': ' '.join([q if q.strip().endswith('?') else q+'?',a_dict['answer']]),
				'question': q,
				'answer_dict': a_dict,
			}
			for q,a_dict_list in question_answer_matrix.items()
			for a_dict in unique_everseen(a_dict_list, key=lambda x:(x['answer'],x['sentence']))
			if a_dict['answer'] and q
		]
