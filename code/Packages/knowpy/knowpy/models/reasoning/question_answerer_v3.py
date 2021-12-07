from knowpy.models.reasoning.question_answerer import QuestionAnswerer
from knowpy.models.knowledge_extraction.question_answer_extractor import QuestionAnswerExtractor
from knowpy.models.knowledge_extraction.ontology_builder import OntologyBuilder
from knowpy.misc.levenshtein_lib import remove_similar_labels

class QuestionAnswererV3(QuestionAnswerer):

	def __init__(self, graph, qa_dict_list, model_options, query_concept_classifier_options, answer_classifier_options, answer_summariser_options, qa_extractor_options, ontology_builder_options, triplets_to_use=None):
		if triplets_to_use is None:
			qa_dict_list = filter(lambda x: x['answer_dict']['type'][0] in triplets_to_use, qa_dict_list)
		qa_list = list(map(lambda x: x['abstract'], qa_dict_list))
		self.qa2sentence_dict = {}
		for qa_dict in qa_dict_list:
			sentence = qa_dict['answer_dict']['sentence']
			source_uri = qa_extractor.content_to_source_dict[sentence]
			if qa_dict['abstract'] in self.qa2sentence_dict:
				self.qa2sentence_dict[qa_dict['abstract']].append((sentence,source_uri))
			else:
				self.qa2sentence_dict[qa_dict['abstract']] = [(sentence,source_uri)]

		self.original_adjacency_matrix = AdjacencyMatrix(
			graph, 
			equivalence_relation_set=set(),
			is_sorted=True,
		)

		new_graph = OntologyBuilder(ontology_builder_options).set_content_list(qa_list).build()
		super().__init__(new_graph, model_options, query_concept_classifier_options, answer_classifier_options, answer_summariser_options)

	def get_sub_graph(self, uri, depth=None, predicate_filter_fn=lambda x: x != SUBCLASSOF_PREDICATE and '{obj}' not in x):
		uri_set = self.original_adjacency_matrix.get_predicate_chain(set([uri]), direction_set=['out'], depth=depth, predicate_filter_fn=predicate_filter_fn)
		return [
			(s,p,o)
			for s in uri_set
			for p,o in self.original_adjacency_matrix.get_outcoming_edges_matrix(s)
		]

	def ask(self, *arg, **args):
		question_answer_dict = super().ask(*arg, **args)
		sentence_set = set()
		for k,v in question_answer_dict.items():
			new_v = []
			for i in v:
				for sentence, source_uri_list in self.qa2sentence_dict[i['sentence']]:
					if sentence in sentence_set:
						continue
					source_uri = source_uri_list[0]
					new_i = dict(i)
					new_i['sentence'] = sentence
					new_i['annotation'] = self.get_sub_graph(source_uri) if source_uri else None
					new_i['source_id'] = source_uri if source_uri else sentence
					sentence_set.add(sentence)
					new_v.append(new_i)
			question_answer_dict[k] = remove_similar_labels(new_v, key=lambda x: x['sentence'])
		return question_answer_dict
