import sys
import json
import numpy as np
from more_itertools import unique_everseen
import sentence_transformers as st
import itertools

from knowpy.misc.graph_builder import get_root_set, get_concept_set, get_predicate_set, get_object_set, get_connected_graph_list, get_ancestors, filter_graph_by_root_set, tuplefy, get_concept_description_dict, get_betweenness_centrality
from knowpy.misc.cache_lib import load_or_create_cache, create_cache, load_cache
from knowpy.misc.levenshtein_lib import labels_are_contained, remove_similar_labels
from knowpy.misc.adjacency_matrix import AdjacencyMatrix

from knowpy.misc.doc_reader import DocParser
from knowpy.models.model_manager import ModelManager
from knowpy.models.reasoning.question_answerer import QuestionAnswerer, word_frequency_distribution, is_common_word
from knowpy.models.knowledge_extraction.question_answer_extractor import QuestionAnswerExtractor
from knowpy.models.summarisation.neural_sentence_summariser import NeuralSentenceSummariser
from knowpy.models.classification.concept_classifier import ConceptClassifier
from knowpy.misc.jsonld_lib import *
from knowpy.models.reasoning import singlefy, no_wh_words


cosine_similarity = lambda a,b: float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
# euclidean_distance = lambda a,b: float(np.linalg.norm(a-b))

dot_product_to_percentage = lambda x: (np.arccos(x)*(180/np.pi))/90 # https://math.stackexchange.com/questions/3125703/how-can-i-turn-the-dot-product-of-two-normalized-vectors-into-a-percentage
no_wh_words = lambda x: x.lower() not in ['why','how','what','where','when','who','which']

class QuestionAnswererDisco(QuestionAnswerer):
	def __init__(self, graph, model_options, concept_classifier_options, answer_summariser_options, qa_extractor_options, betweenness_centrality):
		super().__init__(graph, model_options, concept_classifier_options, None, answer_summariser_options, betweenness_centrality)
		self.qa_extractor = QuestionAnswerExtractor(qa_extractor_options)
		# Build content-to-source dict
		self.content_to_source_dict = {}
		for source_uri,sentence_list in self.content_dict.items():
			for sentence in sentence_list:
				source_uri_list = self.content_to_source_dict.get(sentence,None)
				if source_uri_list is None:
					source_uri_list = self.content_to_source_dict[sentence] = []
				source_uri_list.append(source_uri)
		
		self.qa_dict_list = None
		self.corpus_embeddings = None

	def embed_qa(self):
		# Extract corpus embeddings
		print('Embedding QA matrix..')
		sbert_model = self.model_options['sbert_model']['url']
		if sbert_model in ['nq-distilbert-base-v1']:
			question_answer_list = list(map(lambda x:(x['abstract'],x['answer_dict']['sentence']),self.qa_dict_list))
			# Test: worst performance without Elementary Discourse Units? Apparently so
			# question_answer_list = list(map(lambda x:(x['answer_dict']['answer'],x['answer_dict']['sentence']),self.qa_dict_list))
			self.corpus_embeddings = np.array(self.run_sbert_embedding(question_answer_list))
		elif sbert_model == 'facebook-dpr-question_encoder-single-nq-base':
			question_answer_list = list(map(lambda x:f"{x['abstract']} [SEP] {x['answer_dict']['sentence']}",self.qa_dict_list))
			# Test: worst performance without Elementary Discourse Units? Apparently so
			# question_answer_list = list(map(lambda x:f"{x['answer_dict']['answer']} [SEP] {x['answer_dict']['sentence']}",self.qa_dict_list))
			self.corpus_embeddings = np.array(st.SentenceTransformer('facebook-dpr-ctx_encoder-single-nq-base').encode(question_answer_list))
		elif sbert_model == 'facebook-dpr-question_encoder-multiset-base':
			question_answer_list = list(map(lambda x:f"{x['abstract']} [SEP] {x['answer_dict']['sentence']}",self.qa_dict_list))
			# Test: worst performance without Elementary Discourse Units? Apparently so
			# question_answer_list = list(map(lambda x:f"{x['answer_dict']['answer']} [SEP] {x['answer_dict']['sentence']}",self.qa_dict_list))
			self.corpus_embeddings = np.array(st.SentenceTransformer('facebook-dpr-ctx_encoder-multiset-base').encode(question_answer_list))
		else:
			question_answer_list = list(map(lambda x:x['abstract'],self.qa_dict_list))
			self.corpus_embeddings = np.array(self.run_sbert_embedding(question_answer_list))
		# self.corpus_embeddings = np.array(load_or_create_cache('corpus_embeddings.pkl', lambda: self.run_sbert_embedding(question_answer_list)))
		return self.corpus_embeddings

	def store_cache(self, cache_name):
		super().store_cache(cache_name)
		create_cache(cache_name+'.qa_dict_list.pkl', lambda: self.qa_dict_list)
		create_cache(cache_name+'.corpus_embeddings.pkl', lambda: self.corpus_embeddings)

	def load_cache(self, cache_name):
		super().load_cache(cache_name)

		# loaded_cache = load_cache(cache_name+'.qa_extractor.pkl')
		# if loaded_cache:

		# 	qa_dict_list = loaded_cache.get('qa_dict_list',None)
		# 	if qa_dict_list is not None:
		# 		self.qa_dict_list = qa_dict_list

		# 	corpus_embeddings = loaded_cache.get('corpus_embeddings',None)
		# 	if corpus_embeddings is not None:
		# 		self.corpus_embeddings = corpus_embeddings

		qa_dict_list = load_cache(cache_name+'.qa_dict_list.pkl')
		if qa_dict_list is not None:
			self.qa_dict_list = qa_dict_list

		corpus_embeddings = load_cache(cache_name+'.corpus_embeddings.pkl')
		if corpus_embeddings is not None:
			self.corpus_embeddings = corpus_embeddings

	@staticmethod
	def get_question_answer_dict_quality(question_answer_dict, top=5):
		return {
			question: {
				# 'confidence': {
				# 	'best': answers[0]['confidence'],
				# 	'top_mean': sum(map(lambda x: x['confidence'], answers[:top]))/top,
				# },
				# 'syntactic_similarity': {
				# 	'best': answers[0]['syntactic_similarity'],
				# 	'top_mean': sum(map(lambda x: x['syntactic_similarity'], answers[:top]))/top,
				# },
				# 'semantic_similarity': {
				# 	'best': answers[0]['semantic_similarity'],
				# 	'top_mean': sum(map(lambda x: x['semantic_similarity'], answers[:top]))/top,
				# },
				'valid_answers_count': len(answers),
				# 'syntactic_similarity': answers[0]['syntactic_similarity'],
				'semantic_similarity': answers[0]['semantic_similarity'],
			}
			for question,answers in question_answer_dict.items()
			if answers
		}

	def initialise_qa_extractor(self):
		done = False
		if self.qa_dict_list is None:
			self.qa_dict_list = self.qa_extractor.extract(self.graph)
			done = True
		if self.corpus_embeddings is None:
			self.embed_qa()
			done = True
		assert len(self.corpus_embeddings)==len(self.qa_dict_list), f"corpus_embeddings ({len(self.corpus_embeddings)}) should have the same length of qa_dict_list ({len(self.qa_dict_list)})"
		return done

	def ask(self, query_list, query_concept_similarity_threshold=0.55, answer_pertinence_threshold=0.55, with_numbers=True, remove_stopwords=False, lemmatized=False, keep_the_n_most_similar_concepts=1, filter_extra_fn=no_wh_words, include_super_concepts_graph=False, include_sub_concepts_graph=False, query_known_concepts=None, only_relevant_concepts=False, **args):
		self.initialise_qa_extractor()
		# Searching for answers..
		semantic_search_results = st.util.semantic_search(
			query_embeddings= np.array(self.run_sbert_embedding(query_list)),
			corpus_embeddings= self.corpus_embeddings,
			top_k= min(100,len(self.qa_dict_list)),
			score_function=st.util.dot_score,
		)
		# Organising answers by relevant concepts..
		question_answer_dict = {}
		for i,(query,query_results) in enumerate(zip(query_list,semantic_search_results)):
			if only_relevant_concepts:
				if query_known_concepts is None or query not in query_known_concepts:
					concepts_dict = self.concept_classifier.get_concept_dict(
						DocParser().set_content_list([query]),
						similarity_threshold=query_concept_similarity_threshold, 
						with_numbers=with_numbers, 
						remove_stopwords=remove_stopwords, 
						lemmatized=lemmatized,
						filter_extra_fn=filter_extra_fn,
					)
					concept_set = set(unique_everseen((
						concept_similarity_dict["id"]
						for concept_label, concept_count_dict in concepts_dict.items()
						for concept_similarity_dict in itertools.islice(
							unique_everseen(concept_count_dict["similar_to"], key=lambda x: x["id"]), 
							keep_the_n_most_similar_concepts
						)
					)))
				else:
					concept_set = set(query_known_concepts[query])
				expanded_concept_set = set(concept_set)
				# Get sub-classes
				if include_sub_concepts_graph:
					sub_concept_set = self.adjacency_matrix.get_predicate_chain(
						concept_set = concept_set, 
						predicate_filter_fn = lambda x: x == SUBCLASSOF_PREDICATE, 
						direction_set = ['in'],
						depth = None,
					)
					expanded_concept_set |= sub_concept_set
				# Get super-classes
				if include_super_concepts_graph:
					super_concept_set = self.adjacency_matrix.get_predicate_chain(
						concept_set = concept_set, 
						predicate_filter_fn = lambda x: x == SUBCLASSOF_PREDICATE, 
						direction_set = ['out'],
						depth = None,
					)
					expanded_concept_set |= super_concept_set
				if self.log:
					print(f'Concepts in "{query}": {expanded_concept_set}')

				relevant_concept_label_list = sorted(
					sum((
						[
							(label,uri)
							for label in self.label_dict[uri]
						]
						for uri in expanded_concept_set 
						if uri in self.label_dict
					), []),
					key=lambda x: len(x),
					reverse=True,
				)
				if self.log:
					print(f'Relevant labels found in query "{query}": {relevant_concept_label_list}')

				def get_relevant_uri_tuple(txt):
					return tuple(sorted(unique_everseen([
						uri
						for label,uri in relevant_concept_label_list
						if label in txt
					])))

			# uri_answer_dict = {}
			answer_dict_iter = []
			sentence_set = set()
			for result_dict in query_results:
				confidence = cosine_similarity(self.run_sbert_embedding(query_list)[i], self.corpus_embeddings[result_dict['corpus_id']])
				# confidence = float(result_dict['score'])
				if confidence < answer_pertinence_threshold:
					continue
				qa_dict = self.qa_dict_list[result_dict['corpus_id']]
				sentence = qa_dict['answer_dict']['sentence']
				if sentence in sentence_set: # ignore duplicates
					continue
				sentence_set.add(sentence)
				abstract = qa_dict['abstract']
				if only_relevant_concepts:
					relevant_uri_tuple = get_relevant_uri_tuple(abstract)
					if not relevant_uri_tuple: # ignore elements that do not contain relevant labels
						continue
				related_question = qa_dict['question']
				related_answer = qa_dict['answer_dict']['answer']
				related_answer_type = qa_dict['answer_dict']['type']
				if sentence not in self.content_to_source_dict:
					print('Error: sentence not found:', sentence)
				else:
					for source_uri in self.content_to_source_dict[sentence]:
						# uri_answer_dict_list = uri_answer_dict.get(relevant_uri_tuple,None)
						# if not uri_answer_dict_list:
						# 	uri_answer_dict_list = uri_answer_dict[relevant_uri_tuple] = []
						# uri_answer_dict_list.append({
						answer_dict_iter.append({
							'abstract': abstract,
							'confidence': confidence,
							'score': float(result_dict['score']),
							# 'relevant_to': relevant_uri_tuple,
							'syntactic_similarity': confidence,
							'semantic_similarity': confidence,
							'annotation': self.get_sub_graph(source_uri) if source_uri else None,
							'sentence': sentence, 
							'triple': (related_question,related_answer_type,related_answer), 
							'source_id': source_uri if source_uri else sentence, 
						})
			# Merge answer_dict lists
			# answer_dict_iter = itertools.chain(*itertools.zip_longest(*uri_answer_dict.values()))
			answer_dict_iter = filter(lambda x: x is not None, answer_dict_iter)
			answer_dict_iter = sorted(answer_dict_iter, key=lambda x: x['confidence']*x['score'], reverse=True)
			question_answer_dict[query] = remove_similar_labels(
				list(answer_dict_iter), 
				key=lambda x: x['abstract']
			)
		return question_answer_dict

	def get_concept_overview(self, query_template_list, concept_uri, concept_label=None, **args):
		if not concept_label:
			concept_label = self.get_label(concept_uri)
		# set consider_incoming_relations to False with concept-centred generic questions (e.g. what is it?), otherwise the answers won't be the sought ones
		query_list = tuple(map(lambda x:x.replace('{concept}',concept_label), query_template_list))
		query_known_concepts = {
			query: [concept_uri]
			for query in query_list
		}
		return self.ask(query_list, query_known_concepts=query_known_concepts, **args)
