import sched, time
import json
from os import mkdir, path as os_path
base_path = os_path.dirname(os_path.abspath(__file__))
cache_path = os_path.join(base_path,'cache')
document_path = os_path.join(base_path,'documents')

from knowpy.models.knowledge_extraction.ontology_builder import OntologyBuilder as OB
# from knowpy.models.reasoning.question_answerer import QuestionAnswerer
from knowpy.models.reasoning.question_answerer_disco import QuestionAnswererDisco
from knowpy.misc.doc_reader import load_or_create_cache, DocParser
from knowpy.misc.graph_builder import get_concept_description_dict, get_betweenness_centrality, save_graphml
from knowpy.misc.levenshtein_lib import labels_are_contained, remove_similar_labels
from more_itertools import unique_everseen

################ Configuration ################

QA2_OPTIONS = {
	'log': False,
	'sbert_model': {
		'url': 'facebook-dpr-question_encoder-multiset-base', # model for paraphrase identification
		'use_cuda': False,
	},
}

QA2_EXTRACTOR_OPTIONS = {
	'models_dir': '/home/toor/Desktop/data/models', 
	'use_cuda': True,
}

ONTOLOGY_BUILDER_DEFAULT_OPTIONS = {
	'spacy_model': 'en_core_web_trf',
	'n_threads': 1,
	'max_syntagma_length': None,
	'add_source': True,
	'add_label': True,
	'lemmatize_label': False,

	'default_similarity_threshold': 0.75,
	'tf_model': {
		'url': 'https://tfhub.dev/google/universal-sentence-encoder-large/5', # Transformer
		# 'url': 'https://tfhub.dev/google/universal-sentence-encoder/4', # DAN
		# 'cache_dir': '/Users/toor/Documents/Software/DLModels/tf_cache_dir/',
		'use_cuda': False,
	},
	'with_centered_similarity': True,
}

CONCEPT_CLASSIFIER_DEFAULT_OPTIONS = {
	'tf_model': {
		'url': 'https://tfhub.dev/google/universal-sentence-encoder-large/5', # Transformer
		# 'url': 'https://tfhub.dev/google/universal-sentence-encoder/4', # DAN
		# 'cache_dir': '/Users/toor/Documents/Software/DLModels/tf_cache_dir/',
		'use_cuda': False,
	},
	'with_centered_similarity': True,
	'default_similarity_threshold': 0.7,
	# 'default_tfidf_importance': 3/4,
}

SUMMARISER_DEFAULT_OPTIONS = {
	'hf_model': {
		# 'url': 't5-base',
		'url': 'facebook/bart-large-cnn', # baseline
		# 'url': 'google/pegasus-billsum',
		# 'url': 'sshleifer/distilbart-cnn-12-6', # speedup (over the baseline): 1.24
		# 'url': 'sshleifer/distilbart-cnn-12-3', # speedup (over the baseline): 1.78
		# 'url': 'sshleifer/distilbart-cnn-6-6', # speedup (over the baseline): 2.09
		# 'cache_dir': '/Users/toor/Documents/Software/DLModels/hf_cache_dir/',
		'framework': 'pt',
		'use_cuda': False,
	},
}

################ Initialise data structures ################
graph_cache = os_path.join(cache_path,f"OB_cache_lemma-{ONTOLOGY_BUILDER_DEFAULT_OPTIONS['lemmatize_label']}.pkl")
betweenness_centrality_cache = os_path.join(cache_path,'QA2_betweenness_centrality.pkl')
qa_cache = os_path.join(cache_path,'QA2_embedder.pkl')
qa_important_concept_classifier_cache = os_path.join(cache_path,'QA2_important_concept_classifier.pkl')
qa_concept_classifier_cache = os_path.join(cache_path,'QA2_concept_classifier.pkl')
qa_sentence_summariser_cache = os_path.join(cache_path,'QA2_sentence_summariser.pkl')
########################################################################
print('Building Ontology Edge List..')
graph = load_or_create_cache(
	graph_cache, 
	lambda: OB(ONTOLOGY_BUILDER_DEFAULT_OPTIONS).set_documents_path(document_path).build()
)
# save_graphml(graph, 'ontology')
print('Ontology size:', len(graph))
print('Grammatical Clauses:', len(list(filter(lambda x: '{obj}' in x[1], graph))))
########################################################################
print('Building Question Answerer..')
betweenness_centrality = load_or_create_cache(
	betweenness_centrality_cache, 
	lambda: get_betweenness_centrality(filter(lambda x: '{obj}' in x[1], graph))
)
qa = QuestionAnswererDisco(
	graph= graph, 
	model_options= QA2_OPTIONS, 
	concept_classifier_options= CONCEPT_CLASSIFIER_DEFAULT_OPTIONS, 
	answer_summariser_options= SUMMARISER_DEFAULT_OPTIONS,
	qa_extractor_options= QA2_EXTRACTOR_OPTIONS,
	betweenness_centrality= betweenness_centrality,
)
qa.load_cache(qa_cache)
qa.important_concept_classifier.load_cache(qa_important_concept_classifier_cache)
qa.concept_classifier.load_cache(qa_concept_classifier_cache)
qa.sentence_summariser.load_cache(qa_sentence_summariser_cache)

if qa.initialise_qa_extractor():
	qa.store_cache(qa_cache)


################ Define methods ################
def get_question_answer_dict(question_list, options=None):
	if not options:
		options = {}
	question_answer_dict = qa.ask(question_list, **options)
	# print('######## Question Answers ########')
	# print(json.dumps(question_answer_dict, indent=4))
	return question_answer_dict

def get_question_answer_dict_quality(question_answer_dict, top=5):
	return qa.get_question_answer_dict_quality(question_answer_dict, top=top)

def get_summarised_question_answer_dict(question_answer_dict, options=None):
	if not options:
		options = {}
	question_summary_tree = qa.summarise_question_answer_dict(question_answer_dict, **options)
	# print('######## Summarised Question Answers ########')
	# print(json.dumps(question_summarised_answer_dict, indent=4))
	# qa.sentence_summariser.store_cache(qa_sentence_summariser_cache)
	return question_summary_tree

def get_concept_overview(query_template_list, concept_uri, options=None):
	if not options:
		options = {}
	# set consider_incoming_relations to False with concept-centred generic questions (e.g. what is it?), otherwise the answers won't be the sought ones
	question_answer_dict = qa.get_concept_overview(
		query_template_list = query_template_list, 
		concept_uri = concept_uri,
		**options
	)
	# print('######## Concept Overview ########')
	# print(concept_uri, json.dumps(question_summarised_answer_dict, indent=4))
	# store_cache()
	return question_answer_dict

def annotate_text(sentence, similarity_threshold=None, max_concepts_per_alignment=1):
	return qa.important_concept_classifier.annotate(
		DocParser().set_content_list([sentence]), 
		similarity_threshold=similarity_threshold, 
		max_concepts_per_alignment=max_concepts_per_alignment
	)

def annotate_question_summary_tree(question_summary_tree, similarity_threshold=None, max_concepts_per_alignment=1):
	return qa.annotate_question_summary_tree(question_summary_tree, similarity_threshold=similarity_threshold, max_concepts_per_alignment=max_concepts_per_alignment)

def get_taxonomical_view(concept_uri, depth=0):
	return qa.get_taxonomical_view(concept_uri, depth=depth)

def annotate_taxonomical_view(taxonomical_view, similarity_threshold=None, max_concepts_per_alignment=1):
	return qa.annotate_taxonomical_view(taxonomical_view, similarity_threshold=similarity_threshold, max_concepts_per_alignment=max_concepts_per_alignment)

def get_equivalent_concepts(concept_uri):
	return qa.adjacency_matrix.get_equivalent_concepts(concept_uri)

def store_cache():
	qa.store_cache(qa_cache)
	qa.important_concept_classifier.load_cache(qa_important_concept_classifier_cache)
	qa.concept_classifier.store_cache(qa_concept_classifier_cache)
	qa.sentence_summariser.store_cache(qa_sentence_summariser_cache)

# ############### Cache scheduler ###############
# SCHEDULING_TIMER = 15*60 # 15 minutes
# from threading import Timer
# def my_task(is_first=False):
# 	if not is_first:
# 		store_cache()
# 	Timer(SCHEDULING_TIMER, my_task).start()
# # start your scheduler
# my_task(is_first=True)
