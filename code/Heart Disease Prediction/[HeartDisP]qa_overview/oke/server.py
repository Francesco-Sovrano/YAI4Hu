import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

from bottle import run, get, post, route, hook, request, response, static_file
import json

from more_itertools import unique_everseen
import sys
port = int(sys.argv[1])
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from server_interface import *

###############################################################
# CORS

@route('/<:re:.*>', method='OPTIONS')
def enable_cors_generic_route():
	"""
	This route takes priority over all others. So any request with an OPTIONS
	method will be handled by this function.

	See: https://github.com/bottlepy/bottle/issues/402

	NOTE: This means we won't 404 any invalid path that is an OPTIONS request.
	"""
	add_cors_headers()

@hook('after_request')
def enable_cors_after_request_hook():
	"""
	This executes after every route. We use it to attach CORS headers when
	applicable.
	"""
	add_cors_headers()

def add_cors_headers():
	try:
		response.headers['Access-Control-Allow-Origin'] = '*'
		response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, OPTIONS'
		response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token'
	except Exception as e:
		print('Error:',e)

def get_from_cache(cache, key, build_fn):
	if key not in cache:
		cache[key] = json.dumps(build_fn())
	return cache[key]

###############################################################
# API - Question Answerer

ANSWERS_CACHE = {}
@get('/answer')
def get_answer():
	response.content_type = 'application/json'
	# question = request.forms.get('question') # post
	question = request.query.get('question')
	if not question.endswith('?'):
		question += '?'
	def build_fn():
		print('Answering..')
		# print(question)
		question_answer_dict = get_question_answer_dict(
			[question],
			options={
				'answer_pertinence_threshold': 0.25, 
				'keep_the_n_most_similar_concepts': 1, 
				'query_concept_similarity_threshold': 0.55, 
				'include_super_concepts_graph': False, 
				'include_sub_concepts_graph': True, 
				'only_relevant_concepts': False,
			}
		)
		print('Summarising..')
		if not question_answer_dict:
			return None
		tree_arity = 1
		question_summary_tree = get_summarised_question_answer_dict(
			question_answer_dict,
			options={
				'ignore_non_grounded_answers': False, 
				'use_abstracts': False, 
				'summary_horizon': 10,
				'tree_arity': tree_arity, 
				# 'cut_factor': 2, 
				'depth': 1,
				'remove_duplicates': False,
				'min_size_for_summarising': 75,
			}
		)
		print('Annotating..')
		return {
			'question_summary_tree': question_summary_tree,
			'annotation_list': annotate_question_summary_tree(question_summary_tree),
			'quality': get_question_answer_dict_quality(question_answer_dict, top=tree_arity),
		}
	return get_from_cache(ANSWERS_CACHE, question, build_fn)

OVERVIEW_CACHE = {}
@get('/overview')
def get_overview():
	response.content_type = 'application/json'
	concept_uri = request.query.get('concept_uri')
	# concept_uri = concept_uri.lower().strip()
	# query_template_list = json.loads(request.query.get('query_template_list'))
	def build_fn():
		print('Answering..')
		query_template_list = [
			##### Descriptive
			'What is {concept}?',
			##### Causal + Justificatory + Theleological
			'Why {concept}?',
			##### Theleological
			# 'What is {concept} for?',
			##### Expository
			'How {concept}?',
			#####
			'Who {concept}?',
			##### Spatial
			'Where {concept}?',
			##### Temporal
			'When {concept}?',
			##### Extra
			# 'Who by?',
			# 'Why not?',
		]
		question_answer_dict = get_concept_overview(
			query_template_list, 
			concept_uri, 
			options={
				'answer_pertinence_threshold': 0.15, 
				'keep_the_n_most_similar_concepts': 1, 
				'query_concept_similarity_threshold': 0.55, 
				'include_super_concepts_graph': False, 
				'include_sub_concepts_graph': True, 
				'only_relevant_concepts': True,
			}
		)
		# # Normalize confidence scores: max is assumed to be 0.1
		# for formatted_answer_list in question_answer_dict.values():
		# 	for answer_dict in formatted_answer_list:
		# 		answer_dict['confidence'] = min(1.,answer_dict['confidence']/0.1)
		print('Summarising..')
		if question_answer_dict:
			question_summary_tree = get_summarised_question_answer_dict(
				question_answer_dict,
				options={
					'ignore_non_grounded_answers': False, 
					'use_abstracts': False, 
					'summary_horizon': 10,
					'tree_arity': 1, 
					# 'cut_factor': 1, 
					'depth': 1,
					'remove_duplicates': True,
					'min_size_for_summarising': 75,
				}
			)
		else:
			question_summary_tree = None
		print('Getting taxonomical view..')
		taxonomical_view = get_taxonomical_view(concept_uri, depth=0)
		print('Annotating..')
		annotation_iter = unique_everseen(annotate_question_summary_tree(question_summary_tree) + annotate_taxonomical_view(taxonomical_view))
		equivalent_concept_uri_set = get_equivalent_concepts(concept_uri)
		equivalent_concept_uri_set.add(concept_uri)
		annotation_iter = filter(lambda x: x['annotation'] not in equivalent_concept_uri_set, annotation_iter)
		return {
			'question_summary_tree': question_summary_tree,
			'taxonomical_view': taxonomical_view,
			'annotation_list': list(annotation_iter),
		}
	return get_from_cache(OVERVIEW_CACHE, concept_uri, build_fn)

ANNOTATION_CACHE = {}
@get('/annotation')
def get_annotation():
	response.content_type = 'application/json'
	# question = request.forms.get('question') # post
	sentence = request.query.get('sentence')
	def build_fn():
		print('Annotating..')
		return annotate_text(sentence, max_concepts_per_alignment=1)
	return get_from_cache(ANNOTATION_CACHE, sentence, build_fn)

if __name__ == "__main__":
	run(host='0.0.0.0', port=port+2, debug=True)
	