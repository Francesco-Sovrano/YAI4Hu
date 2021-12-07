from bottle import run, get, post, route, hook, request, response, static_file
import json

import os
import sys
port = int(sys.argv[1])
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
# API - Explainable Classifier

from explainable_loan_risk_classifier import ExplainableLoanRiskClassifier, TRAIN_MODEL, PROCESS_DATA
EXPERIMENT = ExplainableLoanRiskClassifier(train=TRAIN_MODEL, process_data=PROCESS_DATA, debug=False)

@get('/sample')
def get_sample():
	response.content_type = 'application/json'
	idx = int(request.query.get('idx'))
	return json.dumps(EXPERIMENT.get_sample(idx))

@get('/classification')
def get_classification():
	response.content_type = 'application/json'
	sample_value = json.loads(request.query.get('sample_value')) # post
	return json.dumps(EXPERIMENT.classify(sample_value))

LRC_CACHE = {}
@get('/explainable_classification')
def get_explainable_classification():
	response.content_type = 'application/json'
	sample_value = json.loads(request.query.get('sample_value')) # post
	sample_value_str = str(sample_value)
	def build_fn():
		classifier_output = EXPERIMENT.classify(sample_value)
		result = EXPERIMENT.get_explainable_classification(sample_value, classifier_output['value'])
		result['output'] = classifier_output
		return result
	return get_from_cache(LRC_CACHE,sample_value_str,build_fn)

if __name__ == "__main__":
	run(host='0.0.0.0', port=port+1, debug=True)
	