import csv
import numpy as np
import json
import scipy.stats as scipy_stats
import pandas as pd
import math
import copy

VERSIONS_TO_INCLUDE = [
	# 'NXE',
	'OSE',
	'HWN',
	'YAI4Hu',
]
MERGE_ALL_VERSIONS = False

TASK_TO_CONSIDER = [
	'CA',
	'HD'
]

TASK_VERSION_TO_FILENAME = {
	'CA': {
		'NXE': 'CA-1',
		'OSE': 'CA-2',
		'HWN': 'CA-3',
		'YAI4Hu': 'CA-4',
	},
	'HD': {
		'NXE': 'HD-1',
		'OSE': 'HD-2',
		'HWN': 'HD-3',
		'YAI4Hu': 'HD-4',
	},
}

SCORES_TO_SHOW = [
	'Satisfaction',
	'Effectiveness',
	'Elapsed Seconds',
	# 'Learnability',
	# 'NCS'
]

QUESTION_TYPE_FILTER = set([
	'what',
	'why',
	'why-not',
	'how',
	# 'when',
])

ONLY_ANSWERS_IN_INITIAL_EXPLANANS = False
ONLY_ANSWERS_NOT_IN_INITIAL_EXPLANANS = False
KEEP_INCOMPLETE = False
## no filters on NCS
# MIN_NCS = None # Min is -10
# MAX_NCS = None # Max is 20
## with NCS
# MIN_NCS = -10 # Min is -10
# MAX_NCS = 20 # Max is 20
## with normal NCS
MIN_NCS = 5 # Lower quartile is 5
MAX_NCS = 11 # Upper quartile is 11

MIN_SUS = None
MAX_SUS = None # Median SUS is 42.5

MIN_EFF = None #% -> at least 2 correct answer on CA and 3 on HD
MAX_EFF = None # Median score is 42.8%

MIN_SECONDS = None # 1 min
MAX_SECONDS = 40*60 # 40 min

SHOW_FLIERS = True
# SECONDS_LIMIT = 40*60
SCATTER_PLOT_NCS = False

MIN_CORRECT_ANSWERS = 1
ATTENTION_CHECK = False

is_empty = lambda x: x=='' or math.isnan(x)
def is_valid_answer(x):
	if isinstance(x,str):
		return x and x.lower()!='result'
	if isinstance(x,(float,int)):
		return not np.isnan(x)
	return False

format_answers = lambda answers: list(map(int, answers))
format_sus = format_answers
format_ncs = format_answers
format_time = lambda time_str: tuple(map(int,time_str.split(':')))

def time_to_seconds(time):
	h,m,s = time
	return 60*60*h+60*m+s

def compute_sus_score(sus):
	x = sum(sus[::2]) - 5 # Each item’s score contribution ranges from 0 to 4. For items 1, 3, 5, 7, and 9 (the positively worded items) the score contribution is the scale position minus 1.
	y = 25 - sum(sus[1::2]) # For items 2, 4, 6, 8, and 10 (the negatively worded items), the contribution is 5 minus the scale position.
	return (x + y)*2.5

def compute_sus_learnability_score(sus):
	return 50*(5-sus[3] + 5-sus[9])/4

def compute_ncs_score(ncs):
	x = -sum(ncs[2:4])
	y = sum(ncs[:2]+ncs[4:])
	return (x + y)

def get_idx_of_last_valid_answer(answers):
	last_idx = 0
	for i,a in enumerate(answers):
		if is_valid_answer(a):
			last_idx = i+1
	return last_idx

def get_stat_dict(value_list):
	# print(value_list)
	return {
		'median': np.median(value_list),
		'lower_quartile': np.quantile(value_list,0.25),
		'upper_quartile': np.quantile(value_list,0.75),
		'IQR': np.quantile(value_list,0.75)-np.quantile(value_list,0.25),
		'mean': np.mean(value_list),
		'std': np.std(value_list),
		'max': max(value_list),
		'min': min(value_list),
		'len': len(value_list),
	}

file_list = [
	(t+'-'+v,TASK_VERSION_TO_FILENAME[t][v])
	for t in ['CA','HD']
	for v in VERSIONS_TO_INCLUDE
]

content_list = []
for label,filename in file_list:
	# with open(f'data/{filename}.csv', newline='') as csvfile:
	# 	content_list.append(list(map(lambda x: [filename]+x, csv.reader(csvfile, delimiter=','))))
	df_dict = pd.read_excel(f'data/{filename}.xlsx', header=[0, 1])
	# print(df_dict.values.tolist())
	content_list.append(list(map(lambda x: [label]+x, df_dict.values.tolist())))

ID_FIELDS = 2
CA_QUESTIONS_TYPE = [
	{
		'question':"What did the Credit Approval System decide for Mary's application?",
		'types':('what','how'), 
		'answer_in_initial_explanans':True,
	},
	{
		'question':"What is an inquiry (in this context)?",
		'types':'what', 
		'answer_in_initial_explanans':False,
	},
	{
		'question':"What type of inquiries can affect Mary's score, the hard or the soft ones?",
		'types':('what','how'), 
		'answer_in_initial_explanans':False,
	},
	{
		'question':"What is an example of hard inquiry?",
		'types':'what', 
		'answer_in_initial_explanans':False,
	},
	{
		'question':"How can an account become delinquent?",
		'types':('how','why'), 
		'answer_in_initial_explanans':False,
	},
	{
		'question':"Which specific process was used by the Bank to automatically decide whether to assign the loan?",
		'types':('what','how'), 
		'answer_in_initial_explanans':True,
	},
	{
		'question':"What are the known issues of the specific technology used by the Bank (to automatically predict Mary's risk performance and to suggest avenues for improvement)?",
		'types':('what','why'), 
		'answer_in_initial_explanans':False,
	}
]
HD_QUESTIONS_TYPE = [
	{
		'question':"What are the most important factors leading that patient to a medium risk of heart disease?",
		'types':('what','why'), 
		'answer_in_initial_explanans':True,
	},
	{
		'question':"What is the easiest thing that the patient could actually do to change his heart disease risk from medium to low?",
		'types':('what','how'), 
		'answer_in_initial_explanans':True,
	},
	{
		'question':"According to the predictor, what level of serum cholesterol is needed to shift the heart disease risk from medium to high?",
		'types':('what','how'), 
		'answer_in_initial_explanans':True,
	},
	{
		'question':"How could the patient avoid raising bad cholesterol, preventing his heart disease risk to shift from medium to high?",
		'types':'how', 
		'answer_in_initial_explanans':False,
	},
	{
		'question':"What kind of tests can be done to measure bad cholesterol levels in the blood?",
		'types':('what','how'), 
		'answer_in_initial_explanans':False,
	},
	{
		'question':"What are the risks of high cholesterol?",
		'types':('what','why-not'), 
		'answer_in_initial_explanans':False,
	},
	{
		'question':"What is LDL?",
		'types':'what', 
		'answer_in_initial_explanans':False,
	},
	{
		'question':"What is Serum Cholestrol?",
		'types':'what', 
		'answer_in_initial_explanans':False,
	},
	{
		'question':"What types of chest pain are typical of heart disease?",
		'types':('what','how'), 
		'answer_in_initial_explanans':False,
	},
	{
		'question':"What is the most common type of heart disease in the USA?",
		'types':'what', 
		'answer_in_initial_explanans':False,
	},
	{
		'question':"What are the causes of angina?",
		'types':('what','why'), 
		'answer_in_initial_explanans':False,
	},
	{
		'question':"What kind of chest pain do you feel with angina?",
		'types':('what','how'), 
		'answer_in_initial_explanans':False,
	},
	{
		'question':"What are the effects of high blood pressure?",
		'types':('what','why-not'), 
		'answer_in_initial_explanans':False,
	},
	{
		'question':"What are the symptoms of high blood pressure?",
		'types':('what','why'), 
		'answer_in_initial_explanans':False,
	},
	{
		'question':"What are the effects of smoking to the cardiovascular system?",
		'types':('what','why-not'), 
		'answer_in_initial_explanans':False,
	},
	{
		'question':"How can the patient increase his heart rate?",
		'types':'how', 
		'answer_in_initial_explanans':False,
	},
	{
		'question':"How can the patient try to prevent a stroke?",
		'types':'how', 
		'answer_in_initial_explanans':False,
	},
	{
		'question':"What is a Thallium stress test?",
		'types':('what','why'), 
		'answer_in_initial_explanans':False,
	},
	{
		'question':"What is the most probable age to get a heart disease?",
		'types':('what','when'), 
		'answer_in_initial_explanans':False,
	}
]
SUS_QUESTIONS = 10
NFC_QUESTIONS = 6

def filter_by_question_type(scores, version):
	questions_types = CA_QUESTIONS_TYPE if 'CA' in version else HD_QUESTIONS_TYPE
	valid_scores = []
	for qtypes_dict,s in zip(questions_types,scores):
		qtypes = qtypes_dict['types']
		if not isinstance(qtypes,(tuple,list)):
			qtypes = [qtypes]
		else: # 'what' is too generic, remove it
			qtypes = list(filter(lambda x:x!='what', qtypes))
		next_q = next(filter(lambda t: t in QUESTION_TYPE_FILTER, qtypes), False)
		if next_q:
			valid_scores.append(s)
	return valid_scores

def filter_by_initial_explanans(scores, version):
	questions_types = CA_QUESTIONS_TYPE if 'CA' in version else HD_QUESTIONS_TYPE
	return [
		s
		for qtypes_dict,s in zip(questions_types,scores)
		if qtypes_dict['answer_in_initial_explanans']
	]

def filter_by_not_in_initial_explanans(scores, version):
	questions_types = CA_QUESTIONS_TYPE if 'CA' in version else HD_QUESTIONS_TYPE
	return [
		s
		for qtypes_dict,s in zip(questions_types,scores)
		if not qtypes_dict['answer_in_initial_explanans']
	]

def get_normalised_total_score(scores, version):
	scores = scores if not QUESTION_TYPE_FILTER else filter_by_question_type(scores, version)
	scores = scores if not ONLY_ANSWERS_IN_INITIAL_EXPLANANS else filter_by_initial_explanans(scores, version)
	scores = scores if not ONLY_ANSWERS_NOT_IN_INITIAL_EXPLANANS else filter_by_not_in_initial_explanans(scores, version)
	total_score = sum(scores)
	q_len = len(CA_QUESTIONS_TYPE) if 'CA' in version else len(HD_QUESTIONS_TYPE)
	q_len -= q_len-len(scores)
	# print(q_len)
	return 100*total_score/q_len

experiment_sus_dict = {}
ncs_score_dict = {}
tot_ncs_list = []
tot_sus_list = []
tot_sus_learnability_list = []
tot_eff_list = []
old_answers_dict = {
	l: {}
	for l,_ in file_list
}
banned_ips = set()
ip_set = set()
for row_list in content_list:
	for i,row in enumerate(row_list):
		# Timestamp, Gender, Age, Do you have experience with Credit Approval Systems or Finance, What browser are you using?, What time is it NOW?
		version,uid = row[:ID_FIELDS]
		if version not in experiment_sus_dict:
			experiment_sus_dict[version] = {}
		EFFECTIVENESS_QUESTIONS = len(HD_QUESTIONS_TYPE) if 'HD' in version else len(CA_QUESTIONS_TYPE)
		answers_and_scores = row[ID_FIELDS:ID_FIELDS+EFFECTIVENESS_QUESTIONS*2]
		sus = row[ID_FIELDS+EFFECTIVENESS_QUESTIONS*2:ID_FIELDS+EFFECTIVENESS_QUESTIONS*2+SUS_QUESTIONS*2:2]
		if 'CA' in version:
			ncs = row[ID_FIELDS+EFFECTIVENESS_QUESTIONS*2+SUS_QUESTIONS*2:ID_FIELDS+EFFECTIVENESS_QUESTIONS*2+SUS_QUESTIONS*2+NFC_QUESTIONS*2:2]
			_, country, date, time_taken, ip_address, _, status = row[ID_FIELDS+EFFECTIVENESS_QUESTIONS*2+SUS_QUESTIONS*2+NFC_QUESTIONS*2:]
		else:
			ncs = None
			_, country, date, time_taken, ip_address, _, status = row[ID_FIELDS+EFFECTIVENESS_QUESTIONS*2+SUS_QUESTIONS*2:]
		
		elapsed_seconds = time_to_seconds(format_time(time_taken))
		ip_address = ip_address.split(',')[0]
		ip_set.add(ip_address)
		answers = answers_and_scores[::2]
		scores = format_answers(answers_and_scores[1::2])

		scores_with_answer = scores[:get_idx_of_last_valid_answer(answers)]
		if len(scores_with_answer) == 0:
			continue
		if ip_address in old_answers_dict[version]:
			old_scores_with_answer, old_elapsed_seconds = old_answers_dict[version][ip_address]
			elapsed_seconds += old_elapsed_seconds
			scores_with_answer = np.minimum(scores[:len(old_scores_with_answer)],old_scores_with_answer).tolist()+scores_with_answer[len(old_scores_with_answer):]
			old_scores_len = len(scores)
			scores = scores_with_answer + scores[len(scores_with_answer):]
			assert old_scores_len==len(scores), 'something is wrong'
		old_answers_dict[version][ip_address] = (scores_with_answer,elapsed_seconds)
		# elapsed_seconds = min(SECONDS_LIMIT,elapsed_seconds)

		if not KEEP_INCOMPLETE and len(scores_with_answer) != len(scores):
			continue

		if ATTENTION_CHECK:
			if 'CA' in version:
				if scores[0] == 0:
					banned_ips.add(ip_address)
					continue
			if ip_address in banned_ips:
				continue

		# filtered_scores = scores
		# filtered_scores = filtered_scores if not QUESTION_TYPE_FILTER else filter_by_question_type(filtered_scores, version)
		# filtered_scores = filtered_scores if not ONLY_ANSWERS_IN_INITIAL_EXPLANANS else filter_by_initial_explanans(filtered_scores, version)
		# filtered_scores = filtered_scores if not ONLY_ANSWERS_NOT_IN_INITIAL_EXPLANANS else filter_by_not_in_initial_explanans(filtered_scores, version)
		# total_score = sum(scores)
		if sum(scores) < MIN_CORRECT_ANSWERS:
			continue
		total_score = get_normalised_total_score(scores, version)
		
		# if status!='Incomplete' and len(list(filter(is_empty, sus))) == 0:
		if len(list(filter(is_empty, sus))) > 0:
			sus = None
			sus_score = 0 if len(scores_with_answer) < len(scores) else None
			sus_learnability_score = 0 if len(scores_with_answer) < len(scores) else None
		else:
			sus = format_sus(sus)
			sus_score = compute_sus_score(sus)
			sus_learnability_score = compute_sus_learnability_score(sus)

		if 'CA' in version:
			if len(list(filter(is_empty, ncs))) > 0:
				ncs_score = None
			else:
				ncs_score_dict[ip_address] = ncs_score = compute_ncs_score(format_ncs(ncs))
		else:
			ncs_score = ncs_score_dict.get(ip_address,None)
		
		if MIN_NCS is not None and (ncs_score is None or ncs_score < MIN_NCS):
			if ncs_score is None:
				print(f'Cannot process this row: {row} because the ncs_score is None')
			continue
		if MAX_NCS is not None and (ncs_score is None or ncs_score > MAX_NCS):
			if ncs_score is None:
				print(f'Cannot process this row: {row} because the ncs_score is None')
			continue
		if MIN_SUS is not None and (sus_score is None or sus_score < MIN_SUS):
			if sus_score is None:
				print(f'Cannot process this row: {row} because the sus_score is None')
			continue
		if MAX_SUS is not None and (sus_score is None or sus_score > MAX_SUS):
			if sus_score is None:
				print(f'Cannot process this row: {row} because the sus_score is None')
			continue
		if MIN_EFF is not None and total_score < MIN_EFF:
			continue
		if MAX_EFF is not None and total_score > MAX_EFF:
			continue
		if MIN_SECONDS is not None and elapsed_seconds < MIN_SECONDS:
			continue
		if MAX_SECONDS is not None and elapsed_seconds > MAX_SECONDS:
			continue
			
		if ncs_score is not None:
			tot_ncs_list.append(ncs_score)
		if sus_score is not None:
			tot_sus_list.append(sus_score)
		if sus_learnability_score is not None:
			tot_sus_learnability_list.append(sus_learnability_score)
		if total_score is not None:
			tot_eff_list.append(total_score)
		row_dict = {
			'Elapsed Seconds': elapsed_seconds,
			'Effectiveness': total_score,
			'Satisfaction': sus_score,
			'Learnability': sus_learnability_score,
			'NCS': ncs_score,
			'scale': sus
		}
		experiment_sus_dict[version][ip_address] = row_dict

print('IP list:')
for i,ip in enumerate(ip_set):
	print(i, ip)

if MERGE_ALL_VERSIONS:
	for t in TASK_TO_CONSIDER:
		experiment_sus_dict[f'{t}-All'] = {}
		for i in VERSIONS_TO_INCLUDE:
			for k,v in experiment_sus_dict[f'{t}-{i}'].items():
				experiment_sus_dict[f'{t}-All'][k] = copy.deepcopy(v)

print('Global NCS stats:', json.dumps(get_stat_dict(tot_ncs_list), indent=4))
print('Global SUS stats:', json.dumps(get_stat_dict(tot_sus_list), indent=4))
print('Global SUS-Learnability stats:', json.dumps(get_stat_dict(tot_sus_learnability_list), indent=4))
print('Global EFF stats:', json.dumps(get_stat_dict(tot_eff_list), indent=4))

result_dict = {}

for version, ip_row_dict in experiment_sus_dict.items():
	sus_list = list(filter(lambda x:x is not None, map(lambda x: x['Satisfaction'], ip_row_dict.values())))
	sus_learnability_list = list(filter(lambda x:x is not None, map(lambda x: x['Learnability'], ip_row_dict.values())))
	ncs_list = list(filter(lambda x:x is not None, map(lambda x: x['NCS'], ip_row_dict.values())))
	sus_scale_list = list(filter(lambda x:x is not None, map(lambda x: x['scale'], ip_row_dict.values())))
	efficacy_list = list(map(lambda x: x['Effectiveness'], ip_row_dict.values()))
	seconds_list = list(map(lambda x: x['Elapsed Seconds'], ip_row_dict.values()))

	key_result_dict = {
		'test_count': len(efficacy_list),
		'Elapsed Seconds': get_stat_dict(seconds_list),
		'Satisfaction': get_stat_dict(sus_list),
		'Learnability': get_stat_dict(sus_learnability_list),
		'NCS': get_stat_dict(ncs_list),
		'Effectiveness': get_stat_dict(efficacy_list),
	}

	key_result_dict['question_dict'] = {}
	median_sus = []
	for e,q_list in enumerate(zip(*sus_scale_list)):
		key_result_dict['question_dict'][e] = get_stat_dict(q_list)
		median_sus.append(key_result_dict['question_dict'][e]['median'])
	key_result_dict['median_score'] = compute_sus_score(median_sus)
	result_dict[version] = key_result_dict

print('stats:', json.dumps(result_dict, indent=4))

#This test can be used to investigate whether two independent samples were selected from populations having the same distribution.
'''
A low pvalue implies that .
A high pvalue implies that Elapsed Seconds in "No" are not statistically greater than Elapsed Seconds in "Yes".
'''
def test_hypothesis(a, b):
	a_value, a_label = a
	b_value, b_label = b
	# params_dict = {}
	# sse_dict = {}
	# for distr, params, sse in best_fit_distribution(a_value):
	# 	sse_dict[distr] = sse
	# 	params_dict[distr] = [params]
	# for distr, params, sse in best_fit_distribution(b_value):
	# 	if distr not in sse_dict:
	# 		continue
	# 	sse_dict[distr] += sse
	# 	params_dict[distr].append(params)
	# best_distribution = sorted(sse_dict.items(), key=lambda x:x[-1])[0][0]
	# fit_params_a, fit_params_b = params_dict[best_distribution]
	alternatives = ['two-sided','less','greater']
	mannwhitneyu_dict = {}
	for alternative in alternatives:
		mannwhitneyu_dict[a_label + ' is stochastically ' + alternative + (' to ' if alternative == 'two-sided' else ' than ') + b_label] = scipy_stats.mannwhitneyu(a_value, b_value, use_continuity=True, alternative=alternative)
	return {
		# 'wilcoxon': scipy_stats.wilcoxon(a_value,b_value), # The Wilcoxon signed-rank test tests the null hypothesis that two related paired samples come from the same distribution. In particular, it tests whether the distribution of the differences x - y is symmetric about zero. It is a non-parametric version of the paired T-test.
		# 'best_fit_distribution': best_distribution.name,
		# 'params': {
		# 	'a': get_params_description(best_distribution, fit_params_a),
		# 	'b': get_params_description(best_distribution, fit_params_b)
		# },
		'mannwhitneyu': mannwhitneyu_dict,
		'kruskal': scipy_stats.kruskal(a_value,b_value), # Due to the assumption that H has a chi square distribution, the number of samples in each group must not be too small. A typical rule is that each sample must have at least 5 measurements.
	}

def test_omnibus_hypothesis(d_value_list):
	return {
		# 'wilcoxon': scipy_stats.wilcoxon(a_value,b_value), # The Wilcoxon signed-rank test tests the null hypothesis that two related paired samples come from the same distribution. In particular, it tests whether the distribution of the differences x - y is symmetric about zero. It is a non-parametric version of the paired T-test.
		# 'best_fit_distribution': best_distribution.name,
		# 'params': {
		# 	'a': get_params_description(best_distribution, fit_params_a),
		# 	'b': get_params_description(best_distribution, fit_params_b)
		# },
		'kruskal': scipy_stats.kruskal(*d_value_list), # Due to the assumption that H has a chi square distribution, the number of samples in each group must not be too small. A typical rule is that each sample must have at least 5 measurements.
	}

print('Testing hypothesis..')
last_version_name = VERSIONS_TO_INCLUDE[-1]
for v in TASK_TO_CONSIDER:
	print(v, 'omnibus', 'Elapsed Seconds', test_omnibus_hypothesis([[y['Elapsed Seconds'] for y in x.values()] for n,x in experiment_sus_dict.items() if v in n]))
	print(v, 'omnibus', 'Effectiveness', test_omnibus_hypothesis([[y['Effectiveness'] for y in x.values()] for n,x in experiment_sus_dict.items() if v in n]))
	print(v, 'omnibus', 'Satisfaction', test_omnibus_hypothesis([[y['Satisfaction'] for y in x.values()] for n,x in experiment_sus_dict.items() if v in n]))
	print(v, 'omnibus', 'Learnability', test_omnibus_hypothesis([[y['Learnability'] for y in x.values()] for n,x in experiment_sus_dict.items() if v in n]))
	for version_name in VERSIONS_TO_INCLUDE[:-1]:
		version_value = f'{v}-{version_name}'
		last_version_value = f'{v}-{last_version_name}'
		# follows loglaplace distribution
		print(version_value, 'Elapsed Seconds', json.dumps(test_hypothesis( # A low mannwhitneyu pvalue (<0.05) implies that Elapsed Seconds in 'No' are statistically greater than Elapsed Seconds in 'Yes'
			(list(map(lambda x: x['Elapsed Seconds'], experiment_sus_dict[version_value].values())),version_value),
			(list(map(lambda x: x['Elapsed Seconds'], experiment_sus_dict[last_version_value].values())),last_version_value),
		), indent=4))

		# follows gennorm distribution
		print(version_value, 'Effectiveness', json.dumps(test_hypothesis( # A low mannwhitneyu pvalue (<0.05) implies that Effectiveness in 'No' are statistically lower than Effectiveness in 'Yes'
			(list(map(lambda x: x['Effectiveness'], experiment_sus_dict[version_value].values())),version_value),
			(list(map(lambda x: x['Effectiveness'], experiment_sus_dict[last_version_value].values())),last_version_value),
		), indent=4))

		print(version_value, 'Learnability', json.dumps(test_hypothesis( # A high pvalue (>0.95) implies that 'Yes' and 'No' have very similar scores
			(list(filter(lambda x:x is not None, map(lambda x: x['Learnability'], experiment_sus_dict[version_value].values()))),version_value),
			(list(filter(lambda x:x is not None, map(lambda x: x['Learnability'], experiment_sus_dict[last_version_value].values()))),last_version_value),
		), indent=4))

		# follows dgamma distribution
		print(version_value, 'Satisfaction', json.dumps(test_hypothesis( # A high pvalue (>0.95) implies that 'Yes' and 'No' have very similar scores
			(list(filter(lambda x:x is not None, map(lambda x: x['Satisfaction'], experiment_sus_dict[version_value].values()))),version_value),
			(list(filter(lambda x:x is not None, map(lambda x: x['Satisfaction'], experiment_sus_dict[last_version_value].values()))),last_version_value),
		), indent=4))

		a = filter(lambda x:x is not None, map(lambda x: x['scale'], experiment_sus_dict[version_value].values()))
		b = filter(lambda x:x is not None, map(lambda x: x['scale'], experiment_sus_dict[last_version_value].values()))
		print(version_value, 'Single SUS scales:')
		sus_scale_dict = {}
		for e,(a_list,b_list) in enumerate(zip(zip(*a),zip(*b))):
			sus_scale_dict[int(e)+1] = test_hypothesis(
				(a_list,version_value),
				(b_list,last_version_value),
			)
		print(json.dumps(sus_scale_dict, indent=4))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

item_list = list(VERSIONS_TO_INCLUDE)
if MERGE_ALL_VERSIONS:
	item_list.append('All')
df_list = []
for v in TASK_TO_CONSIDER:
	for i in item_list:
		df = pd.DataFrame(experiment_sus_dict[f'{v}-{i}'].values())
		###########
		if SCATTER_PLOT_NCS:
			pp = sns.pairplot(data=df, y_vars=['NCS'], x_vars=['Satisfaction','Effectiveness','Elapsed Seconds','Learnability'])
			plt.savefig(f'scatter_plot_{v}-{i}.png')
			plt.clf()
			plt.cla()
			plt.close()
		###########
		df = pd.melt(df, value_vars=SCORES_TO_SHOW)
		df['version'] = i
		df['domain'] = v
		df_list.append(df)
df = pd.concat(df_list,ignore_index=True)
# print(df)
# print(df.loc[df['variable'] == 'Effectiveness'])

sns.set_style("whitegrid")
if len(SCORES_TO_SHOW) > 1:
	g = sns.FacetGrid(df, col="variable", row='domain', sharex=False, sharey=False,)
else:
	g = sns.FacetGrid(df, col='domain', row="variable", sharex=False, sharey=False,)
def my_boxplot(**kwargs):
	x = kwargs.pop('x')
	y = kwargs.pop('y')
	d = kwargs.pop('data')
	box_plot = sns.boxplot(x=x, y=y, data=d, showfliers=kwargs.get('showfliers'), autorange=kwargs.get('autorange'))
	# box_plot.set_xticklabels(item_list)

	# Calculate number of obs per group & median to position labels
	medians = d.groupby([x])[y].median()
	get_label = lambda s: str(np.round(s, 2))

	for xtick,item in zip(box_plot.get_xticks(),box_plot.get_xticklabels()):
		box_plot.text(
			xtick,
			medians[item.get_text()],
			get_label(medians[item.get_text()]), 
			horizontalalignment='center',
			size='medium',
			# color='w',
			weight='bold',
			ha="center", va="center",
			bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8))
		)
	return box_plot
ax = g.map_dataframe(my_boxplot, x='version', y='value', showfliers=SHOW_FLIERS, autorange=True).set_titles("{row_name} | {col_name}",bbox=dict(boxstyle="round", ec=(0., 0., 0.), fc=(0.9, 0.9, 0.9))).set_axis_labels('version','value')
# Iterate thorugh each axis
for ax in g.axes.flat:
    # Make x and y-axis labels slightly larger
    ax.set_xlabel(ax.get_xlabel(), fontsize='x-large', fontweight='bold')
    ax.set_ylabel(ax.get_ylabel(), fontsize='x-large', fontweight='bold')

# plt.legend()
plt.tight_layout()
plt.savefig('boxplot.png')
plt.clf()
plt.cla()
plt.close()

######################
######################
