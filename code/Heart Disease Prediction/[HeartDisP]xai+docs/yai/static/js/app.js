const AIX360_SERVER_URL = location.protocol+'//'+location.hostname+(location.port ? ':'+(parseInt(location.port,10)+1): '')+'/';
console.log('SHAP_SERVER_URL:', AIX360_SERVER_URL);
const GET_EXPLAINABLE_CLASSIFICATION_API = AIX360_SERVER_URL+"explainable_classification";
const OKE_SERVER_URL = location.protocol+'//'+location.hostname+(location.port ? ':'+(parseInt(location.port,10)+2): '')+'/';
console.log('OKE_SERVER_URL:', OKE_SERVER_URL);
const GET_OVERVIEW_API = OKE_SERVER_URL+"overview";
const GET_ANSWER_API = OKE_SERVER_URL+"answer";
const GET_ANNOTATION_API = OKE_SERVER_URL+"annotation";

var PROCESS_GRAPH = [];
var COUNTERFACTUAL_SUGGESTION_DICT = {};

const PROCESS_ID = 'my:HeartDiseasePredictor';
var PARAMETER_DICT = {
	'age': 'Age (years)', 
	'resting_bp': 'Diastolic Blood Pressure (mmHg)', 
	'serum_cholesterol': 'Serum Cholestrol (mg/dl)', 
	'maximum_hr': 'Max. heart rate (bpm)', 
	'ST_depression_exercise_vs_rest': 'ST depression after exercise stress (mm)',
	'num_affected_major_vessels': 'Number of major vessels affected by stenosis', 
	'sex': 'Gender', 
	'chest_pain': 'Chest-pain', 
	'chest_pain_anginal_pain': 'Chest-pain: Angina',
	'chest_pain_asymptomatic': 'No chest-pain',
	'chest_pain_non_anginal_pain': 'Chest-pain: Non-anginal',
	'high_fasting_blood_sugar': 'Fasting blood sugar greater than 120mg/dl', 
	'resting_ecg': 'Rest Electrocardiographic (ECG) results',
	'exercise_induced_angina': 'Exercise induced angina', 
	'peak_exercise_ST_segment_slope': 'ST-segment slope after exercise stress',
	'thallium_stress_test_bf': 'Thallium Stress Test results',
	'heart_disease': 'Heart Disease',
};
for (var k in PARAMETER_DICT)
	PARAMETER_DICT[k] = annotate_text('my:'+k,PARAMETER_DICT[k]);

Vue.component("plotly", Plotly);
var app = new Vue({
	el: '#app',
	data: {
		file_mapping: get_unique_elements(FILE_MAPPING, x=>x["@id"]),
		application: {
			'name': 'Heart Disease Predictor',
			'welcome': "Welcome,",
			'question_input_placeholder': `Write a question about heart diseases, we'll try to answer.. e.g. What is angina?`,
			'intro': {
				'desc1': "Here you can:",
				'subdesc1': [
					`Predict the likelihood of having a ${annotate_text('my:heart_disease','heart disease')}, from ${annotate_text('my:vital_parameter','vital parameters')} (e.g. ${annotate_text('my:blood_pressure','blood pressure')}).`,
					`Understand the next concrete actions to suggest to the patient, to prevent/treat an ${annotate_text('my:heart_disease','heart disease')}.`,
				],
				'title': `Heart Disease Predictor`,
				'summary': `We are using ${annotate_text('my:xgboost','XGBoost')} for predicting the likelihood of having a heart disease, and on top of it we are using ${annotate_text('my:tree_shap','Tree SHAP')} to show the relative importance of the ${annotate_text('my:vital_parameter','vital parameters')} used as input. Tree SHAP should help you to detect the vital parameters that are most likely to result in a ${annotate_text('my:heart_disease','heart disease')}.`,
				'disclosure': `N.B. The information provided in this dashboard should not replace the advice or instruction of your Doctor or Health Care Professional.`,
				'data': 'The data used to create and test our predictor is composed by a cohort of 303 patients at the Cleveland Clinic were assessed on multiple characteristics and whether they had heart disease was also recorded. In total 45% (n=139) of patients had heart disease.',
				'training': 'The data was split into a training set (65%, n=196) and a test set (35%, n=107). Using 10-fold cross validation with the training set an initial model was developed and further refined with hyper-parameter tuning. The final model achieved an average AUC of 0.91 (+/- 0.11) in the training set and 0.88 in the test set. Figure 1 to the far right indicates what the model identified as importance of predictors of heart disease. The more important features are towards the top of the figure, which includes characteristics like affected major vessels, asymptomatic chest pain and if the thallium stress test indicates normal blood flow.',
			},
			'feature_importance_explanation': {
				'header': 'Factors contributing to predicted likelihood of heart disease',
				'description': `The figure below indicates the estimated numerical impact of the ${annotate_text('my:vital_parameter','vital parameters')} on the ${annotate_text('my:model','model')} prediction of heart disease likelihood. Blue bars indicate a positive impact and red bars indicate a negative impact to the likelihood of ${annotate_text('my:heart_disease','heart disease')}.`,
			},
			'prediction': {
				'header': 'Predicted heart disease risk',
				'prediction_number': null,
				'prediction_string': `...`,
			},
			'recommended_action': {
				'header': 'Recommended actions',
				'next_action': `...`,
				'risk_dict': {
					'low risk': 'Discuss with a doctor any single large risk factors the patient may have, and otherwise continue/begin healthy lifestyle habits. Follow-up in 12 months',
					'medium risk': 'Discuss lifestyle with a doctor and identify changes to reduce risk. Schedule follow-up in 3 months on how changes are progressing. Recommend performing simple tests to assess positive impact of changes.',
					'high risk': 'Immediate follow-up with a doctor to discuss next steps including additional follow-up tests, lifestyle changes and medications.'
				}
			},
		},
		loader: {
			'loading': true,
			'loading_label': 'Loading...',
		},
		alert: {
			'no_change_in_prediction': {
				'show': false,
				'message': 'Prediction did not change.',
			}
		},
		vital_parameters: [
			{
				'label': `Patient Demographics`,
				'features': {
					'age': {
						'label': PARAMETER_DICT['age'],
						'type':'number',
						'range':[0,120],
						'value':21
					},
					'sex': {
						'label': PARAMETER_DICT['sex'],
						'type':'enum',
						'options':[
							{'text': 'Female', 'value': 'female'},
							{'text': 'Male', 'value': 'male'}
						],
						'value':'male'
					},
				}
			},
			{
				'label': `Patient Health`,
				'features': {
					'resting_bp': {
						'label': PARAMETER_DICT['resting_bp'],
						'type':'number',
						'range':[0,2000],
						'value':105
					},
					'maximum_hr': {
						'label': PARAMETER_DICT['maximum_hr'],
						'type':'number',
						'range':[0,2000],
						'value':151
					},
					'serum_cholesterol': {
						'label': PARAMETER_DICT['serum_cholesterol'],
						'type':'number',
						'range':[0,2000],
						'value':247
					},
					'high_fasting_blood_sugar': {
						'label': PARAMETER_DICT['high_fasting_blood_sugar'],
						'type':'enum',
						'options':[
							{'text': 'No', 'value': 'no'},
							{'text': 'Yes', 'value': 'yes'}
						],
						'value':'no'
					},
					'chest_pain': {
						'label': PARAMETER_DICT['chest_pain'],
						'type':'enum',
						'options':[
							{'text': 'None', 'value': 'asymptomatic'},
							{'text': 'Angina', 'value': 'anginal_pain'},
							{'text': 'Non-anginal', 'value': 'non_anginal_pain'}
						],
						'value':'asymptomatic'
					},
					'exercise_induced_angina': {
						'label': PARAMETER_DICT['exercise_induced_angina'],
						'type':'enum',
						'options':[
							{'text': 'No', 'value': 'no'},
							{'text': 'Yes', 'value': 'yes'}
						],
						'value':'no'
					},
				}
			},
			{
				'label': `ECG results`,
				'features': {
					'resting_ecg': {
						'label': PARAMETER_DICT['resting_ecg'],
						'type':'enum',
						'options':[
							{'text': 'Normal', 'value': 'normal'},
							{'text': 'Not normal', 'value': 'not_normal'}
						],
						'value':'normal'
					},
					'ST_depression_exercise_vs_rest': {
						'label': PARAMETER_DICT['ST_depression_exercise_vs_rest'],
						'type':'number',
						'range':[0,5],
						'value':3
					},
					'peak_exercise_ST_segment_slope': {
						'label': PARAMETER_DICT['peak_exercise_ST_segment_slope'],
						'type':'enum',
						'options':[
							{'text': 'Upsloping', 'value': 'upsloping'},
							{'text': 'Flat', 'value': 'flat'},
							{'text': 'Downsloping', 'value': 'downsloping'}
						],
						'value':'upsloping'
					},
				}
			},
			// {
			// 	'label': `Thallium stress test results`,
			// 	'features': {
			// 		'thallium_stress_test_bf': {
			// 			'label': PARAMETER_DICT['thallium_stress_test_bf'],
			// 			'type':'enum',
			// 			'options':[
			// 				{'text': 'Normal', 'value': 'normal'},
			// 				{'text': 'Defect', 'value': 'defect'}
			// 			],
			// 			'value':'normal'
			// 		},
			// 		'num_affected_major_vessels': {
			// 			'label': PARAMETER_DICT['num_affected_major_vessels'],
			// 			'type':'number',
			// 			'range':[0,3],
			// 			'value':0
			// 		},
			// 	}
			// },
		],
		// YAI-specific fields
		show_overview_modal: false,
		cards: [],
		current_card_index: 0,
	},
	methods: {
		getCustomerName: function () {
			// return this.vital_parameters[0].features.sex.value=='female'?'Mary':'John';
			return 'Responder'
		},
		getExplainablePrediction: function (with_alert=true) {
			// this.loader.loading = true;

			var feature_dict = {};
			for (const group of this.vital_parameters) {
				for (const [k, v_dict] of Object.entries(group.features)) {
					feature_dict[k] = v_dict['value'];
				}
			}
			console.log('Feature dictionary:', feature_dict);

			const self = this;
			$.ajax({
				type: "GET",
				url: GET_EXPLAINABLE_CLASSIFICATION_API,
				responseType:'application/json',
				data: {
					'feature_dict': JSON.stringify(feature_dict),
				},
				success: x=>self.displayExplainablePrediction(with_alert,x),
			});
		},
		displayExplainablePrediction: function (with_alert,result_dict) {
			// console.log(result_dict);
			if (this.loader.loading)
				this.loader.loading = false;

			// this.application.prediction.risk_group = result_dict.risk_group;
			if (with_alert)
				this.alert.no_change_in_prediction.show = result_dict.prediction == this.application.prediction.prediction_number;
			this.application.prediction.prediction_number = result_dict.prediction;
			var prediction_figure = JSON.parse(result_dict.fig_prediction);
			prediction_figure['layout'] = $.extend(prediction_figure['layout'], {
				'autosize': true,
				// 'width': 300,
				'height': 100,
				'margin': {
					'l': 0,
					'r': 0,
					'b': 0,
					't': 0,
					'pad': 0
				},
			});
			prediction_figure['config'] = $.extend(prediction_figure['config'], {
				'staticPlot': true,
				'responsive': true,
				'displayModeBar': false,
			});
			Plotly.newPlot(this.$refs.prediction, prediction_figure);
			this.application.prediction.prediction_string = `Based on the patient's profile, the predicted likelihood of heart disease is ${result_dict.prediction}. This patient is in the ${result_dict.risk_group} group.`;

			this.application.recommended_action.next_action = this.application.recommended_action.risk_dict[result_dict.risk_group];
			this.application.recommended_action.header = `Recommended action(s) for a patient in the ${result_dict.risk_group} group`;

			var feature_importance_figure = JSON.parse(result_dict.fig_shap);
			feature_importance_figure['layout'] = $.extend(feature_importance_figure['layout'], {
				'autosize': true,
				// 'width': 300,
				// 'height': 100,
				// 'margin': {
				// 	'l': 0,
				// 	'r': 0,
				// 	'b': 0,
				// 	't': 0,
				// 	'pad': 0
				// },
			});
			feature_importance_figure['config'] = $.extend(feature_importance_figure['config'], {
				'staticPlot': true,
				'responsive': true,
				'displayModeBar': false,
			});
			Plotly.newPlot(this.$refs.feature_importance_explanation, feature_importance_figure);
		},
	},
	mounted: function(){
		this.getExplainablePrediction(false);
	},
	created: function(){
		this.getExplainablePrediction(false);
	}
})

// app.getExplainablePrediction();
function annotate_text(annotation_uri, text) {
	// return template_expand(text, annotation_uri);
	return text;
}

// console.log(FILE_MAPPING);