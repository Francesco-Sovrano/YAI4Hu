# Standard
import pandas as pd
import numpy as np
import os

# For plotting risk indicator and for creating waterfall plot
import plotly.graph_objs as go
import shap

# To import pkl file model objects
import joblib
from pycaret.classification import load_model

# Load model and pipeline
current_folder = os.path.dirname(__file__)
hdpred_model = load_model(os.path.join(current_folder, 'model', 'heart_disease_prediction_model_Jul2020'))

PARAMETER_DICT = {
	'age': 'Age (years)', 
	'resting_bp': 'Diastolic Blood Pressure (mmHg)', 
	'serum_cholesterol': 'Serum Cholestrol (mg/dl)', 
	'maximum_hr': 'Max. heart rate (bpm)', 
	'ST_depression_exercise_vs_rest': 'ST depression after exercise stress (mm)',
	'num_affected_major_vessels_0.0': 'Number of major vessels affected by stenosis', 
	'sex_male': 'Gender', 
	'chest_pain': 'Chest-pain', 
	'chest_pain_anginal_pain': 'Chest-pain: Angina',
	'chest_pain_non_anginal_pain': 'Chest-pain: Non-anginal',
	'chest_pain_asymptomatic': 'Chest-pain: None',
	'high_fasting_blood_sugar_no': 'Fasting blood sugar lower than 120mg/dl', 
	'high_fasting_blood_sugar_yes': 'Fasting blood sugar greater than 120mg/dl', 
	'resting_ecg_not_normal': 'Abnormal Rest Electrocardiographic (ECG) results',
	'resting_ecg_normal': 'Normal Rest Electrocardiographic (ECG) results',
	'exercise_induced_angina_yes': 'Exercise induced angina', 
	'exercise_induced_angina_no': 'No exercise induced angina', 
	'peak_exercise_ST_segment_slope_upsloping': 'ST-segment slope after exercise stress: Upsloping',
	'peak_exercise_ST_segment_slope_flat': 'ST-segment slope after exercise stress: Flat',
	'peak_exercise_ST_segment_slope_downsloping': 'ST-segment slope after exercise stress: Downsloping',
	'thallium_stress_test_bf_normal': 'Thallium Stress Test: Normal',
	'thallium_stress_test_bf_defect': 'Thallium Stress Test: Defect',
	'heart_disease': 'Heart Disease',
}

def generate_feature_matrix(feature_dict):

	# generate a new X_matrix for use in the predictive models
	column_names = ['age', 'resting_bp', 'serum_cholesterol', 'maximum_hr', 'ST_depression_exercise_vs_rest',
					'sex', 'chest_pain', 'high_fasting_blood_sugar', 'resting_ecg',
					'exercise_induced_angina', 'peak_exercise_ST_segment_slope']

	values = [
		feature_dict[k]
		for k in column_names
	]

	x_patient = pd.DataFrame(data=[values],
							 columns=column_names,
							 index=[0])

	return x_patient.to_json()

def predict_hd_summary(feature_dict):
	data_patient = generate_feature_matrix(feature_dict)

	# read in data and predict likelihood of heart disease
	x_new = pd.read_json(data_patient)
	y_val = hdpred_model.predict_proba(x_new)[:, 1]*100
	text_val = str(np.round(y_val[0], 1)) + "%"

	# assign a risk group
	if y_val/100 <= 0.275685:
		risk_grp = 'low risk'
	elif y_val/100 <= 0.795583:
		risk_grp = 'medium risk'
	else:
		risk_grp = 'high risk'

	# # assign an action related to the risk group
	# rg_actions = {'low risk': ['Discuss with patient any single large risk factors they may have, and otherwise '
	# 						   'continue supporting healthy lifestyle habits. Follow-up in 12 months'],
	# 			  'medium risk': ['Discuss lifestyle with patient and identify changes to reduce risk. '
	# 							  'Schedule follow-up with patient in 3 months on how changes are progressing. '
	# 							  'Recommend performing simple tests to assess positive impact of changes.'],
	# 			  'high risk': ['Immediate follow-up with patient to discuss next steps including additional '
	# 							'follow-up tests, lifestyle changes and medications.']}

	# next_action = rg_actions[risk_grp][0]

	# create a single bar plot showing likelihood of heart disease
	fig1 = go.Figure()
	fig1.add_trace(go.Bar(
		y=[''],
		x=y_val,
		marker_color='rgb(112, 128, 144)',
		orientation='h',
		# width=1,
		text=text_val,
		textposition='auto',
		hoverinfo='skip'
	))

	# add blocks for risk groups
	bot_val = 0.5
	top_val = 1

	fig1.add_shape(
		type="rect",
		x0=0,
		y0=bot_val,
		x1=0.275686 * 100,
		y1=top_val,
		line=dict(
			color="white",
		),
		fillcolor="green"
	)
	fig1.add_shape(
		type="rect",
		x0=0.275686 * 100,
		y0=bot_val,
		x1=0.795584 * 100,
		y1=top_val,
		line=dict(
			color="white",
		),
		fillcolor="orange"
	)
	fig1.add_shape(
		type="rect",
		x0=0.795584 * 100,
		y0=bot_val,
		x1=1 * 100,
		y1=top_val,
		line=dict(
			color="white",
		),
		fillcolor="red"
	)
	fig1.add_annotation(
		x=0.275686 / 2 * 100,
		y=0.75,
		text="Low risk",
		showarrow=False,
		font=dict(color="black", size=14)
	)
	fig1.add_annotation(
		x=0.53 * 100,
		y=0.75,
		text="Medium risk",
		showarrow=False,
		font=dict(color="black", size=14)
	)
	fig1.add_annotation(
		x=0.9 * 100,
		y=0.75,
		text="High risk",
		showarrow=False,
		font=dict(color="black", size=14)
	)
	fig1.update_layout(
		# autosize=True,
		# margin=dict(l=0, r=50, t=10, b=10), 
		xaxis={'range': [0, 100]}
	)

	# do shap value calculations for basic waterfall plot
	# print(hdpred_model.named_steps)
	steps = ['dtypes','imputer','new_levels1','new_levels','feature_time','dummy','fix_perfect','clean_names']
	for step in steps:
		x_new = hdpred_model.named_steps[step].transform(x_new)

	explainer_patient = shap.TreeExplainer(hdpred_model.named_steps["trained_model"])
	shap_values_patient = explainer_patient.shap_values(x_new)
	updated_fnames = x_new.T.reset_index()
	updated_fnames.columns = ['feature', 'value']
	updated_fnames['shap_original'] = pd.Series(shap_values_patient[0].flatten())
	updated_fnames['shap_abs'] = updated_fnames['shap_original'].abs()
	updated_fnames = updated_fnames.sort_values(by=['shap_abs'], ascending=True)

	# need to collapse those after first 9, so plot always shows 10 bars
	show_features = 9
	num_other_features = updated_fnames.shape[0] - show_features
	col_other_name = f"{num_other_features} other features"
	f_group = pd.DataFrame(updated_fnames.head(num_other_features).sum()).T
	f_group['feature'] = col_other_name
	plot_data = pd.concat([f_group, updated_fnames.tail(show_features)])

	# additional things for plotting
	plot_range = plot_data['shap_original'].cumsum().max() - plot_data['shap_original'].cumsum().min()
	plot_data['text_pos'] = np.where(plot_data['shap_original'].abs() > (1/9)*plot_range, "inside", "outside")
	plot_data['text_col'] = "white"
	plot_data.loc[(plot_data['text_pos'] == "outside") & (plot_data['shap_original'] < 0), 'text_col'] = "#3283FE"
	plot_data.loc[(plot_data['text_pos'] == "outside") & (plot_data['shap_original'] > 0), 'text_col'] = "#F6222E"

	fig2 = go.Figure(go.Waterfall(
		name="",
		orientation="h",
		measure=['absolute'] + ['relative']*show_features,
		base=explainer_patient.expected_value,
		textposition=plot_data['text_pos'],
		text=plot_data['shap_original'],
		textfont={"color": plot_data['text_col']},
		texttemplate='%{text:+.2f}',
		y=list(map(lambda x: PARAMETER_DICT.get(x,x), plot_data['feature'])),
		x=plot_data['shap_original'],
		connector={"mode": "spanning", "line": {"width": 1, "color": "rgb(102, 102, 102)", "dash": "dot"}},
		decreasing={"marker": {"color": "#3283FE"}},
		increasing={"marker": {"color": "#F6222E"}},
		hoverinfo="skip"
	))
	fig2.update_layout(
		waterfallgap=0.2,
		# autosize=True,
		# width=800,
		# height=400,
		paper_bgcolor='rgba(0,0,0,0)',
		plot_bgcolor='rgba(0,0,0,0)',
		yaxis=dict(
			showgrid=True,
			zeroline=True,
			showline=True,
			gridcolor='lightgray'
		),
		xaxis=dict(
			showgrid=False,
			zeroline=False,
			showline=True,
			showticklabels=True,
			linecolor='black',
			tickcolor='black',
			ticks='outside',
			ticklen=5
		),
		# margin={'t': 25, 'b': 50},
		shapes=[
			dict(
				type='line',
				yref='paper', y0=0, y1=1.02,
				xref='x', x0=plot_data['shap_original'].sum()+explainer_patient.expected_value,
				x1=plot_data['shap_original'].sum()+explainer_patient.expected_value,
				layer="below",
				line=dict(
					color="black",
					width=1,
					dash="dot")
			)
		]
	)
	total_contribution = plot_data['shap_original'].sum()+explainer_patient.expected_value
	fig2.update_yaxes(automargin=True)
	fig2.add_annotation(
		yref='paper',
		xref='x',
		x=total_contribution,
		y=-0.2,
		text="High probability of Heart Disease" if total_contribution > 0 else "Low probability of Heart Disease",
		showarrow=False,
		font=dict(color="black", size=14)
	)
	fig2.add_annotation(
		yref='paper',
		xref='x',
		x=total_contribution,
		y=1.075,
		text=f"Sum of contributions is {'positive' if total_contribution >= 0 else 'negative'} ({total_contribution:.2f})",
		showarrow=False,
		font=dict(color="black", size=14)
	)

	return fig1.to_json(), text_val, risk_grp, fig2.to_json()

