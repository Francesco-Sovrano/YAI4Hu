const AIX360_SERVER_URL = location.protocol+'//'+location.hostname+(location.port ? ':'+(parseInt(location.port,10)+1): '')+'/';
console.log('AIX360_SERVER_URL:', AIX360_SERVER_URL);
const GET_SAMPLE_API = AIX360_SERVER_URL+"sample";
const GET_CLASSIFICATION_API = AIX360_SERVER_URL+"classification";
const GET_EXPLAINABLE_CLASSIFICATION_API = AIX360_SERVER_URL+"explainable_classification";
const OKE_SERVER_URL = location.protocol+'//'+location.hostname+(location.port ? ':'+(parseInt(location.port,10)+2): '')+'/';
console.log('OKE_SERVER_URL:', OKE_SERVER_URL);
const GET_OVERVIEW_API = OKE_SERVER_URL+"overview";
const GET_ANSWER_API = OKE_SERVER_URL+"answer";
const GET_ANNOTATION_API = OKE_SERVER_URL+"annotation";

var PROCESS_GRAPH = [];
var COUNTERFACTUAL_SUGGESTION_DICT = {};

const PROCESS_ID = 'my:AIPoweredCreditApprovalSystem';
var PARAMETER_DICT = {
	"ExternalRiskEstimate": "Consolidated risk markers", 
	"MSinceOldestTradeOpen": "Age of oldest account in months", 
	"MSinceMostRecentTradeOpen": "Age of most recent account in months", 
	"AverageMInFile": "Average age of accounts in months", 
	"NumSatisfactoryTrades": "Number of satisfactory accounts", 
	"NumTrades60Ever2DerogPubRec": "Number of accounts ever delinquent by 60 days or more", 
	"NumTrades90Ever2DerogPubRec": "Number of accounts ever delinquent by 90 days or more", 
	"PercentTradesNeverDelq": "Percentage of accounts that were never delinquent", 
	"MSinceMostRecentDelq": "Months since most recent delinquency", 
	"MaxDelq2PublicRecLast12M": "Worst delinquency score from last 12 months of public record", 
	"MaxDelqEver": "Worst delinquency score ever", 
	"NumTotalTrades": "Total number of accounts", 
	"NumTradesOpeninLast12M": "Number of accounts opened in last 12 months", 
	"PercentInstallTrades": "Percentage of accounts that are installment debt", 
	"MSinceMostRecentInqexcl7days": "Months since most recent credit inquiry not within the last 7 days", 
	"NumInqLast6M": "Number of credit inquiries in last 6 months", 
	"NumInqLast6Mexcl7days": "Number of credit inquiries in last 6 months excluding the last 7 days", 
	"NetFractionRevolvingBurden": "Revolving debt (e.g. credit card) balance as a percentage of credit limit", 
	"NetFractionInstallBurden": "Installment debt (e.g. car loan) balance as a percentage of original loan amount", 
	"NumRevolvingTradesWBalance": "Number of revolving debt accounts with a balance", 
	"NumInstallTradesWBalance": "Number of installment debt accounts with a balance", 
	"NumBank2NatlTradesWHighUtilization": "Number of accounts with high utilization", 
	"PercentTradesWBalance": "Percentage of accounts with a balance", 
	"RiskPerformance": "Risk Performance",
}
for (var k in PARAMETER_DICT)
	PARAMETER_DICT[k] = annotate_text('my:'+k,PARAMETER_DICT[k]);

Vue.component('apexchart', VueApexCharts);
var app = new Vue({
	el: '#app',
	data: {
		application: {
			'name': 'Explaining Decisions on Loan Application',
			'welcome': "Welcome",
			'question_input_placeholder': `Write a question about loans that is not too specific to your loan application, we'll try to answer.. e.g. What is an inquiry?`,
			'intro': {
				'desc1': "Here you can:",
				'subdesc1': [
					"Check the results of your loan application.",
					"Understand why your loan application was rejected/approved by the Bank.",
					"Understand what you can improve to increase the likelihood that your loan application is going to be accepted.",
				],
				'title': `The AI-Powered Credit Approval System`,
				'summary': `The Bank is using an ${annotate_text('my:neural_network','Artificial Neural Network')} for predicting your ${annotate_text('my:risk_performance','Risk Performance')}, and on top of it the Bank is using the ${annotate_text('my:cem','Contrastive Explanations Method (CEM)')} to suggest avenues for improvement. CEM should help you to detect the things (e.g. amount of time since last credit ${annotate_text('my:inquiry','inquiry')}, average age of accounts) that caused your ${annotate_text('my:loan_application','loan application')} rejection, by falling outside the acceptable range.`,
			},
			'contrastiveExplanation': {
				'factorsCount': 0,
			},
		},
		loader: {
			'label': "Show loan application of Customer: ",
			'loading': true,
			'loading_label': 'Loading...',
			'value': 48,
			'min': 0,
			'max': 2465
		},
		expansionModal: {
			'activeCards': [],
			'expandedTopics': [],
			'sections': {
				'class': {
					'header': `Type`,
				},
				'description': {
					'header': `Description`,
				},
				'connections': {
					'header': `Connections`,
				},
				'counterfactual': {
					'header': `What If ...?`,
					'incipit': `What if you could change the ${annotate_text("my:feature", 'features', false)} used to decide whether to deny the loan application? Well, actually you can try to modify them and see how their changes affect results.`,
				},
				'details': {
					'header': `Details`,
				},
			},
		},
		counterfactualModal: {
			'header': `What If ...?`,
			'incipit': '<p>With this change, the input to the Credit Approval System is:</p>',
		},
		// YAI-specific fields
		show_overview_modal: false,
		cards: [],
		current_card_index: 0,
	},
	methods: {
		getCustomerName: function (id) {
			const names = ['Mary','John','Bob','Judy'];
			return names[id%names.length];
		},
		getExplanation: function () {
			this.loader.value = clip(this.loader.value, this.loader.min,this.loader.max);
			this.loader.loading = true;

			const self = this;
			$.ajax({
				type: "GET",
				url: GET_SAMPLE_API,
				responseType:'application/json',
				data: {
					'idx': this.loader.value,
				},
				success: function (sample) {
					console.log('Sample:', sample);
					var sample_id = sample.id;
					var sample_label = sample.label;
					var sample_value = sample.value;
					// console.log(sample_id, sample_label, sample_value)
					$.ajax({
						type: "GET",
						url: GET_EXPLAINABLE_CLASSIFICATION_API,
						responseType:'application/json',
						data: {
							'sample_value': JSON.stringify(sample_value),
						},
						success: function (explainable_classification) {
							PROCESS_GRAPH = self.buildProcessGraph(sample, explainable_classification);
							const minimal_entity_static_graph = build_minimal_entity_graph([KNOWN_KNOWLEDGE_GRAPH, PROCESS_GRAPH]);
							KNOWN_ENTITY_DICT = get_entity_dict(minimal_entity_static_graph.concat(build_minimal_type_graph(minimal_entity_static_graph)));
							// console.log(explainable_classification)
							self.displayExplainableClassification(sample_id, explainable_classification);
						}
					});
				}
			});
		},
		displayExplainableClassification: function (sample_id, data) {
			this.loader.loading = false;
			var contrastive_explanations = data.contrastive_explanations;
			console.log('Contrastive Explanation:', contrastive_explanations)
			var performance = data.performance;
			console.log('Process result:', data.output.label, 'is', data.output.value);
			var is_approved = data.output.value == 'Good';
			var explanation_dict = this.application.contrastiveExplanation;

			explanation_dict.resultHeader = `Final Decision`;
			explanation_dict.result = `Your ${PARAMETER_DICT['RiskPerformance']} has been predicted to be <b>${data.output.value}</b>, thus your ${annotate_text('my:loan_application','loan application')} has been <b>${is_approved?'Approved':'Denied'}</b>.`;
			explanation_dict.factorHeader = `Factors contributing to application ${is_approved?'Approval':'Denial'}`;

			explanation_dict.factors = [];
			// 0 is the normalized sample
			// 1 is Pertinent Negative/Positive
			// 2 is PN/PP - normalized sample
			var example_difference = contrastive_explanations.data[2];
			var example_difference_length = example_difference.length;
			var performance_dict = {};
			COUNTERFACTUAL_SUGGESTION_DICT = {};
			for (var index in example_difference)
			{
				if (index == example_difference_length - 1)
					continue
				var elem = example_difference[index]
				var param = contrastive_explanations.columns[index];
				var val = Math.round(contrastive_explanations.data[0][index]);
				COUNTERFACTUAL_SUGGESTION_DICT[param] = [null,val];
				if (elem == 0)
					continue;
				var expected = Math.round(contrastive_explanations.data[1][index]);
				if (val == expected)
					continue;
				var param_label = PARAMETER_DICT[param];
				COUNTERFACTUAL_SUGGESTION_DICT[param] = [expected,val];
				var change_action = val > expected ? "reduced" : "increased";
				var new_factor = `Your <b>${param_label}</b> should be ${change_action} from ${val.toString().replace('.',',')} to ${expected.toString().replace('.',',')}. `;
				explanation_dict.factors.push(new_factor);
				performance_dict[param] = performance[index];
			}
			explanation_dict.factorsCount = explanation_dict.factors.length;

			if (is_approved)
				explanation_dict.factorIncipit = `We observe that your loan application would still have been accepted even if:`;
			else
			{
				if (explanation_dict.factorsCount == 1)
					explanation_dict.factorIncipit = `One thing in your loan application falls outside the acceptable range:`;
				else
					explanation_dict.factorIncipit = `Some things in your loan application fall outside the acceptable range. All would need to improve before acceptance was recommended:`;
			}

			// Add factors relevance
			if (explanation_dict.factorsCount > 1)
			{
				var sorted_performance = Object.entries(performance_dict).sort((b, a) => a[1]-b[1]);
				console.log('Performance:', sorted_performance);
				var [sorted_performance_labels, sorted_performance_values] = zip(sorted_performance);
				sorted_performance_values = sorted_performance_values.map(x=>Math.round(x * 10) / 10);
				// Display chart
				// explanation_dict.chartLabels = sorted_performance_labels.map(x=>$(PARAMETER_DICT[x]).text());
				// explanation_dict.chartData = [sorted_performance_values];
				// explanation_dict.chartDatasetOverride = [{
				// 	label: 'Importance'
				// }];
				// explanation_dict.chartOptions = { legend: { display: true } };

				explanation_dict.chartSeries = [{
					name: "Importance",
					data: sorted_performance_values,
				}];
				explanation_dict.chartOptions = {
					chart: {
						type: 'bar',
						height: 350
					},
					plotOptions: {
						bar: {
							horizontal: true,
						}
					},
					dataLabels: {
						enabled: false
					},
					xaxis: {
						categories: sorted_performance_labels.map(x=>$(PARAMETER_DICT[x]).text()),
					},
				};
				// Display summary
				explanation_dict.importantFactorHeader = `Relative importance of factors contributing to ${is_approved?'Approval':'Denial'}`
				var mostImportant = PARAMETER_DICT[sorted_performance_labels[0]];
				if (is_approved)
					explanation_dict.importantFactorIncipit = `While all ${explanation_dict.factorsCount} factors influenced positively on the final decision, the most important is the <b>${mostImportant}</b>.`
				else
					explanation_dict.importantFactorIncipit = `While all ${explanation_dict.factorsCount} factors need to improve as indicated above, the most important to improve first is the <b>${mostImportant}</b>. You now have insight into what you can do to improve your likelihood of being accepted.`
			}
		},
		buildProcessGraph: function (sample, explainable_classification) {
			var jsonld_graph = {
				'@id': PROCESS_ID,
				'rdfs:label': 'AI-Powered Credit Approval System',
				'dbo:abstract': "This Credit Approval System is used to decide whether to give a loan to a Bank customer. The automated decision process used for this Credit Approval System is based on a customized Artificial Neural Network (ANN) obtained by training the ANN on the HELOC dataset. This customized ANN takes as input some information about the Bank customer, and it produces as output an estimate of the risk (of giving a loan to that customer) together with a brief attempt of explanation of the reasons behind the ANN's decision.",
				'rdfs:seeAlso': ['my:artificial_neural_network','my:fico_heloc_dataset'],
				'@type': 'my:Process',
				'my:api_list': [
					{
						'@id': GET_SAMPLE_API,
						'@type': 'my:API',
						'rdfs:label': 'get_sample',
					},
					{
						'@id': GET_CLASSIFICATION_API,
						'@type': 'my:API',
						'rdfs:label': 'get_classification',
					},
					{
						'@id': GET_EXPLAINABLE_CLASSIFICATION_API,
						'@type': 'my:API',
						'rdfs:label': 'get_explainable_classification',
					},
				],
				'my:process_input': [], 
				'my:process_output': [],
			};
			for (var i in sample.label)
			{
				var input_id = 'Input_'+i;
				var feature_label = sample.label[i];
				var feature_value = sample.value[i];
				var feature_id = 'my:'+feature_label;
				jsonld_graph['my:process_input'].push({
					'@id': feature_id,
					// 'rdfs:label': $(PARAMETER_DICT[feature_label]).text(),
					'my:value': feature_value,
					'my:feature_order': i,
					'my:counterfactual_api_url': GET_CLASSIFICATION_API,
				});
			}
			jsonld_graph['my:process_output'].push({
				'@id': 'my:'+explainable_classification.output.label,
				// 'rdfs:label': $(PARAMETER_DICT[explainable_classification.output.label]).text(),
				'my:value': explainable_classification.output.value,
			});
			return format_jsonld(jsonld_graph);
		},
	}
})
app.getExplanation();
// annotate_sentence(app.application.intro.title, x=>app.application.intro.title=x);
// annotate_sentence(app.application.intro.summary, x=>app.application.intro.summary=x);
// annotate_sentence(app.application.intro.subdesc1[0], x=>app.application.intro.subdesc1[0]=x);
// annotate_sentence(app.application.intro.subdesc1[1], x=>app.application.intro.subdesc1[1]=x);
// annotate_sentence(app.application.intro.subdesc1[2], x=>app.application.intro.subdesc1[2]=x);

// function annotate_sentence(sentence, callback_fn)
// {
// 	$.ajax({
// 		type: "GET",
// 		url: GET_ANNOTATION_API,
// 		responseType:'application/json',
// 		data: {
// 			'sentence': sentence,
// 		},
// 		success: function (annotation_list) {
// 			// console.log('annotation_list', annotation_list);
// 			callback_fn(annotate_html(sentence, annotation_list, linkify));
// 		},
// 	});
// }

// // test fn
// app.show_overview_modal = true;
// app.cards = [
// 	{
// 		'uri':'my:obligation',
// 		'label':titlefy('obligation1'),
// 		'deleted':false,
// 	},
// 	{
// 		'uri':'my:obligation',
// 		'label':titlefy('obligation2'),
// 		'deleted':false,
// 	},
// 	{
// 		'uri':'my:obligation',
// 		'label':titlefy('obligation3'),
// 		'deleted':false,
// 	},
// ]
function annotate_text(annotation_uri, text) {
	return template_expand(text, annotation_uri);
}
