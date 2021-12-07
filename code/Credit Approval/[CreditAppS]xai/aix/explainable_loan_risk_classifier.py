import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import os
import json
import random
import pandas as pd
import numpy as np
from keras.models import Sequential, Model, load_model, model_from_json
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from aix360.algorithms.contrastive import CEMExplainer, KerasClassifier
#from aix360.algorithms.protodash import ProtodashExplainer
from aix360.datasets.heloc_dataset import HELOCDataset

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
TRAIN_MODEL = False # Train model or load a trained model
PROCESS_DATA = False # Clean data and split dataset into train/test

class ExplainableLoanRiskClassifier(object):
	CLASS_NAMES = ['Bad', 'Good']
	# <a name="contrastive"></a>
	# ## 4. Customer: Contrastive explanations for HELOC Use Case
	# 
	# We now demonstrate how to compute contrastive explanations using AIX360 and how such explanations can help home owners understand the decisions made by AI models that approve or reject their HELOC applications. 
	# 
	# Typically, home owners would like to understand why they do not qualify for a line of credit and if so what changes in their application would qualify them. On the other hand, if they qualified, they might want to know what factors led to the approval of their application. 
	# 
	# In this context, contrastive explanations provide information to applicants about what minimal changes to their profile would have changed the decision of the AI model from reject to accept or vice-versa (_pertinent negatives_). For example, increasing the number of satisfactory trades to a certain value may have led to the acceptance of the application everything else being the same. 
	# 
	# The method presented here also highlights a minimal set of features and their values that would still maintain the original decision (_pertinent positives_). For example, for an applicant whose HELOC application was approved, the 
	# explanation may say that even if the number of satisfactory trades was reduced to a lower number, the loan would have still gotten through.
	# 
	# Additionally, organizations (Banks, financial institutions, etc.) would like to understand trends in the behavior of their AI models in approving loan applications, which could be done by studying contrastive explanations for individuals whose loans were either accepted or rejected. Looking at the aggregate statistics of pertinent positives for approved applicants the organization can get insight into what minimal set of features and their values play an important role in acceptances. While studying the aggregate statistics of pertinent negatives the organization can get insight into features that could change the status of rejected applicants and potentially uncover ways that an applicant may game the system by changing potentially non-important features that could alter the models outcome. 
	# 
	# The contrastive explanations in AIX360 are implemented using the algorithm developed in the following work:
	# ###### [Explanations based on the Missing: Towards Contrastive Explanations with Pertinent Negatives](https://arxiv.org/abs/1802.07623)
	# 
	# We now provide a brief overview of the method. As mentioned above the algorithm outputs a contrastive explanation which consists of two parts: a) pertinent negatives (PNs) and b) pertinent positives (PPs). PNs identify a minimal set of features which if altered would change the classification of the original input. For example, in the loan case if a person's credit score is increased their loan application status may change from reject to accept. The manner in which the method accomplishes this is by optimizing a change in the prediction probability loss while enforcing an elastic norm constraint that results in minimal change of features and their values. Optionally, an auto-encoder may also be used to force these minimal changes to produce realistic PNs. PPs on the other hand identify a minimal set of features and their values that are sufficient to yield the original input's classification. For example, an individual's loan may still be accepted if the salary was 50K as opposed to 100K. Here again we have an elastic norm term so that the amount of information needed is minimal, however, the first loss term in this case tries to make the original input's class to be the winning class. For a more in-depth discussion, please refer to the above work.
	# 
	# The three main steps to obtain a contrastive explanation are shown below. The first two steps are more about processing the data and building an AI model while the third step computes the actual explanation. 
	# 
	#  [Step 1. Process and Normalize HELOC dataset for training](#c1)<br>
	#  [Step 2. Define and train a NN classifier](#c2)<br>
	#  [Step 3. Compute contrastive explanations for a few applicants](#c3)<br>
	
	def __init__(self, train, process_data, debug=False, np_random_seed=None, tf_random_seed=None):
		self.train = train
		self.debug = debug
		self.process_data = process_data
		# Load HELOC dataset
		heloc = HELOCDataset(
			custom_preprocessing=self.default_preprocessing, 
			dirpath=SCRIPT_DIR
		)
		self.heloc_dataframe = heloc.dataframe()
		self.heloc_dataframe = self.heloc_dataframe[[
			"ExternalRiskEstimate",
			"MSinceOldestTradeOpen",
			"MSinceMostRecentTradeOpen",
			"AverageMInFile",
			"NumSatisfactoryTrades",
			"NumTrades60Ever2DerogPubRec",
			"NumTrades90Ever2DerogPubRec",
			"PercentTradesNeverDelq",
			"MSinceMostRecentDelq",
			"MaxDelq2PublicRecLast12M",
			"MaxDelqEver",
			"NumTotalTrades",
			"NumTradesOpeninLast12M",
			"PercentInstallTrades",
			"MSinceMostRecentInqexcl7days",
			"NumInqLast6M",
			"NumInqLast6Mexcl7days",
			"NetFractionRevolvingBurden",
			"NetFractionInstallBurden",
			"NumRevolvingTradesWBalance",
			"NumInstallTradesWBalance",
			"NumBank2NatlTradesWHighUtilization",
			"PercentTradesWBalance",
			"RiskPerformance",
		]]
		if self.debug:
			pd.set_option('display.max_rows', 500)
			pd.set_option('display.max_columns', 24)
			pd.set_option('display.width', 1000)
			print("Size of HELOC dataset:", self.heloc_dataframe.shape)
			print("Number of \"Good\" applicants:", np.sum(self.heloc_dataframe['RiskPerformance']=='Good'))
			print("Number of \"Bad\" applicants:", np.sum(self.heloc_dataframe['RiskPerformance']=='Bad'))
			print("Sample Applicants:")
			self.heloc_dataframe.head(10).transpose()

		# <a name="c1"></a>
		# ### Step 1. Process and Normalize HELOC dataset for training
		# 
		# We will first process the HELOC dataset before using it to train an NN model that can predict the
		# target variable RiskPerformance. The HELOC dataset is a tabular dataset with numerical values. However, some of the values are negative and need to be filtered. The processed data is stored in the file heloc.npz for easy access. The dataset is also normalized for training.
		# 
		# The data processing and model building is very similar to the Loan Officer persona above, where ProtoDash was the method of choice. We repeat these steps here so that both the use cases can be run independently.
		# #### a. Process the dataset

		if self.process_data:
			(data, self.trainset_x, self.testset_x, self.trainset_y, self.testset_y) = heloc.split()
			np.savez(
				os.path.join(SCRIPT_DIR,'heloc.npz'),
				Data=data, 
				x_train=self.trainset_x, 
				x_test=self.testset_x, 
				y_train_b=self.trainset_y, 
				y_test_b=self.testset_y
			)
		else:
			heloc = np.load(os.path.join(SCRIPT_DIR,'heloc.npz'), allow_pickle = True)
			# data = heloc['Data']
			self.trainset_x = heloc['x_train']
			self.testset_x  = heloc['x_test']
			self.trainset_y = heloc['y_train_b']
			self.testset_y  = heloc['y_test_b']

		# #### b. Normalize the dataset
		self.dataset = np.vstack((self.trainset_x, self.testset_x))
		self.Zmax = np.max(self.dataset, axis=0)
		self.Zmin = np.min(self.dataset, axis=0)

		self.normalized_dataset = self.normalize(self.dataset)
		self.normalized_trainset_x = self.normalized_dataset[0:self.trainset_x.shape[0], :]
		self.normalized_testset_x  = self.normalized_dataset[self.trainset_x.shape[0]:, :]

		# <a name="c2"></a>
		# ### Step 2. Define and train a NN classifier
		# 
		# Let us now build a loan approval model based on the HELOC dataset.
		# 
		# #### a. Define NN architecture
		# We now define the architecture of a 2-layer neural network classifier whose predictions we will try to interpret. 

		# #### b. Train the NN
		# Set random seeds for repeatability
		if np_random_seed is not None:
			np.random.seed(np_random_seed)
		if tf_random_seed is not None:
			tf.set_random_seed(tf_random_seed) 

		# compile and print model summary
		self.tf_model = self.nn_small()
		self.tf_model.compile(loss=self.loss_function, optimizer='adam', metrics=['accuracy'])
		if self.debug:
			self.tf_model.summary()

		if self.train:
			if self.debug:
				print('Training model')
			self.tf_model.fit(self.normalized_trainset_x, self.trainset_y, batch_size=128, epochs=1000, verbose=1, shuffle=False)
			self.tf_model.save_weights(os.path.join(SCRIPT_DIR,'heloc_nnsmall.h5'))
		else:
			if self.debug:
				print('Loading pre-trained model')
			self.tf_model.load_weights(os.path.join(SCRIPT_DIR,'heloc_nnsmall.h5'))

		print('Building Keras Classifier')
		mymodel = KerasClassifier(self.tf_model)
		print('Building CEM Explainer')
		self.explainer = CEMExplainer(mymodel)

	@staticmethod
	def default_preprocessing(df):
		# Details and preprocessing for FICO dataset
		# minimize dependence on ordering of columns in heloc data
		# x_cols, y_col = df.columns[0:-1], df.columns[-1]
		x_cols = list(df.columns.values)
		x_cols.remove('RiskPerformance')
		y_col = list(['RiskPerformance'])

		# Preprocessing the HELOC dataset
		# Remove all the rows containing -9 in the ExternalRiskEstimate column
		df = df[df.ExternalRiskEstimate != -9]
		# add columns for -7 and -8 in the dataset
		for col in x_cols:
			df[col][df[col].isin([-7, -8, -9])] = 0
		# Get the column names for the covariates and the dependent variable
		df = df[(df[x_cols].T != 0).any()]

		# minimize dependence on ordering of columns in heloc data
		# x = df.values[:, 0:-1]
		x = df[x_cols].values

		# encode target variable ('bad', 'good')
		cat_values = df[y_col].values
		enc = LabelEncoder()
		enc.fit(cat_values)
		num_values = enc.transform(cat_values)
		y = np.array(num_values)

		return np.hstack((x, y.reshape(y.shape[0], 1)))

	@staticmethod
	def loss_function(correct, predicted): # loss function
		return tf.nn.softmax_cross_entropy_with_logits(labels=correct, logits=predicted)

	def normalize(self, V): #normalize an array of samples to range [-0.5, 0.5]
		VN = (V - self.Zmin)/(self.Zmax - self.Zmin)
		VN = VN - 0.5
		return(VN)
		
	def rescale(self, normalized_X): # rescale a sample to recover original values for normalized values. 
		return(np.multiply ( normalized_X + 0.5, (self.Zmax - self.Zmin) ) + self.Zmin)

	@staticmethod
	def nn_small(): # self.tf_model with no softmax
		model = Sequential()
		model.add(Dense(10, input_dim=23, kernel_initializer='normal', activation='relu'))
		model.add(Dense(2, kernel_initializer='normal'))	
		return model

	def evaluate_model(self):
		# evaluate model accuracy	   
		score = self.tf_model.evaluate(self.normalized_trainset_x, self.trainset_y, verbose=0) #Compute training set accuracy
		#print('Train loss:', score[0])
		print('Train accuracy:', score[1])

		score = self.tf_model.evaluate(self.normalized_testset_x, self.testset_y, verbose=0) #Compute test set accuracy
		#print('Test loss:', score[0])
		print('Test accuracy:', score[1])

	def get_sample(self, idx): # Sample a user
		idx %= len(self.testset_x)
		print(f'Getting sample {idx} of {len(self.testset_x)}')
		return {
			'id': idx,
			'label': self.heloc_dataframe.columns[:-1].tolist(),
			'value': self.testset_x[idx].tolist()
		}

	def classify(self, X):
		normalized_X = self.normalize(X)
		normalized_X = normalized_X.reshape((1,) + normalized_X.shape)
		classifier_output = np.argmax(self.tf_model.predict_proba(normalized_X))
		return {
			'label': self.heloc_dataframe.columns[-1],
			'value': self.CLASS_NAMES[classifier_output],
		}

	# ### Step 3. Compute contrastive explanations for a few applicants
	# 
	# Given the trained NN model to decide on loan approvals, let us first examine an applicant whose application was denied and what (minimal) changes to his/her application would lead to approval (i.e. finding pertinent negatives). We will then look at another applicant whose loan was approved and ascertain features that would minimally suffice in him/her still getting a positive outcome (i.e. finding pertinent positives).
	def get_explainable_classification(self, X, classifier_output_label, arg_max_iter=10, arg_init_const=10.0, arg_b=9, arg_kappa=0.1, arg_beta=1e-1, arg_gamma=100, my_AE_model=None):
		'''
		normalized_X: normalized sample
		arg_mode: PN or PP; Find pertinent negatives
		arg_max_iter: Maximum number of iterations to search for the optimal PN for given parameter settings
		arg_init_const: Initial coefficient value for main loss term that encourages class change
		arg_b: No. of updates to the coefficient of the main loss term
		arg_kappa: Minimum confidence gap between the PNs (changed) class probability and original class' probability
		arg_beta: Controls sparsity of the solution (L1 loss)
		arg_gamma: Controls how much to adhere to a (optionally trained) auto-encoder
		my_AE_model: Pointer to an auto-encoder
		'''
		arg_mode = 'PP' if classifier_output_label == self.CLASS_NAMES[1] else 'PN'
		
		normalized_X = self.normalize(X)
		normalized_X = normalized_X.reshape((1,) + normalized_X.shape)
	
		# #### a. Compute Pertinent Negatives (PN): 
		# In order to compute pertinent negatives, the CEM explainer computes a user profile that is close to the original applicant but for whom the decision of HELOC application is different. The explainer alters a minimal set of features by a minimal (positive) amount. This will help the user whose loan application was initially rejected say, to ascertain how to get it accepted. 
		(adv_pn, delta_pn, info_pn) = (adv_pn, delta_pn, info_pn) = self.explainer.explain_instance(normalized_X, arg_mode, my_AE_model, arg_kappa, arg_b, arg_max_iter, arg_init_const, arg_beta, arg_gamma)
		if self.debug:
			print('Adversarial Pertinent Negatives:', adv_pn)

		# Let us start by examining one particular loan application that was denied for applicant 1272. We showcase below how the decision could have been different through minimal changes to the profile conveyed by the pertinent negative. We also indicate the importance of different features to produce the change in the application status. The column delta in the table below indicates the necessary deviations for each of the features to produce this change. A human friendly explanation is then provided based on these deviations following the feature importance plot.
		Xpn = adv_pn
		classes = [ 
			classifier_output_label,
			self.CLASS_NAMES[np.argmax(self.tf_model.predict_proba(Xpn))], 
			'NIL' 
		]

		X_re = self.rescale(normalized_X) # Convert values back to original scale from normalized
		Xpn_re = self.rescale(Xpn)
		Xpn_re = np.around(Xpn_re.astype(np.double), 2)

		delta_re = Xpn_re - X_re
		delta_re = np.around(delta_re.astype(np.double), 2)
		delta_re[np.absolute(delta_re) < 1e-4] = 0

		X3 = np.vstack((X_re, Xpn_re, delta_re))

		dfre = pd.DataFrame.from_records(X3) # Create dataframe to display original point, PN and difference (delta)
		dfre[23] = classes

		dfre.columns = self.heloc_dataframe.columns.tolist()
		dfre.rename(index={0:'normalized_X',1:'X_P', 2:'(X_P - normalized_X)'}, inplace=True)
		dfret = dfre.transpose()

		if arg_mode == 'PP':
			fi = abs(delta_re.astype('double'))/np.std(self.normalized_trainset_x.astype('double'), axis=0) # Compute PP feature importance
		else:
			fi = abs((normalized_X-Xpn).astype('double'))/np.std(self.normalized_trainset_x.astype('double'), axis=0) # Compute PN feature importance
		return {
			'contrastive_explanations': dfre.to_dict(orient="split"),
			'performance': fi[0].tolist(),
		}

# experiment = ExplainableLoanRiskClassifier(train=TRAIN_MODEL, process_data=PROCESS_DATA, debug=False)
# X = experiment.get_sample()
# classifier_output_label = experiment.classify(X['value'])['value']
# conclusions_dict = experiment.get_explainable_classification(X['value'], classifier_output_label)
# print('Premises:', X)
# print('Conclusions:', conclusions_dict)
