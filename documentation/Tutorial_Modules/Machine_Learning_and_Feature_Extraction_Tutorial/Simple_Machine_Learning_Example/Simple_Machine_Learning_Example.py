import csv
import math

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg, ndimage

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from scipy.stats import pearsonr, spearmanr
from mpl_toolkits.mplot3d import Axes3D
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, BaggingRegressor, BaggingClassifier
from sklearn import datasets, svm
from sklearn.decomposition import PCA, KernelPCA
from sklearn import metrics
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth, AffinityPropagation
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale, normalize, PolynomialFeatures, robust_scale

def RandomForest_Classifier():

	np.set_printoptions(precision=3, suppress=True)
	data_title = 'LungData'

	Data = np.loadtxt('../Lung_Challenge_Features/Lung_Feature_Results_Final.csv', delimiter=",", skiprows=1, dtype="object")

	# Randomly divide the data according to a .5 proportion into train and test
	# In order to get the same random split, make sure that the np.random.seed
	# line is left uncommented.
	# np.random.seed(0)

	TrainData = np.zeros((0, Data.shape[1]))
	TestData = np.zeros((0, Data.shape[1]))

	for endpoint in np.arange(100, 600, 100):
		TempData = Data[endpoint-100:endpoint]
		random_mask = np.random.rand(len(TempData)) < .5
		TrainData = np.vstack((TrainData, TempData[random_mask]))
		TestData = np.vstack((TestData, TempData[~random_mask]))

	np.random.shuffle(TrainData)
	np.random.shuffle(TestData)

	dimsT = np.shape(TestData)
	dims = np.shape(TrainData)
	
	yT =  TestData[:,1].astype(float)
	XT = TestData[:,2:].astype(float)

	y =  TrainData[:,1].astype(float)
	X = TrainData[:,2:].astype(float)

	# X = normalize(robust_scale(X, axis=1), axis=1)
	# XT = normalize(robust_scale(XT, axis=1), axis=1)

	X = robust_scale(normalize(X, axis=1), axis=1)
	XT = robust_scale(normalize(XT, axis=1), axis=1)

	best = [np.inf, np.inf, np.inf]
	Results = np.zeros((5, 2))
	Predictions = np.zeros((dimsT[0],2))
	Predictions[:,0] = yT
	Results[:,0] = [1,2,3,4,5]
	Winner = []
	ParameterTuner = 20

	# for i in range(1,ParameterTuner+1):
	for i in [15]:
		# Some different machine learning options to test..

		for k in [[1, RandomForestClassifier(n_estimators=i)]]:
		# for k in [[1, RandomForestClassifier(n_estimators=(int(math.ceil(float(i)/10))), n_jobs=-1)]]:
		# for k in [[1, svm.LinearSVC(C=100,dual=True)]]:
		# for k in [[1, svm.SVC(C=10,kernel="rbf", degree=2)]]:
		# for k in [[1, BaggingRegressor(base_estimator=RandomForestRegressor(n_estimators=10), n_jobs=-1, n_estimators=20)]]:
		# for k in [[1, svm.NuSVC(nu=(.3), kernel="rbf", verbose=True, probability=True, tol=1e-6, decision_function_shape='ovr')]]:

			print 'Parameter Value: ' + str(i)

			clf = k[1].fit(X, y)

			XP = clf.predict(X)
			XTP = clf.predict(XT)
			RTP = np.random.choice(yT, dims[0])

			TempResults = np.zeros((5, 2))

			for category in range(1,6):

				print 'Category: ' + str(category)
				# Assess the accuracy on training and testing data.
				# Also compare against a random model.
				count = [0, 0, 0]
				total = [0, 0, 0]
				actual = [y, yT, yT]
				predicted = [XP, XTP, RTP]
				prediction_labels = ['Training','Testing','Random']

				for j in range(0, min(dims[0],dimsT[0])):
					for TestTrainRandom in xrange(3):
						if actual[TestTrainRandom][j] == category:
							if str(actual[TestTrainRandom][j]) != str(predicted[TestTrainRandom][j]):
								count[TestTrainRandom] += 1
							total[TestTrainRandom] += 1

				if not 0 in total:
					for ptype in xrange(3):
						print prediction_labels[ptype] + ' Error Rate: ' + str(float(count[ptype]) / total[ptype])
					print ""
					TempResults[category-1,k[0]] = float(count[1]) / total[1]

			if np.mean(TempResults[:,k[0]]) < best[k[0]]:
				best[k[0]] = np.mean(TempResults[:,k[0]])
				# Results[k[0]+2] = np.std(TempResults[:,k[0]])
				Results[:,k[0]] = TempResults[:,k[0]]
				Predictions[:,k[0]] = XTP
				Winner += [i]

	Results[:,1] = 1 - Results[:,1]
	print 'Best Parameters: '
	print Winner
	print 'Accuracy per Category: '
	print Results

	np.savetxt('Results' + data_title + '.csv', Results, fmt="%s", delimiter=",")
	np.savetxt('Predictions' + data_title + '.csv', Predictions, fmt="%s", delimiter=",")

	print "All done!"

if __name__ == "__main__":
	RandomForest_Classifier()