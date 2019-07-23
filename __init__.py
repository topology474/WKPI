import os
import random
import numpy as np
import sys, getopt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from WKPI import *
from metricLearner import *


def initialization(k, pdiagram):
	# Initialization for mixture Gaussian weight functions
	# k: The number of mixtures in mixture Gaussians
	# pdiagram: The persistence points in the training data set
	# weights: weight of each Gaussian function; 
	# centers: the x,y coordinates of centers for each Gaussian function;
	# sigma_for_weight: sigma of each Gaussian function
	kmeans = KMeans(n_clusters = k, random_state = 0).fit(pdiagram)
	centers = kmeans.cluster_centers_
	weights = np.array([1.0] * k)
	sigma_for_weight = np.array([0.1] * k)
	return weights, centers, sigma_for_weight

def wkpiSVM(pimageTrain, pimageTest, coordinates, classNum, labelTrain, weights, centers, sigma_for_weight, sigma_for_kernel):
	# Using wkpi kernel based SVM to predict labels
	# pimageTrain, pimageTest: The persistence images in training / test dataset
	# classNum: Number of different classes in dataset
	# labelTrain: Label of persistence images in training dataset
	# weights: weight of each Gaussian function; 
	# centers: the x,y coordinates of centers for each Gaussian function;
	# sigma_for_weight: sigma of each Gaussian function
	# sigma_for_kernel: sigma_for_kernel: sigma of kernels
	
	# Learning metrics
	metriclearner = metricLearner(pimageTrain, coordinates, classNum, labelTrain)
	weights, centers, sigma_for_weight, cost = metriclearner.learnMetric(weights, centers, sigma_for_weight, sigma_for_kernel)
	labelList = [np.where(labelTrain == i)[0] for i in range(classNum)]
	# Train and test on WKPI kernel
	wkpi_train = WKPI(pimageTrain, coordinates, classNum, labelList)
	wkpi_train.computeWeight(weights, centers, sigma_for_weight)
	wkpi_train.computeGramMatrix(sigma_for_kernel)
	wkpi_train.computeTestGramMatrix(pimageTest, sigma_for_kernel)
	clf = SVC(kernel = 'precomputed')
	clf.fit(wkpi_train.getGramMatrix(), labelTrain)
	label_pred = clf.predict(wkpi_train.getTestGramMatrix())
	return label_pred

def main(argv):
	# Input the file path of persistence diagrams and persistence images. 
	# "-d" or "--pdpath" for persistence diagrams files path.
	# "-i" or "--pipath" for persistence images files path.
	# "-f" or "--framework" for choosing different training and test framework. 
	pdiagram_path = ""
	pimage_path = ""
	learning_framework = ""
	try:
		opts, args = getopt.getopt(argv,"hd:i:f:", ["help", "pdpath=", "pipath=", "learning_framework="])
	except getopt.GetoptError:
		print('Error: metricLearn.py -d <pdiagram_path> -i <pimage_path>')
		print('or: metricLearn.py --pdpath=<pdiagram_path> --pipath=<pimage_path>')
		sys.exit(2)

	for opt, arg in opts:
		if opt in ("-h", "--help"):
			print("metricLearn.py -d <pdiagram_path> -i <pimage_path>")
			print('or: metricLearn.py --pdpath=<pdiagram_path> --pipath=<pimage_path>')
			sys.exit(2)
		elif opt in ("-d", "--pdpath"):
			pdiagram_path = arg
		elif opt in ("-i", "--pipath"):
			pimage_path = arg
		elif opt in ("-f", "--framwork"):
			learning_framework = arg
			if learning_framework not in ["joint", "separate"]:
				print("The learning framework should be 'joint' or 'framework'.")
				sys.exit(2)
	
	# Put persistence points of all persistence diagrams into one list, persistence_points
	pdNum = len(os.listdir(pdiagram_path))
	persistence_points = [np.loadtxt(pdiagram_path + str(i) + "_PD.pdg") for i in range(pdNum)]
			
	# Put the persistence images into one list, persistence_images
	persistence_images = np.array([np.loadtxt(pimage_path + str(i) + "_PI.pdg") for i in range(pdNum)])
	
	# Put the graph labels into one list, labels
	label_information = np.loadtxt(pimage_path + "labels.txt")
	labels = np.array([int(label_information[i]) for i in range(pdNum)])
	classNum = len(set(labels.tolist()))
	
	# Load information about x,y coordinates
	coordinates = np.loadtxt(pimage_path + "coordinates.txt")
	
	# The number of Gaussians in the mixture Gaussian weight function
	kNum = [3, 4, 5]
	sigma_list = [0.1, 1]
	hyperparameters = [(k, sigma_for_kernel) for k in kNum  for sigma_for_kernel in sigma_list]
	
	if (learning_framework == "separate"):
		kf = KFold(n_splits = 10, shuffle = True)
		scoreList = []
		for train_index, test_index in kf.split(persistence_images):
			pimage_train, pimage_test = persistence_images[train_index], persistence_images[test_index]
			label_train, label_test = labels[train_index], labels[test_index]
			pdiagram_train = [persistence_points[each] for each in train_index.tolist()]
			persistencePoints_train = []
			for each in pdiagram_train:
				if (each.shape[0] == 1):
					persistencePoints_train = persistencePoints_train + [each.tolist()]
				persistencePoints_train = persistencePoints_train + each.tolist()
			normalizer = preprocessing.Normalizer().fit(pimage_train)
			pimage_train_stand = normalizer.transform(pimage_train)
			pimage_test_stand = normalizer.transform(pimage_test)
			costList = []
			paramList = []
			for k, sigma_for_kernel in hyperparameters:
				weights, centers, sigma_for_weight = initialization(k, persistencePoints_train)
				metriclearner = metricLearner(pimage_train_stand, coordinates, classNum, label_train)
				weights, centers, sigma_for_weight, cost = metriclearner.learnMetric(weights, centers, sigma_for_weight, sigma_for_kernel)
				costList.append(cost)
				paramList.append((weights, centers, sigma_for_weight))
			minCostIndex = costList.index(min(costList))
			weights, centers, sigma_for_weight = paramList[minCostIndex]
			k, sigma_for_kernel = hyperparameters[minCostIndex]
			labelList = [np.where(label_train == i)[0] for i in range(classNum)]
			wkpi = WKPI(pimage_train_stand, coordinates, classNum, labelList)
			wkpi.computeWeight(weights, centers, sigma_for_weight)
			wkpi.computeGramMatrix(sigma_for_kernel)
			wkpi.computeTestGramMatrix(pimage_test_stand, sigma_for_kernel)
			clf = SVC(kernel = 'precomputed')
			clf.fit(wkpi.getGramMatrix(), label_train)
			label_pred = clf.predict(wkpi.getTestGramMatrix())
			result = accuracy_score(label_test, label_pred)
			scoreList.append(result)
			print("The test result with k=%d and sigma=%f is %f" %(k, sigma_for_kernel, result))
		print("The accuracy on each test fold:")
		print(scoreList)	
	else:	
		outKF = KFold(n_splits = 10, shuffle = True)
		scoreList = []
		for train_val_index, test_index in outKF.split(persistence_images):
		# The outer loop of cross_validation
			pimage_train_val, pimage_test = persistence_images[train_val_index], persistence_images[test_index]
			label_train_val, label_test = labels[train_val_index], labels[test_index]
			pdiagram_train_val = [persistence_points[each] for each in train_val_index.tolist()]
			val_result = np.zeros((len(hyperparameters), 1))
			innerKF = KFold(n_splits = 10, shuffle = True)
			for train_index, val_index in innerKF.split(pimage_train_val):
			# The inner loop of cross_validation
				pimage_train, pimage_val = pimage_train_val[train_index], pimage_train_val[val_index]
				label_train, label_val = label_train_val[train_index], label_train_val[val_index]
				pdiagram_train = []
				for each in train_index.tolist():
					if(pdiagram_train_val[each].shape[0] == 1):
						pdiagram_train = pdiagram_train + [pdiagram_train_val[each].tolist()]
					pdiagram_train = pdiagram_train + pdiagram_train_val[each].tolist()
				normalizer = preprocessing.Normalizer().fit(pimage_train)
				pimage_train_stand = normalizer.transform(pimage_train)
				pimage_val_stand = normalizer.transform(pimage_val)
				innerLoopResult = []
				for k,sigma_for_kernel in hyperparameters:
					weights, centers, sigma_for_weight = initialization(k, pdiagram_train)
					label_pred = wkpiSVM(pimage_train_stand, pimage_val_stand, coordinates, classNum, label_train, weights, centers, sigma_for_weight, sigma_for_kernel)
					result = accuracy_score(label_val, label_pred)
					print("The validation result with k=%d and sigma=%f is %f" %(k, sigma_for_kernel, result))
					innerLoopResult.append(result)
				val_result += np.array(innerLoopResult).reshape((len(hyperparameters), 1))
					
			k, sigma_for_kernel = hyperparameters[np.where(val_result == np.max(val_result))[0][0]]
			pdiagram_trainval = []
			for each in pdiagram_train_val:
				if (each.shape[0] == 1):
					pdiagram_trainval = pdiagram_trainval + [each.tolist()]
				pdiagram_trainval = pdiagram_trainval + each.tolist()
			weights, centers, sigma_for_weight = initialization(k, pdiagram_trainval)
			normalizer = preprocessing.Normalizer().fit(pimage_train_val)
			pimage_trainval_stand = normalizer.transform(pimage_train_val)
			pimage_test_stand = normalizer.transform(pimage_test)
			label_pred = wkpiSVM(pimage_trainval_stand, pimage_test_stand, coordinates, classNum, label_train_val, weights, centers, sigma_for_weight, sigma_for_kernel)
			result = accuracy_score(label_test, label_pred)
			scoreList.append(result)
			print("The test result is %f" %result)
			print("------------------------------------")
		print("The accuracy on each test fold:")
		print(scoreList)
if __name__ == "__main__":
	main(sys.argv[1:])