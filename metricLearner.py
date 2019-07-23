import random
import numpy as np
from WKPI import *

class metricLearner:
	def __init__(self, pimage, coordinates, classNum, labels):
		# pimage: Persistence images in training dataset
		# coordinates: The coordinates of persistence image cell
		# classNum: Number of different classes in dataset
		# labels: Label of persistence images in training dataset
		self.pimage = pimage
		self.coordinates = coordinates
		self.classNum = classNum
		self.labels = labels
	
	def computeGradients(self, weights, centers, sigma_for_weight, sigma_for_kernel, method, pimage, labelList):
		# Compute the gradients of parameters
		# weights: weight of each Gaussian function; 
		# centers: the x,y coordinates of centers for each Gaussian function;
		# sigma_for_weight: sigma of each Gaussian function
		# sigma_for_kernel: sigma_for_kernel: sigma of kernels
		# method: True: compute gradients by \sum_{t=1}^k cost(t,t)/cost(t,.); False: compute gradients by k-Tr(HLH^T)
		# pimage: Persistence images
		# labelList: labelList[i]: The index of cases in the training dataset that belongs to the i_th class
		wkpi_kernel = WKPI(pimage, self.coordinates, self.classNum, labelList)
		wkpi_kernel.computeWeight(weights, centers, sigma_for_weight)
		wkpi_kernel.computeGramMatrix(sigma_for_kernel)
		wkpi_kernel.computeMetricMatrix()
		if(method):
			wkpi_kernel.computeCost()
			wkpi_kernel.computeGradients()
		else:
			wkpi_kernel.computeLaplaceAndH()
			wkpi_kernel.computeCostByLaplace()
			wkpi_kernel.computeGradientsByLaplace()
		cost = wkpi_kernel.getCost()
		gradients = wkpi_kernel.getGradients()
		
		return cost, gradients
	
	def learnMetric(self, weights, centers, sigma_for_weight, sigma_for_kernel, batch = True, batch_size = 50, method = False, max_iter = 1000, max_epsilon = 10 * (-4), stop_flag = 5, learning_rate = 0.01):
		# weights: weight of each Gaussian function; 
		# centers: the x,y coordinates of centers for each Gaussian function;
		# sigma_for_weight: sigma of each Gaussian function
		# sigma_for_kernel: sigma_for_kernel: sigma of kernels
		# batch: True: use minibatch methods; False: use the entire training dataset for learning
		# method: True: compute gradients by \sum_{t=1}^k cost(t,t)/cost(t,.); False: compute gradients by k-Tr(HLH^T)
		# max_iter: The max iteration times
		# max_epsilon, stop_flag: If the values of cost function among 'stop_flag' times iteration differ each other within max_epsilon, then stop iteratons
		oldCost = 0
		newCost = 0
		flag = 0
		for iter in range(max_iter):
			if(batch) and (batch_size < self.pimage.shape[0]):
				batch_index = random.sample(range(self.pimage.shape[0]), batch_size)
				batch_pimage = self.pimage[batch_index]
				batch_label = self.labels[batch_index]
			else:
				batch_pimage = self.pimage
				batch_label = self.labels
			labelList = [np.where(batch_label == i)[0] for i in range(self.classNum)]
			[newCost, gradients] = self.computeGradients(weights, centers, sigma_for_weight, sigma_for_kernel, method, batch_pimage, labelList)
			epsilon = abs(newCost - oldCost)
			oldCost = newCost
			if(epsilon <= max_epsilon):
				flag += 1
			else:
				flag = 0
			weights -= learning_rate * gradients[0]
			centers[:,0] -= learning_rate * gradients[1]
			centers[:,1] -= learning_rate * gradients[2]
			sigma_for_weight -= learning_rate * gradients[3]
			if(flag >= stop_flag):
				break
		
		return weights, centers, sigma_for_weight, newCost
			
		
		