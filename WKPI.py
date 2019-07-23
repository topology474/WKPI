import numpy as np

class WKPI():
	
	def __init__(self, pimage_1, coordinates, classNum, labelList):
		# pimage_1: Persistence images in training dataset
		# coordinates: The coordinates of persistence image cell
		# classNum: Number of different classes in dataset
		# labelList: labelList[i]: The index of cases in the training dataset that belongs to the i_th class
		self.pimage_1 = pimage_1
		self.pdnum_1 = pimage_1.shape[0]
		self.coordinates = coordinates
		self.coordinatesLength = coordinates.shape[0]
		self.classNum = classNum
		self.labelList = labelList
		
	def computeWeight(self, weights, centers, sigma_for_weight):
		# Compute the value of mixture Gaussian functions for each persistence image cell
		# weights: weight of each Gaussian function; 
		# centers: the x,y coordinates of centers for each Gaussian function;
		# sigma_for_weight: sigma of each Gaussian function
		self.gaussianNum = weights.shape[0]
		self.weightGaussian = weights
		self.cetnersGaussian = centers
		self.sigmaGaussian = sigma_for_weight
		sum = np.zeros((self.coordinatesLength, 1))
		gaussianPixels = []
		for i in range(self.gaussianNum):
			centerCoordinate = np.tile(self.cetnersGaussian[i,:], (self.coordinatesLength, 1))
			gaussian = (np.exp(np.sum((self.coordinates - centerCoordinate) ** 2 / (- self.sigmaGaussian[i] ** 2), axis = 1))).reshape((self.coordinatesLength,1))
			gaussianPixels.append(gaussian)
			sum = sum + gaussian * self.weightGaussian[i]
		self.gaussianPixels = np.array(gaussianPixels)
		self.weightFunc = sum
	
	def computeGramMatrix(self, sigma_for_kernel):
		# Compute the training Gram matrix for svm classifier
		# sigma_for_kernel: sigma of kernels
		kernel = np.zeros((self.pdnum_1, self.pdnum_1))
		expMatrixList = []
		for i in range(self.coordinatesLength):
			pimage_1_ipixel = np.tile(self.pimage_1[:, i], (self.pdnum_1, 1))
			expMatrix = np.exp(-(pimage_1_ipixel.T - pimage_1_ipixel) ** 2 / (sigma_for_kernel ** 2))
			expMatrixList.append(expMatrix)
			kernel += expMatrix * self.weightFunc[i]
		self.GramMatrix = kernel
		self.expMatrixList = expMatrixList
	
	def computeTestGramMatrix(self, pimage_2, sigma_for_kernel):
		# Compute the test Gram matrix for svm classifier
		# pimage_2: The test persistence images
		# sigma_for_kernel: sigma of kernels
		pdnum_2 = pimage_2.shape[0]
		kernel = np.zeros((pdnum_2, self.pdnum_1))
		for i in range(self.coordinatesLength):
			pimage_1_ipixel = np.tile(self.pimage_1[:, i], (pdnum_2, 1))
			pimage_2_ipixel = np.tile(pimage_2[:, i], (self.pdnum_1, 1))
			expMatrix = np.exp(-(pimage_1_ipixel - pimage_2_ipixel.T) ** 2 / (sigma_for_kernel ** 2))
			kernel += expMatrix * self.weightFunc[i]
		self.testGramMatrix = kernel
	
	def computeMetricMatrix(self):	
		# Compute the metric matrix of the training persistence images
		self.metricMatrix = 2 * np.sum(self.weightFunc) - 2 * self.GramMatrix
	
	def computeLaplaceAndH(self):
		# Compute the Laplacian matrix and the matrix H which will be used further
		degree = np.diag(np.array([np.sum(self.GramMatrix[i]) for i in range(self.pdnum_1)]))
		self.LaplacianMatrix = degree - self.GramMatrix
		# sumClassCost[i] stores cost(i,.)
		self.sumClassCost = [np.sum(self.GramMatrix[self.labelList[i]]) for i in range(self.classNum)]		
		matrixH = np.zeros((self.classNum, self.pdnum_1))
		for i in range(self.classNum):
			class_i_List = self.labelList[i].tolist()
			vector_h_i = [1/(self.sumClassCost[i] ** 0.5) if j in class_i_List else 0.0 for j in range(self.pdnum_1)]
			matrixH[i] = np.array(vector_h_i)
		self.matrixH = matrixH
	
	def computeCostByLaplace(self):
		# Compute the cost function by k-Tr(HLH^T)
		trace = sum([np.dot(np.dot(self.matrixH[i], self.LaplacianMatrix), self.matrixH[i]) for i in range(self.classNum)])
		self.cost = self.classNum - trace
		
	def computeGradientsByLaplace(self, rate = 1.0):
		# Compute the gradients of cost function by k-Tr(HLH^T)
		gradient_w = []
		gradient_x = []
		gradient_y = []
		gradient_sigma = []
		for i in range(self.gaussianNum):
			# The gradients of pseudo metric matrix over parameters			
			gradient_metric_wi = np.zeros((self.pdnum_1, self.pdnum_1))
			gradient_metric_xi = np.zeros((self.pdnum_1, self.pdnum_1))
			gradient_metric_yi = np.zeros((self.pdnum_1, self.pdnum_1))
			gradient_metric_sigmai = np.zeros((self.pdnum_1, self.pdnum_1))
			for j in range(self.coordinatesLength):	
				const = 2 - 2 * self.expMatrixList[j]
				gradient_metric_wi += self.gaussianPixels[i][j] * const
				gradient_metric_xi += self.weightGaussian[i] * self.gaussianPixels[i][j] * ((self.cetnersGaussian[i][0] - self.coordinates[j][0])) * (-2) * const / (self.sigmaGaussian[i] ** 2)
				gradient_metric_yi += self.weightGaussian[i] * self.gaussianPixels[i][j] * ((self.cetnersGaussian[i][1] - self.coordinates[j][1])) * (-2) * const / (self.sigmaGaussian[i] ** 2)
				gradient_metric_sigmai += self.weightGaussian[i] * self.gaussianPixels[i][j] * sum((self.coordinates[j] - self.cetnersGaussian[i]) ** 2) * 2 * const / (self.sigmaGaussian[i] ** 3)
			# The gradients of degree matrix over parameters
			degree_wi = np.diag(np.sum(gradient_metric_wi, axis = 1))
			degree_xi = np.diag(np.sum(gradient_metric_xi, axis = 1))
			degree_yi = np.diag(np.sum(gradient_metric_yi, axis = 1))
			degree_sigmai = np.diag(np.sum(gradient_metric_sigmai, axis = 1))
			# The gradients of Laplacian matrix over parameters
			gradient_Lap_wi = degree_wi - gradient_metric_wi
			gradient_Lap_xi = degree_xi - gradient_metric_xi
			gradient_Lap_yi = degree_yi - gradient_metric_yi
			gradient_Lap_sigmai = degree_sigmai - gradient_metric_sigmai
			
			# The gradients of H over parameters
			gradient_H_wi = np.zeros((self.classNum, self.pdnum_1))
			gradient_H_xi = np.zeros((self.classNum, self.pdnum_1))
			gradient_H_yi = np.zeros((self.classNum, self.pdnum_1))
			gradient_H_sigmai = np.zeros((self.classNum, self.pdnum_1))
			for j in range(self.classNum):
				coefficient = -0.5 * self.sumClassCost[j] ** (-1.5)
				gradient_hj_wi = np.sum(gradient_metric_wi[self.labelList[j]]) * coefficient
				gradient_hj_xi = np.sum(gradient_metric_xi[self.labelList[j]]) * coefficient
				gradient_hj_yi = np.sum(gradient_metric_yi[self.labelList[j]]) * coefficient
				gradient_hj_sigmai = np.sum(gradient_metric_sigmai[self.labelList[j]]) * coefficient
				index = self.labelList[j].tolist()
				gradient_H_wi[j] += np.array([gradient_hj_wi if (each in index) else 0.0 for each in range(self.pdnum_1)])
				gradient_H_xi[j] += np.array([gradient_hj_xi if (each in index) else 0.0 for each in range(self.pdnum_1)])
				gradient_H_yi[j] += np.array([gradient_hj_yi if (each in index) else 0.0 for each in range(self.pdnum_1)])
				gradient_H_sigmai[j] += np.array([gradient_hj_sigmai if (each in index) else 0.0 for each in range(self.pdnum_1)])
			
			# The gradients of cost function over parameters
			gradient_wi = 0
			gradient_xi = 0
			gradient_yi = 0
			gradient_sigmai = 0
			for j in range(self.classNum):
				vector_h = self.matrixH[j]
				gradient_wi += (-1) * (np.dot(np.dot(vector_h, self.LaplacianMatrix), gradient_H_wi[j]) + np.dot(np.dot(gradient_H_wi[j], self.LaplacianMatrix), vector_h) + np.dot(np.dot(vector_h, gradient_Lap_wi), vector_h))
				gradient_xi += (-1) * (np.dot(np.dot(vector_h, self.LaplacianMatrix), gradient_H_xi[j]) + np.dot(np.dot(gradient_H_xi[j], self.LaplacianMatrix), vector_h) + np.dot(np.dot(vector_h, gradient_Lap_xi), vector_h))
				gradient_yi += (-1) * (np.dot(np.dot(vector_h, self.LaplacianMatrix), gradient_H_yi[j]) + np.dot(np.dot(gradient_H_yi[j], self.LaplacianMatrix), vector_h) + np.dot(np.dot(vector_h, gradient_Lap_yi), vector_h))
				gradient_sigmai += (-1) * (np.dot(np.dot(vector_h, self.LaplacianMatrix), gradient_H_sigmai[j]) + np.dot(np.dot(gradient_H_sigmai[j], self.LaplacianMatrix), vector_h) + np.dot(np.dot(vector_h, gradient_Lap_sigmai), vector_h))
			
			gradient_w.append(gradient_wi - 1 / np.exp(self.weightGaussian[i]) * rate)
			gradient_x.append(gradient_xi)
			gradient_y.append(gradient_yi)
			gradient_sigma.append(gradient_sigmai)
		
		self.gradientW = np.array(gradient_w)
		self.gradientX = np.array(gradient_x)
		self.gradientY = np.array(gradient_y)
		self.gradientS = np.array(gradient_sigma)
		
	def computeCost(self):
		# Compute the cost function by \sum_{t=1}^k cost(t,t)/cost(t,.)
		
		# Metric within class: cost(t,t)
		metricInClass = [np.sum(self.metricMatrix[self.labelList[i]][:, self.labelList[i]]) for i in range(self.classNum)]
		# Interclasses metric: cost(t, .)
		metricInterClass = [np.sum(self.metricMatrix[self.labelList[i]]) for i in range(self.classNum)]
		self.cost = np.sum(np.array(metricInClass) / np.array(metricInterClass))
		self.metricInClass = metricInClass
		self.metricInterClass = metricInterClass
		
	def computeGradients(self, rate = 1.0):
		# Compute the gradients by \sum_{t=1}^k cost(t,t)/cost(t,.)
		gradient_w = []
		gradient_x = []
		gradient_y = []
		gradient_sigma = []
		for i in range(self.gaussianNum):
			gradient_wi = 0
			gradient_xi = 0
			gradient_yi = 0
			gradient_sigmai = 0
			# The gradients of pseudo metric matrix over parameters			
			gradient_metric_wi = np.zeros((self.pdnum_1, self.pdnum_1))
			gradient_metric_xi = np.zeros((self.pdnum_1, self.pdnum_1))
			gradient_metric_yi = np.zeros((self.pdnum_1, self.pdnum_1))
			gradient_metric_sigmai = np.zeros((self.pdnum_1, self.pdnum_1))
			for j in range(self.coordinatesLength):
				const = 2 - 2 * self.expMatrixList[j]
				gradient_metric_wi += self.gaussianPixels[i][j] * const
				gradient_metric_xi += self.weightGaussian[i] * self.gaussianPixels[i][j] * ((self.cetnersGaussian[i][0] - self.coordinates[j][0])) * (-2) * const / (self.sigmaGaussian[i] ** 2)
				gradient_metric_yi += self.weightGaussian[i] * self.gaussianPixels[i][j] * ((self.cetnersGaussian[i][1] - self.coordinates[j][1])) * (-2) * const / (self.sigmaGaussian[i] ** 2)
				gradient_metric_sigmai += self.weightGaussian[i] * self.gaussianPixels[i][j] * sum((self.coordinates[j] - self.cetnersGaussian[i]) ** 2) * 2 * const / (self.sigmaGaussian[i] ** 3)
			for j in range(self.classNum):
				gradient_wi += (np.sum(gradient_metric_wi[self.labelList[j]][:, self.labelList[j]]) * self.metricInterClass[j] - np.sum(gradient_metric_wi[self.labelList[j]]) * self.metricInClass[j]) / self.metricInterClass[j] ** 2
				gradient_xi += (np.sum(gradient_metric_xi[self.labelList[j]][:, self.labelList[j]]) * self.metricInterClass[j] - np.sum(gradient_metric_xi[self.labelList[j]]) * self.metricInClass[j]) / self.metricInterClass[j] ** 2
				gradient_yi += (np.sum(gradient_metric_yi[self.labelList[j]][:, self.labelList[j]]) * self.metricInterClass[j] - np.sum(gradient_metric_yi[self.labelList[j]]) * self.metricInClass[j]) / self.metricInterClass[j] ** 2
				gradient_sigmai += (np.sum(gradient_metric_sigmai[self.labelList[j]][:, self.labelList[j]]) * self.metricInterClass[j] - np.sum(gradient_metric_sigmai[self.labelList[j]]) * self.metricInClass[j]) / self.metricInterClass[j] ** 2
			gradient_w.append(gradient_wi - 1 / np.exp(self.weightGaussian[i]) * rate)
			gradient_x.append(gradient_xi)
			gradient_y.append(gradient_yi)
			gradient_sigma.append(gradient_sigmai)
			
		self.gradientW = np.array(gradient_w)
		self.gradientX = np.array(gradient_x)
		self.gradientY = np.array(gradient_y)
		self.gradientS = np.array(gradient_sigma)
		
	def getCost(self):
		# Return the value of cost function
		return self.cost
		
	def getGradients(self):
		# Return the gradients of parameters
		return self.gradientW, self.gradientX, self.gradientY, self.gradientS
	
	def getGramMatrix(self):
		# Return the training Gram matrix
		return self.GramMatrix
		
	def getTestGramMatrix(self):
		# Return the test Gram matrix
		return self.testGramMatrix

				
			
		
			