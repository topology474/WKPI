# WKPI
WKPI: A kernel based on persistent homology

Implementation for paper: Learning Metrics for Persistence-based Summaries and Applications for Graph Classification


WKPI kernel is constructed from persistence images. Given two persistence images PI and PI', the WKPI between them is wkpi(PI, PI') = \sum_s w(s)k(PI(s), PI'(s)), where s is the persistence images cell, w(s) is the weight function on s, and k(., .) is a kernel between two cells. In order to computing the weight function w(s), a cost function and metric learning scheme is developed. Details are in Learning metric for persistence-based summaries and graph classification.

--------------------------------------

__init__.py, metricLearner.py and WKPI.py provides an example of WKPI SVM classifier. In this example, the weight function is set as a mixture Gaussian function, and the kernel between two persistence images cells is set as Gaussian kernel.

There are 3 input augments.

"-d" or "--pdpath" for persistence diagrams files path.

"-i" or "--pipath" for persistence images files path.

"-f" or "--framework" for choosing different training and test framework. Please input joint or separate.

You can run our provided data examples by:
>python __init__.py -d <persistence_diagram_file_path> -i <persistence_image_file_path> -f <joint/separate> 

or 

>python __init__.py --pdpath <persistence_diagram_file_path> --pipath <persistence_image_file_path> --framework <joint/separate>

------------------------------------------------------------

About augments:

There are 4 types of files.

The first is files recording persistence diagrams (or persistence points) of each object, corresponding to "-d" or "--pdpath".

The second is files recording persistence images generated from persistence diagrams of each objects, corresponding to "-i" or "--pipath".

The third is file recording the x-y coordinates of persistence image cells. In our provided data examples, it is in the same folder with persistence images.

The fourth is the file recording the class labels of each object. In our provided data examples, it is in the same folder with persistence images.

The centers in mixture Gaussian functions are initialized by kmeans. Persistence points in training dataset will be collected in advance, then the clustering centers of kmeans result should be set as the initialized centers.

---------------------------------------------------

There are two hyperparameters: the number of mixtures in weight function (denoted as k) and "\sigma" in Gaussian kernel between persistence images cells. Two frameworks are provided to choose hyperparameters for WKPI-SVM. 

"joint" is a K * L fold nested cross validation, the procedure is as follows:

1. Split the dataset into K folds at random.

2. For each fold k = 1,2,...,K: outer loop for evaluation

	2.1. Set fold k as TEST set. All other folds are set as TRAINVAL set.
	
	2.2. Split the TRAINVAL set into L folds at random.
	
	2.3. For each fold l = 1,2,...,L: inner loop for tuning hyperparameters
	
		2.3.1. Set fold l in TRAINVAL set as VALIDATION set. All other folds in TRAINVAL set are set as TRAIN set.
		
		2.3.2. Learn metrics and train SVM classifier with each hyperparameter on TRAIN, and evaluate it on VALIDATION.
		
		2.3.3. Record the validation performance of each hyperparameter setting.
		
	2.4. Choose the hyperparameter setting with the highest average accuracy in validation.
	
	2.5. Learn metrics and train SVM classifier with the selected hyperparameter on TRAINVAL, evaluate it on TEST. Record the scores.
	
3. The average accuracy among K folds is the performance of WKPI based SVM.

In our example, we set K and L as 10 and 10.




"separate" is a K fold cross validation, the procedure is as follows:

1. Split the dataset into K folds at random.

2. For each fold k = 1,2,...,K:

	2.1. Set fold k as TEST set. All other folds are set as TRAIN set.
	
	2.2. Learn metrics with each hyperparameter on TRAIN, record the cost function values.
	
	2.3. Choose the hyperparameter setting with the minimum cost value on TRAIN data set.
	
	2.4. Train SVM classifier with the selected hyperparameter setting on TRAIN, evaluate it on TEST.
	
3. The average accuracy among K folds is the performance of WKPI based SVM.

In our example, we set K as 10.



The main difference between two frameworks: 

The criterion for selecting hyperparameters in "joint" is the performance on classifying validation set, which means learning metrics and training classifier jointly. That in "seperate" is the cost function value on training set, which means metrics learning and classifier training are seperate.

------------------------------------------------


There are some other choices you can make in this example.

First, you can use the entire training set to learn metrics or just a subset of training set. See "batch" and "batch_size" in metricLearner.learnMetric.

Second, you can choose methods to compute the gradients in metric learning process. See "method" in metricLearner. learnMetric.

-----------------------------------------------


About data examples:

We provide two data sets: MUTAG and PROTEIN, which are popular benchmarks in graph classification task.

./mutagPD/ and ./proteinPD/ are persistence diagrams files.

./mutagPI/ and ./proteinPI/ are persistence images files. As we said above, the coordinates and label files are included as well. 

The filtrations in generating persistence points is Ricci curvature. Both 0-dimsional and 1-dimensional persistence points are included.
Note that in our provided persistence points, the death might be lower than birth, which is different from some persistent homology settings. The reason is we use both sub-level subset and super-level subset.
