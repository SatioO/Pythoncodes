# Data Leakage 
"""
https://www.quora.com/Whats-data-leakage-in-data-science
This Quora Page Answer the Question Correctly. Here we will build our
own Cross-Validation function so that we prevent the model from
the effect of Data Leakage


Effect: Cross-Validation results will be very high but the model
will Perform poorly on test data.

Small Note:
When you r creating features from variables like say 
 - mean of the income of household in a country or state and do 
 cross-validation, the mean of the income is calculated on the entire
 dataset.So, when you are doing cross-validation the train data has
 some data captured from the test data and this will give you 
 better results in the cross-validation but the model fails when you
 bring new data.

 Some types of data leakage include:
 1. Leaking test data into the training data
 2. Leaking the correct prediction or ground truth into the test data
 3. Leaking of information from the future into the past
 4. Retaining proxies for removed variables a model is restricted from knowing
 5. Reversing of intentional obfuscation, randomization or anonymization.
 6. Inclusion of data not present in the model's operational environment.
 7. Distorting information from samples outside of scope of the models intended use.
 8. Any of the above present in third party data joined to the training set.

"""

"""
# Process 
All the variables created will be on fly
Let original Data frame be 
x_train - Predictors: Age, Salary,State, Gender
          Class - loan_Status 
1. Define the Grid 
2. Design all Combinations
3. For each combination 
   - 1. Divide the data into required folds randomly (5
   - 2. Join Four Folds 
          Train      Test
         5 4 3 2  -   1
         5 4 3 1  -   2
         5 4 1 2  -   3
         5 1 2 3  -   4
         1 2 3 4  -   5
   - 3. Create the new features by combining the train data and then test 
   it on test data. 
   - 4. Average the resultant metric you intended to improve.
4. Print the result. 

This framework cannot be defined as a function as most of the 
Features you create will be different for different datasets and 
different according to people thinking 
"""
from sklearn import cross_validation 
from sklearn import tree 
import itertools
import numpy as np 
import pandas as pd 


n_fold = 5 # num of folds 
nrow = len(x_train) # Number of tupules 

# We will build a Decision tree using cross-validation to decide on
# max depth for a tree 
# This algorithm is for grid search 

# Here we are defining the grid for Decision Tree and Can be changed according to the need.
param_tree = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [1, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "criterion": ["gini", "entropy"]}

varNames = sorted(param_tree) # Get the Parameters you defined

# Pull all the combinations (2*3*3*3*2 = 108)
combinations = [dict(zip(varNames, prod)) for prod in itertools.product(*(param_tree[varName] for varName in varNames))]


CV_score = list() # Save all the scores here. You can store nested scores if need here

# For each possible combinations run the alogrithm to find which is the best fit.
for i in range(len(combinations)):
	np.random.shuffle(list(zip(x_train,y_train)))
	X_folds= np.array_split(x_train,n_fold) # Split the Predictor data into n_folds
	Y_folds= np.array_split(y_train,n_fold) # Split the Class data into n_folds
	scores = list() # The score 
	for k in range(n_folds): # for each fold
	    # convert the X_fold to list - each element contains one fold
		X_train = list(X_folds) 
		Y_train = list(Y_folds)
		# Pop one fold and make it a test data
		X_test  = X_train.pop(k)
		# Concatenate the remaining list into one data frame
		X_train = np.concatenate(X_train)
		"""
		 -Build your features on X_train and do all Kind of feature Processing
		 -Append those features to X_test and do all Kind of feature Processing 
		
		"""
		Y_test  = y_train.pop(k) # Outcome variable of test data
		Y_train = np.concatenate(Y_train) # Outcome variable of train data
		# Building the Decision Tree Classifier.
		ct = tree.DecisionTreeClassifier(max_depth=combinations[i]["max_depth"],
			max_features=combinations[i]["max_features"],
			min_samples_split=combinations[i]["min_samples_split"],
			min_samples_leaf=combinations[i]["min_samples_leaf"],
			criterion=combinations[i]["criterion"])

		# Fit the model and append the scores 
		scores.append(ct.fit(X_train,Y_train).score(X_test,Y_test)) # score you intended to find 
	CV_score.append({"mean":np.mean(scores),"sd":np.std(scores)})

Best_CV = max([x["mean"] for x in CV_score])
# The score can be anything and we can change it according to our scope
# 1. Accuracy
# 2. Kappa 
# 3. Sensitivity 
# 4. Specificity
# 5. AUC 
# 6. Any other metric of user choice which he can define on his own 
