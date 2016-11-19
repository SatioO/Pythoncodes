exec(compile(open("fe.py","rb").read(),"fe.py","exec"))
exec(compile(open("helper.py","rb").read(),"helper.py","exec"))

#############################Logisitic Regression 
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


LR = LogisticRegression()
LR.fit(x_train,y_train)
print(LR)

predicted = LR.predict(x_test)

#summarize the fit of the model 
print (metrics.classification_report(y_test,predicted))
print(metrics.confusion_matrix(y_test,predicted))

########################## without relationship 

LR = LogisticRegression()
LR.fit(x_train.drop(["relationship"],axis=1),y_train)
print(LR)

predicted = LR.predict(x_test.drop(["relationship"],axis=1))

#summarize the fit of the model 
print (metrics.classification_report(y_test,predicted))
print(metrics.confusion_matrix(y_test,predicted))


########################## Random Forest (check in helper file for functions)
rfc = RandomForestClassifier(n_estimators = 20)

# run randomized search 
start = time()
n_iter_search = 100
random_search = RandomizedSearchCV(rfc,param_distributions=param_dist,
	n_iter =n_iter_search)
random_search.fit(x_train,y_train)
start-time()

report(random_search.grid_scores_) # Choose among the best

# Build the random forest  
rfc = RandomForestClassifier(n_estimators=1000,
	max_depth= None, 
	min_samples_leaf= 6, 
	max_features= 3, 
	criterion= 'gini', 
	min_samples_split= 9, 
	bootstrap= False)
rfc.fit(x_train,y_train)

predicted = rfc.predict(x_test)

#summarize the fit of the model 
print (metrics.classification_report(y_test,predicted))
print(metrics.confusion_matrix(y_test,predicted))

rfc.feature_importances_ # Feature Importance 

########################## CART Trees 
from sklearn import tree 

clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train,y_train)

predicted = clf.predict(x_test)

#summarize the fit of the model 
print (metrics.classification_report(y_test,predicted))
print(metrics.confusion_matrix(y_test,predicted))


############################ Bagging logistic Regression
BC = BaggingClassifier(base_estimator = LogisticRegression())

start = time()
n_iter_search = 100
random_search = RandomizedSearchCV(BC,param_distributions=param_bag,
	n_iter =n_iter_search)
random_search.fit(x_train,y_train)
start-time()

BC = BaggingClassifier(base_estimator= LogisticRegression())
BC = BC.fit(x_train,y_train)

## Summarize the fit of the model 
print (metrics.classification_report(y_test,predicted))
print (metrics.confusion_matrix(y_test,predicted))


############################ gradient Boosting machines 
GBM = GBC()

start = time()
n_iter_search = 100
random_search = RandomizedSearchCV(GBM,param_distributions=param_gbm,
	cv = 5,
	n_iter =n_iter_search)
random_search.fit(x_train,y_train)
start-time()



report(random_search.grid_scores_)
GBM = BaggingClassifier({
	n_estimators=100,
	max_depth=3,
	max_features=8,
	max_leaf_nodes=5,
	min_samples_leaf=6,
	min_samples_split=5,
	learning_rate=0.1,
	min_weight_fraction_leaf=0.2
})
GBM = GBM.fit(x_train,y_train)

## Summarize the fit of the model 
print (metrics.classification_report(y_test,predicted))
print (metrics.confusion_matrix(y_test,predicted))

################################ Stacking 


############################### Xgboost 