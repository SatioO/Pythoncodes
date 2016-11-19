## helper 
## Random forest 
param_dist ={
	"max_depth":[3,None],
	"max_features":sp_randint(1,11),
	"min_samples_split":sp_randint(1,11),
	"min_samples_leaf":sp_randint(1,11),
	"bootstrap":[True,False],
	"criterion":["gini","entropy"]
}


def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


param_bag ={
	"max_features":sp_randint(1,12),
	"bootstrap":[True,False],
	"bootstrap_features":[False,True]
}


param_gbm ={
	"n_estimators":[50,100,150,200],
	"max_depth":[3,None],
	"max_features":sp_randint(1,11),
	"max_leaf_nodes":sp_randint(3,11),
	"min_samples_leaf":sp_randint(1,11),
	"min_samples_split":sp_randint(1,11),
	"learning_rate":np.arange(0.1,0.6,0.1),
	"min_weight_fraction_leaf":np.arange(0,0.6,0.1)
}
