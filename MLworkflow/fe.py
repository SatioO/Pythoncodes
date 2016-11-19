exec(compile(open("data.py","rb").read(),"data.py","exec"))


# feature engineering 
ntrain = x_train.shape[0]
train_test = x_train.append(x_test)

#type_employer
train_test["type_employer"].value_counts()
# Given “Never worked” and “Without-Pay” are both very small groups, and they are 
# likely very similar, we can combine them to form a “Not Working” Category. 
# In a similar vein, we can combine government employee categories, and self-employed 
# categories. This allows us to reduce the number of categories significantly.
train_test["type_employer"] = train_test["type_employer"].map({
	"Federal-gov":"Federal-Govt",
	"Local-gov":"Other-Govt",
	"State-gov":"Other-Govt",
	"Private":"Private",
	"Self-emp-inc":"Self-Employed",
	"Self-emp-not-inc":"Self-Employed",
	"Without-pay":"Not-Working"
	})


#Occupation 
train_test["occupation"] = train_test["occupation"].map({
	"Adm-clerial":"Admin",
	"Armed-Forces":"Military",
	"Craft-repair":"Blue-Collar",
	"Exec-managerial":"White-Collar",
	'Other-service':"Service",
    'Handlers-cleaners':"Blue-Collar",
    'Tech-support':"Other-Occupations", 
    'Sales':"Sales",
    'Machine-op-inspct':"Blue-Collar", 
    'Farming-fishing':"Blue-Collar", 
    'Prof-specialty':"Professional",
    'Transport-moving':"Blue-Collar", 
    'Protective-serv':"Other-Occupations", 
    'Priv-house-serv':"Service",
	})

#Country
train_test["country"]= train_test["country"].map({
	'United-States':"United-States",
	'Jamaica':"Latin-America",
	'Germany':"Euro_1", 
	'Puerto-Rico':"Latin-America", 
	'Portugal':"Euro_2",
    'England':"British-Commonwealth", 
    'Japan':"Other", 
    'Guatemala':"Latin-America", 
    'El-Salvador':"South-America", 
    'Iran':"Other", 
    'Ecuador':"South-America",
    'China':"China", 
    'Mexico':"Latin-America", 
    'Ireland':"British-Commonwealth", 
    'India':"British-Commonwealth", 
    'South':"Euro_2", 
    'Haiti':"Latin-America",
    'Philippines':"SE-Asia", 
    'Poland':"Euro_2", 
    'Nicaragua':"Latin-America", 
    'Canada':"British-Commonwealth", 
    'Thailand':"SE-Asia",
    'Greece':"Euro_2", 
    'Cuba':"Other", 
    'Columbia':"South-America", 
    'Scotland':"British-Commonwealth", 
    'Italy':"Euro_1", 
    'Hungary':"Euro_2",
    'Vietnam':"SE-Asia", 
    'Peru':"South-America", 
    'Trinadad&Tobago':"Latin-America", 
    'France':"Euro_1",
    'Dominican-Republic':"Latin-America", 
    'Cambodia':"SE-Asia", 
    'Yugoslavia':"Euro_2", 
    'Laos':"SE-Asia", 
    'Taiwan':"China",
    'Hong':"China", 
    'Honduras':"Latin-America", 
    'Outlying-US(Guam-USVI-etc)':"Latin-America",
    'Holand-Netherlands':"Euro_1"
	})


#education
train_test["education"] = train_test["education"].map({
	'Masters':"Masters", 
	'Assoc-acdm':"Associates", 
	'Some-college':"HS-Graduate", 
	'HS-grad':"HS-Graduate", 
	'Bachelors':"Bachelors",
    'Assoc-voc':"Associates", 
    '11th':"Dropout", 
    'Doctorate':"Doctorate", 
    '1st-4th':"Dropout", 
    '7th-8th':"Dropout", 
    '10th':"Dropout",
    '9th':"Dropout", 
    '5th-6th':"Dropout", 
    '12th':"Dropout", 
    'Prof-school':"Prof-School", 
    'Preschool':"Dropout"
	})

#marital
train_test["marital"] = train_test["marital"].map({
	'Divorced':"Not-Married", 
	'Never-married':"Never-married", 
	'Married-civ-spouse':"Married", 
	'Separated':"Not-Married",
    'Widowed':"Widowed", 
    'Married-AF-spouse':"Married", 
    'Married-spouse-absent':"Not-Married"
	})


#race 
train_test["race"] = train_test["race"].map({
	'White':"White", 
	'Black':"Black", 
	'Asian-Pac-Islander':"Asian", 
	'Amer-Indian-Eskimo':"Amer-Indian",
    'Other':"Other"
	})


########################## Label encoding 
cat_columns = train_test.select_dtypes(["object"]).columns

for f in cat_columns:
	train_test[f] = train_test[f].astype("category")
	train_test[f]=train_test[f].cat.codes


########################### Splitting the datasets back 
x_train = train_test[:ntrain]
x_test = train_test[ntrain:]

y_train = y_train.map({">50K":1,"<=50K":0})
y_test = y_test.map({">50K":1,"<=50K":0})

########################### Preparing for xgboost 
#dtrain = xgb.DMatrix(x_train.as_matrix(),label=y_train.as_matrix())
#dtest = xgb.DMatrix(x_test.as_matrix())

#del train_test,cat_columns
#print ("Feature Engineering completed")










