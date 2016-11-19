exec(compile(open("library.py","rb").read(),"library.py","exec"))

data = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
	names= ["age", "type_employer", "fnlwgt", "education", 
                              "education_num","marital", "occupation", "relationship", "race","sex",
                              "capital_gain", "capital_loss", "hr_per_week","country", "income"],
                              skipinitialspace=True)


#Data Dictionary
#age – The age of the individual
#workclass – The type of employer the individual has. Whether they are government, military, private, an d so on.
#fnlwgt – The \# of people the census takers believe that observation represents. We will be ignoring this variable
#education – The highest level of education achieved for that individual
#education_num – Highest level of education in numerical form
#marital – Marital status of the individual
#occupation – The occupation of the individual
#relationship – A bit more difficult to explain. Contains family relationship values like husband, father, and so on, but only contains one per observation. I’m not sure what this is supposed to represent
#race – descriptions of the individuals race. Black, White, Eskimo, and so on
#sex – Biological Sex
#capital_gain – Capital gains recorded
#capital_loss – Capital Losses recorded
#hr_per_week – Hours worked per week
#country – Country of origin for person
#income – Boolean Variable. Whether or not the person makes more than \$50,000 per annum income.

#official preprocessing (from Examples section)
del data["education_num"]
del data["fnlwgt"]
# removed the continous attribute fnlwgt (final weight).
# eliminated eduction-num because it is just a numeric representation of education

######### Check for missing values and remove those rows 
data[data["type_employer"] =="?"]= np.nan
data[data["occupation"] == "?"] = np.nan
data[data["country"] == "?"] = np.nan

# remove the rows which have NaN's
data = data.dropna(how = "all")

# Divide the data into train and test split.
train,test = train_test_split(data,test_size = 0.30, random_state = 20)

y_train = train["income"]
y_test = test["income"]

del train["income"]
del test["income"]

x_train = train 
x_test = test 

del train,test,data 