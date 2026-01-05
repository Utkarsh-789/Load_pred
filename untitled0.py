import pandas as pd                       # for reading the files
import numpy as np                        # for creating multi-dimensional-array
import matplotlib.pyplot as plt           # for plotting
import seaborn as sns                     # for data visulization
import warnings  
from statistics import mean 
from sklearn import tree
from sklearn import metrics# for ignoring the warnings
warnings.filterwarnings("ignore") %matplotlib inline
from sklearn.metrics import classification_report
import scikitplot as skplt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold



# # Import the Data Files


test= pd.read_csv('C:/Users/utkar/Downloads/Loan-Prediction-System-master/test.csv')
train= pd.read_csv('C:/Users/utkar/Downloads/Loan-Prediction-System-master/train.csv')



# Test File


test.head()

test.shape

# Training File


train.head()

train.shape



# Creating a copy of file so that any changes made doesn't affect the original datasets


test_original= test.copy()
train_original= train.copy()



# Checking the Data Types of Variables


test.dtypes

train.dtypes



# # Univariant Analysis
# (Examing each variable individually)


# 1. Target Variable i.e. 'Loan Status'


train['Loan_Status'].value_counts()                    #counting the values of different Loan Status

train['Loan_Status'].value_counts().plot.bar()         

train['Loan_Status'].value_counts(normalize=True).plot.bar()
# normalize = True will give the probability in y-axis

plt.title("Loan Status")



# Plots for Independent Categorical Variables


plt.figure()
plt.subplot(321)
train['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,20),title='Gender')

plt.subplot(322)
train['Married'].value_counts(normalize=True).plot.bar(figsize=(20,20),title='Married')

plt.subplot(323)
train['Education'].value_counts(normalize=True).plot.bar(figsize=(20,20),title='Education')

plt.subplot(324)
train['Self_Employed'].value_counts(normalize=True).plot.bar(figsize=(20,20),title='Self-Employed')

plt.subplot(325)
train['Credit_History'].value_counts(normalize=True).plot.bar(figsize=(20,20),title='Credit_History')

# It can be inferred from the above bar plots that:
# 
# •	80% applicant in the dataset are male.
# 
# •	Around 65% of the applicants in the dataset are married.
# 
# •	Around 15% of the applicants in the dataset are self-employed.
# 
# •	Around 85% applicants have repaid their debts.
# 






# Plots for Independent Ordinal Variables


plt.figure()
plt.subplot(121)
train['Dependents'].value_counts(normalize=True).plot.bar(figsize=(20,5),title='Dependents')

plt.subplot(122)
train['Property_Area'].value_counts(normalize=True).plot.bar(figsize=(20,5),title='Property Area')

#  Following inferences can be made from the above bar plots:
#  
# •	Most of the applicant do not have any dependents.
# 
# •	Around 80% of the applicants are graduates.
# 
# •	Most of the applicant are from Semiurban area.
# 






# Plots for Independent Numerical Variables


# Applicant Income


plt.subplot(121)
sns.distplot(train['ApplicantIncome'])
plt.subplot(122)
train['ApplicantIncome'].plot.box(figsize=(20,5))



train.boxplot(column='ApplicantIncome',by='Education')
plt.suptitle("")

# It can be inferred that most of the data in the distribution of applicant income is towards left which is not normally distributed, and the boxplot confirms the presence of a lot of extreme values/outliers. This can be attributed to the income disparity in the society. 




# Co-applicant Income


plt.subplot(121)
sns.distplot(train['CoapplicantIncome'])
plt.subplot(122)
train['CoapplicantIncome'].plot.box(figsize=(20,5))



df=train.dropna()
plt.subplot(121)
sns.distplot(df['LoanAmount'])

plt.subplot(122)
df['LoanAmount'].plot.box(figsize=(20,5))



# # Bivariant Analysis
# (Examing two variables at a time)


# Frequency Table for Gender and Loan Status


Gender=pd.crosstab(train['Gender'],train['Loan_Status']) 
Gender

Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))



# Frequency Table for Married and Loan Status


Married=pd.crosstab(train['Married'],train['Loan_Status']) 
Married

Married.div(Married.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))



# Frequency Table for Dependents and Loan Status


Dependents=pd.crosstab(train['Dependents'],train['Loan_Status']) 
Dependents

Dependents.div(Dependents.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))



# Frequency Table for Education and Loan Status


Education= pd.crosstab(train['Education'],train['Loan_Status'])
Education

Education.div(Education.sum(1).astype(float),axis=0).plot(kind="bar", stacked=True, figsize=(4,4) )



# Frequency Table for Self Employed and Loan Status


Self_Employed= pd.crosstab(train['Self_Employed'],train['Loan_Status'])
Self_Employed

Self_Employed.div(Self_Employed.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))



# Frequency Table for Credit History and Loan Status


Credit_History= pd.crosstab(train['Credit_History'],train['Loan_Status'])
Credit_History

Credit_History.div(Credit_History.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True, figsize=(4,4))



# Frequency Table for Property Area and Loan Status


Property_Area=pd.crosstab(train['Property_Area'],train['Loan_Status'])
Property_Area

Property_Area.div(Property_Area.sum(1).astype(float),axis=0).plot(kind='bar', stacked=True, figsize=(4,4))



# Plotting of Numerical Categorical Variable and Loan Status


bins=[0,2500,4000,6000,8100] 
group=['Low','Average','High', 'Very high'] 
train['Income_bin']=pd.cut(train['ApplicantIncome'],bins,labels=group)

Income_bin=pd.crosstab(train['Income_bin'],train['Loan_Status']) 
Income_bin.div(Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.xlabel('ApplicantIncome') 
P=plt.ylabel('Percentage')



# Doing the same for Coapplicant Income


bins=[0,1000,3000,42000] 
group=['Low','Average','High'] 
train['Coapplicant_Income_bin']=pd.cut(train['CoapplicantIncome'],bins,labels=group)
Coapplicant_Income_bin=pd.crosstab(train['Coapplicant_Income_bin'],train['Loan_Status']) 
Coapplicant_Income_bin.div(Coapplicant_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.xlabel('CoapplicantIncome') 
P = plt.ylabel('Percentage')

# It shows that if coapplicant’s income is less the chances of loan approval are high. But this does not look right. The possible reason behind this may be that most of the applicants don’t have any coapplicant so the coapplicant income for such applicants is 0 and hence the loan approval is not dependent on it. So we can make a new variable in which we will combine the applicant’s and coapplicant’s income to visualize the combined effect of income on loan approval.
# 
# Let us combine the Applicant Income and Coapplicant Income and see the combined effect of Total Income on the Loan_Status.


train['Total_Income']=train['ApplicantIncome']+train['CoapplicantIncome']
bins=[0,2500,4000,6000,81000] 
group=['Low','Average','High', 'Very high'] 
train['Total_Income_bin']=pd.cut(train['Total_Income'],bins,labels=group)
Total_Income_bin=pd.crosstab(train['Total_Income_bin'],train['Loan_Status']) 
Total_Income_bin.div(Total_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.xlabel('Total_Income') 
P = plt.ylabel('Percentage')



# Plotting of Loan Amount and Loan Status


bins=[0,100,200,700] 
group=['Low','Average','High'] 
train['LoanAmount_bin']=pd.cut(train['LoanAmount'],bins,labels=group)
LoanAmount_bin=pd.crosstab(train['LoanAmount_bin'],train['Loan_Status']) 
LoanAmount_bin.div(LoanAmount_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.xlabel('LoanAmount') 
P = plt.ylabel('Percentage')




# Change the 3+ in dependents variable to 3 to make it a numerical variable.We will also convert the target variable’s categories into 0 and 1 


train['Dependents'].replace('3+', 3,inplace=True) 
test['Dependents'].replace('3+', 3,inplace=True) 



# Convert the target variable 'Loan Status' categories into 0 and 1 for logistic regression


train['Loan_Status'].replace('N', 0,inplace=True) 
train['Loan_Status'].replace('Y', 1,inplace=True)



# Following inferences can be made from the above bar plots:
# 
# •	It seems people with credit history as 1 are more likely to get the loans approved.
# 
# •	Proportion of loans getting approved in semi-urban area is higher than as compared to that in rural and urban areas.
# 
# •	Proportion of married applicants is higher for the approved loans.
# 
# •	Ratio of male and female applicants is more or less same for both approved and unapproved loans.
# 




# # Correlation using Heatmaps


matrix = train.corr() 
plt.figure(figsize=(9,6))
sns.heatmap(matrix, square=True, cmap="BuPu")

# We see that the most correlated variables are (Applicant Income – Loan Amount) and (Credit_History – Loan Status).
# 
# LoanAmount is also correlated with CoapplicantIncome.
# 




train=train.drop(['Income_bin', 'Coapplicant_Income_bin', 'LoanAmount_bin', 'Total_Income_bin', 'Total_Income'], axis=1)

train.head()

# # Handling the missing Data


# Checking the number of null values


train.isnull().sum()

# There are null values in Gender,Married,Dependents,Self_Employed,LoanAmount,Loan_Amount_Term.
# So replacing the null values with the mode of the respective colums so that the values does not affect the result.


train['Gender'].fillna(train['Gender'].mode()[0],inplace=True)

train['Married'].fillna(train['Married'].mode()[0],inplace=True)

train['Dependents'].fillna(train['Dependents'].mode()[0],inplace=True)

train['Self_Employed'].fillna(train['Self_Employed'].mode()[0],inplace=True)

train['Credit_History'].fillna(train['Credit_History'].mode()[0],inplace=True)

train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0],inplace=True)

train['LoanAmount'].fillna(train['LoanAmount'].median(),inplace=True)



test['Gender'].fillna(test['Gender'].mode()[0],inplace=True)

test['Married'].fillna(test['Married'].mode()[0],inplace=True)

test['Dependents'].fillna(test['Dependents'].mode()[0],inplace=True)

test['Self_Employed'].fillna(test['Self_Employed'].mode()[0],inplace=True)

test['Credit_History'].fillna(test['Credit_History'].mode()[0],inplace=True)

test['Loan_Amount_Term'].fillna(test['Loan_Amount_Term'].mode()[0],inplace=True)

test['LoanAmount'].fillna(test['LoanAmount'].median(),inplace=True)

# # Outlier Treatment


# There are many outliers in the LoanAmount.Doing the log transformation to make the distribution look normal.



train['LoanAmount_log'] = np.log(train['LoanAmount']) 
train['LoanAmount_log'].hist(bins=20) 


test['LoanAmount_log'] = np.log(test['LoanAmount'])
test['LoanAmount_log'].hist(bins=20)

# Now the distribution for LoanAmount looks much closer to normal and effect of extreme values has been significantly subsided. 


# The Model Building Steps are done in the "Model Building.ipynb" notebook.


train=train.drop('Loan_ID',axis=1)
train.head()

test=test.drop('Loan_ID',axis=1)
test.head()



train=train.drop('Gender',axis=1)
test=test.drop('Gender',axis=1)



train=train.drop('Dependents',axis=1)
test=test.drop('Dependents',axis=1)



train=train.drop('Self_Employed',axis=1)
test=test.drop('Self_Employed',axis=1)



# Also dropping the Loan_Status column and storing it in another variable.


x=train.drop('Loan_Status',axis=1)
x.head()

y=train['Loan_Status']
y.head()



# Creating Dummy Varible


x=pd.get_dummies(x) 
train=pd.get_dummies(train) 
test=pd.get_dummies(test)



def stratified_cross_validation(model):
    
    """This function performs Stratified Shuffle Split. Accepts the model as an argument and returns stratified 
    randomized fold scores and model predictions"""
    
    counter=1 
    pred_scores=[]
    kf = StratifiedShuffleSplit(n_splits=4,random_state=1,test_size= 0.25) 
    for train_index,test_index in kf.split(X,y):
        xtr,xvl = X.loc[train_index],X.loc[test_index]
        ytr,yvl = y[train_index],y[test_index]
        model.fit(xtr, ytr)
        pred_test = model.predict(xvl)
        score = accuracy_score(yvl,pred_test)   
        counter+=1 
        pred=model.predict_proba(xvl)[:,1]
        pred_scores.append(score)
    return pred_scores, pred_test



def display_cf_matrix(y_cv,model_pred):
    
    """This function draws the confusion matrix. Accepts true values of the target and the predicted values of the target made by 
    the model as an argument"""
    
    cf_matrix = confusion_matrix(y_cv,model_pred)
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    labels = [f"{v1}: {v2}" for v1, v2 in zip(group_names,group_counts)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')



def accuracy_metrics(y_test,predictions):

  """This function prints the classification report of the model which includes Precion, Recall and F-1 Score"""

  report = classification_report(y_test, predictions)
  print(report)



# # Baseline Model: Applying Logistic Regression


x_train, x_cv, y_train, y_cv = train_test_split(x,y, train_size =0.75,random_state=0)

model = LogisticRegression() 
model.fit(x_train, y_train)

baseline_lr = model.predict(x_cv)

print("Accuracy of the Baseline Model is {}".format(round(accuracy_score(y_cv,baseline_lr)*100,2)), "%")

accuracy_metrics(y_cv,baseline_lr)

display_cf_matrix(y_cv,baseline_lr)





# # Feature Engineering


# Based on the domain knowledge, we can come up with new features that might affect the target variable. We can come up with following new three features:


# 1. Total Income: As evident from Exploratory Data Analysis, we will combine the Applicant Income and Coapplicant Income. If the total income is high, chances of loan approval might also be high.
# 
# 


train['Total_Income'] = train['ApplicantIncome'] + train['CoapplicantIncome'] 

test['Total_Income'] = test['ApplicantIncome'] + test['CoapplicantIncome'] 

sns.distplot(train['Total_Income'])



train['Total_Income_log'] = np.log(train['Total_Income'])

test['Total_Income_log'] = np.log(test['Total_Income'])

sns.distplot(train['Total_Income_log'])



# 2. EMI: EMI is the monthly amount to be paid by the applicant to repay the loan. Idea behind making this variable is that people who have high EMI’s might find it difficult to pay back the loan. We can calculate EMI by taking the ratio of loan amount with respect to loan amount term.


train['EMI'] = train['LoanAmount'] / train['Loan_Amount_Term'] 

test['EMI'] = test['LoanAmount'] / test['Loan_Amount_Term'] 

sns.distplot(test['EMI'])



# 3. Balance Income: This is the income left after the EMI has been paid. Idea behind creating this variable is that if the value is high, the chances are high that a person will repay the loan and hence increasing the chances of loan approval.


train['Balance_Income'] = train['Total_Income'] - (train['EMI'] * 1000)

test['Balance_Income'] = test['Total_Income'] - (test['EMI'] * 1000)

sns.distplot(train['Balance_Income'])



train = train.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1)

test = test.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1)

# # Model Building


X=train.drop('Loan_Status',axis=1)
X.head()

y = train.Loan_Status





# # Logistic Regression Model


lr_model = LogisticRegression(random_state=1)

lr_model_score, lr_model_pred = stratified_cross_validation(lr_model)

print("\nMean of Accuracy Scores=",mean(lr_model_score))

print(classification_report(y_cv,lr_model_pred))

display_cf_matrix(y_cv,lr_model_pred)



# # Decision Tree Model


dt_model = tree.DecisionTreeClassifier(random_state=1)

dt_model_score, dt_model_pred = stratified_cross_validation(dt_model)

print("\nMean of Accuracy Scores=",mean(dt_model_score))

print(classification_report(y_cv,dt_model_pred))

display_cf_matrix(y_cv,dt_model_pred)



# # Random Forest Model


rf_model = RandomForestClassifier(random_state=1, max_depth=10)

rf_model_score, rf_model_pred = stratified_cross_validation(rf_model)

print("\nMean of Accuracy Scores=",mean(rf_model_score))

print(classification_report(y_cv,rf_model_pred))

display_cf_matrix(y_cv,rf_model_pred)



# # Hyper-Parameter Tuning


paramgrid = {'max_depth': list(range(1,20,2)), 'n_estimators': list(range(1,200,20)) }

grid_search= GridSearchCV(RandomForestClassifier(random_state=1), paramgrid)

x_train, x_cv, y_train, y_cv = train_test_split(X,y, train_size =0.75,random_state=1)



grid_search.fit(x_train, y_train)



# Estimating the optimized value


grid_search.best_estimator_

# # Tuned Random Forest Model


hyper_rf_model =  RandomForestClassifier(random_state=1, max_depth=3, n_estimators=41)

hyper_rf_model_score, hyper_rf_model_pred = stratified_cross_validation(hyper_rf_model)

print("\nMean of Accuracy Scores=",mean(hyper_rf_model_score))

print(classification_report(y_cv,hyper_rf_model_pred))

display_cf_matrix(y_cv,hyper_rf_model_pred)



# # Feature Importance


importances= pd.Series(hyper_rf_model.feature_importances_, index = X.columns).sort_values()

importances.plot(kind='barh', figsize=(12,8))



# # XGBoost


xgb_model = XGBClassifier(random_state=1, max_depth=4, n_estimators=50)

xgb_model_score, xgb_model_pred = stratified_cross_validation(xgb_model)

print("\nMean of Accuracy Scores=",mean(xgb_model_score))

print(classification_report(y_cv,xgb_model_pred))

display_cf_matrix(y_cv,xgb_model_pred)
