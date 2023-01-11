# Problem statement:
# Cardiovascular diseases are the leading cause of death globally. It is therefore necessary to identify the causes and develop a system to predict heart 
attacks in an effective manner. The data below has the information about the factors that might have an impact on cardiovascular health.
Dataset description:
Variable - Description Age - Age in years Sex - 1 = male; 0 = female cp - Chest pain type trestbps - Resting blood pressure (in mm Hg on admission to the hospital) 
chol - Serum cholesterol in mg/dl fbs - Fasting blood sugar > 120 mg/dl (1 = true; 0 = false) restecg - Resting electrocardiographic results thalach - 
Maximum heart rate achieved exang - Exercise induced angina (1 = yes; 0 = no) oldpeak - ST depression induced by exercise relative to rest slope - 
Slope of the peak exercise ST segment ca - Number of major vessels (0-3) colored by fluoroscopy thal - 3 = normal; 6 = fixed defect; 7 = reversible defect
Target - 1 or 0

#import the libraries required 
import pandas as pd
import numpy as np

#import the data by importing them
data=pd.read_excel('cep_dataset.xlsx')

#Frist five row of data
data.head()

#Shape of the dataset
data.shape

#type of dataset variable
data.info()

# Null value presence
data.isnull().sum(axis=0)

data.isnull().sum(axis=1)

#checking whether categorical or numerical target
data['target'].value_counts()

#Description of the dataset
data.describe()

# datatypes variable
data.dtypes

# here we will import library
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

sns.countplot(x ='sex', data = data,hue='sex')
plt.show()

sns.countplot(x='target',data=data,hue='target')
plt.show()

# finding the categorical variable presence
categorical_val = []
continous_val = []
for column in data.columns:
    print("--------------------")
    print(f"{column} : {data[column].unique()}")
    if len(data[column].unique()) <= 10:
        categorical_val.append(column)
    else:
        continous_val.append(column)

# import libarary        
import hvplot.pandas

#relation of target with gender
data.target.value_counts().hvplot.bar(
    title="Heart Disease Count", xlabel='Heart Disease', ylabel='Count', 
    width=250, height=400
)


sns.countplot(x ='sex', data = data,hue='target')
plt.show()


#Plot between sex and target
have_disease = data.loc[data['target']==1, 'sex'].value_counts().hvplot.bar(alpha=0.4) 
no_disease = data.loc[data['target']==0, 'sex'].value_counts().hvplot.bar(alpha=0.4) 

(no_disease * have_disease).opts(
    title="Heart Disease by Sex", xlabel='Sex', ylabel='Count',
    width=500, height=450, legend_cols=2, legend_position='top_right'
)

#how many have diabetics
have_disease = data.loc[data['target']==1, 'sex'].value_counts()
have_disease 

#how many have no disease
no_disease = data.loc[data['target']==0, 'sex'].value_counts()
no_disease

# have disease with age
have_disease_1 = data.loc[data['target']==1, 'age'].value_counts()
have_disease_1

# have disease with trestbps
have_disease_2 = data.loc[data['target']==1, 'trestbps'].value_counts()
have_disease_2

#have disease  with chol
have_disease_3 = data.loc[data['target']==1, 'chol'].value_counts()
have_disease_3

# create heatmap for proper relation
plt.figure(figsize=(15,10))
sns.heatmap(data.corr(), annot=True,fmt='.0%')

#pair plot 
plt.figure(figsize=(15,10))
sns.pairplot(data.select_dtypes(exclude='object'))
plt.show()

#drop the variable
x=data.drop(columns=['target','age','sex','chol','restecg','exang','oldpeak','trestbps'],axis=1)
print(x)

#separating the target variable
Y=data['target']
print(Y)

#plot with x and Y
x.corrwith(Y).plot.bar(
    figsize=(16,4),title='Correlation with CVD',fontsize=15,
    rot=90,grid=True)

# import the library
from scipy import stats
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,Y, test_size=0.3, random_state=42)
x_train.shape

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression

print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train == 0)))
  
from imblearn.under_sampling import NearMiss
sm = NearMiss() 

X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))
  
print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res == 0)))

lr1 = LogisticRegression()
lr1.fit(X_train_res, y_train_res)
predictions = lr1.predict(X_test)
  
# print classification report
print(classification_report(y_test, predictions))

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=150)
rfc.fit(X_train_res,y_train_res)
predictions1 = rfc.predict(X_test)

print(classification_report(y_test,predictions1))

rfc.score(X_test, y_test)

print(classification_report(y_train, rfc.predict(X_train)))

import statsmodels.api as sm
logit_model=sm.Logit(Y,x)
result=logit_model.fit()
print(result.summary())






# In[ ]:




