### Ex-07-Feature-Selection
### AIM

To Perform the various feature selection techniques on a dataset and save the data to a file.
### Explanation

Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
## ALGORITHM
## STEP 1:

Read the given Data
## STEP 2:

Clean the Data Set using Data Cleaning Process
## STEP 3:

Apply Feature selection techniques to all the features of the data set
## STEP 4:

Save the data to the file
## CODE:

Developed by: s.thirisha
Register Number: 212222230160
```python
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

from sklearn.datasets import load_boston
boston = load_boston()

print(boston['DESCR'])

import pandas as pd
df = pd.DataFrame(boston['data'] )
df.head()

df.columns = boston['feature_names']
df.head()

df['PRICE']= boston['target']
df.head()

df.info()

plt.figure(figsize=(10, 8))
sns.distplot(df['PRICE'], rug=True)
plt.show()

#FILTER METHODS
X=df.drop("PRICE",1)
y=df["PRICE"]

from sklearn.feature_selection import SelectKBest, chi2
X, y = load_boston(return_X_y=True)
X.shape

#1.VARIANCE THRESHOLD
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold()
selector.fit_transform(X)

#2.INFORMATION GAIN/MUTUAL INFORMATION
from sklearn.feature_selection import mutual_info_regression
mi = mutual_info_regression(X, y);
mi = pd.Series(mi)
mi.sort_values(ascending=False)
mi.sort_values(ascending=False).plot.bar(figsize=(10, 4))

#3.SELECTKBEST METHOD
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest,SelectPercentile
skb = SelectKBest(score_func=f_classif, k=2) 
X_data_new = skb.fit_transform(X, y)
print('Number of features before feature selection: {}'.format(X.shape[1]))
print('Number of features after feature selection: {}'.format(X_data_new.shape[1]))

#4.CORRELATION COEFFICIENT
cor=df.corr()
sns.heatmap(cor,annot=True)

#5.MEAN ABSOLUTE DIFFERENCE
mad=np.sum(np.abs(X-np.mean(X,axis=0)),axis=0)/X.shape[0]
plt.bar(np.arange(X.shape[1]),mad,color='teal')

#Processing data into array type.
from sklearn import preprocessing
lab = preprocessing.LabelEncoder()
y_transformed = lab.fit_transform(y)
print(y_transformed)

#6.CHI SQUARE TEST
X = X.astype(int)
chi2_selector = SelectKBest(chi2, k=2)
X_kbest = chi2_selector.fit_transform(X, y_transformed)
print('Original number of features:', X.shape[1])
print('Reduced number of features:', X_kbest.shape[1])

#7.SELECT PERCENTILE METHOD
X_new = SelectPercentile(chi2, percentile=10).fit_transform(X, y_transformed)
X_new.shape

#WRAPPER METHOD

#1.FORWARD FEATURE SELECTION

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression
sfs = SFS(LinearRegression(),
          k_features=10,
          forward=True,
          floating=False,
          scoring = 'r2',
          cv = 0)
sfs.fit(X, y)
sfs.k_feature_names_

#2.BACKWARD FEATURE ELIMINATION

sbs = SFS(LinearRegression(),
         k_features=10,
         forward=False,
         floating=False,
         cv=0)
sbs.fit(X, y)
sbs.k_feature_names_

#3.BI-DIRECTIONAL ELIMINATION

sffs = SFS(LinearRegression(),
         k_features=(3,7),
         forward=True,
         floating=True,
         cv=0)
sffs.fit(X, y)
sffs.k_feature_names_

#4.RECURSIVE FEATURE SELECTION

from sklearn.feature_selection import RFE
lr=LinearRegression()
rfe=RFE(lr,n_features_to_select=7)
rfe.fit(X, y)
print(X.shape, y.shape)
rfe.transform(X)
rfe.get_params(deep=True)
rfe.support_
rfe.ranking_

#EMBEDDED METHOD

#1.RANDOM FOREST IMPORTANCE

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier().fit(X,y_transformed)
importances=model.feature_importances_

final_df=pd.DataFrame({"Features":pd.DataFrame(X).columns,"Importances":importances})
final_df.set_index("Importances")
final_df=final_df.sort_values("Importances")
final_df.plot.bar(color="teal")
```

### OUPUT
![image](https://user-images.githubusercontent.com/120380280/234183391-46e9c0e9-a78c-4ce0-b892-c4cf8aa71fc5.png)
## Analyzing the boston dataset:
![image](https://user-images.githubusercontent.com/120380280/234183564-99335756-f62a-4f61-9831-b07b68af5815.png)

![image](https://user-images.githubusercontent.com/120380280/234183589-9ac64f48-7571-4ccc-b360-fb5233fdb8a9.png)

![image](https://user-images.githubusercontent.com/120380280/234183608-d827421c-74b3-43a5-8c53-d699cb71f432.png)
### Analyzing dataset using Distplot:
![image](https://user-images.githubusercontent.com/120380280/234183694-6c918f9f-bf03-4931-a800-2bb6aca14cba.png)
## Filter Methods:
Variance Threshold:
![image](https://user-images.githubusercontent.com/120380280/234183751-01bb88b7-d3a5-4e30-81bb-a38967457946.png)
## Information Gain:
![image](https://user-images.githubusercontent.com/120380280/234183798-0385a090-9269-43dc-a5e7-19adb5418f0c.png)
## SelectKBest Model:
![image](https://user-images.githubusercontent.com/120380280/234183842-8d04bfd7-36c7-4070-9ded-8458722eed57.png)
## Correlation Coefficient:
![image](https://user-images.githubusercontent.com/120380280/234183891-a7022d87-1d47-42f8-afd6-2f4f032a5c4b.png)
## Mean Absolute difference:
![image](https://user-images.githubusercontent.com/120380280/234183930-40ec06f8-171b-480e-83ed-b4f7b231665f.png)
## Chi Square Test:
![image](https://user-images.githubusercontent.com/120380280/234183965-71ba514e-8393-4caf-ab19-6b0c02ffcaed.png)
![image](https://user-images.githubusercontent.com/120380280/234183992-d054afd6-9a26-4bef-83ad-efc3ab4fa573.png)
## 12
SelectPercentile Method:





















