# Ex-06-Feature-Transformation
# AIM:
To read the given data and perform Feature Transformation process and save the data to a file.

# EXPLANATION:
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

# ALGORITHM:
## STEP 1
Read the given Data

## STEP 2
Clean the Data Set using Data Cleaning Process

## STEP 3
Apply Feature Transformation techniques to all the features of the data set

## STEP 4
Save the data to the file

#CODE:
```
Name : kavya.k
Register Number : 212222230065
Feature Transformation - Data_to_Transform.csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
df = pd.read_csv("/content/Data_to_Transform.csv")
print(df)
df.head()
df.isnull().sum()
df.info()
df.describe()
df1 = df.copy()
sm.qqplot(df1.HighlyPositiveSkew,fit=True,line='45')
plt.show()
sm.qqplot(df1.HighlyNegativeSkew,fit=True,line='45')
plt.show()
sm.qqplot(df1.ModeratePositiveSkew,fit=True,line='45')
plt.show()
sm.qqplot(df1.ModerateNegativeSkew,fit=True,line='45')
plt.show()
df1['HighlyPositiveSkew'] = np.log(df1.HighlyPositiveSkew)
sm.qqplot(df1.HighlyPositiveSkew,fit=True,line='45')
plt.show()
df2 = df.copy()
df2['HighlyPositiveSkew'] = 1/df2.HighlyPositiveSkew
sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')
plt.show()
df3 = df.copy()
df3['HighlyPositiveSkew'] = df3.HighlyPositiveSkew**(1/1.2)
sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')
plt.show()
df4 = df.copy()
df4['ModeratePositiveSkew_1'],parameters =stats.yeojohnson(df4.ModeratePositiveSkew)
sm.qqplot(df4.ModeratePositiveSkew_1,fit=True,line='45')
plt.show()
from sklearn.preprocessing import PowerTransformer 
trans = PowerTransformer("yeo-johnson")
df5 = df.copy()

df5['ModerateNegativeSkew_1'] = pd.DataFrame(trans.fit_transform(df5[['ModerateNegativeSkew']]))
sm.qqplot(df5['ModerateNegativeSkew_1'],line='45')
plt.show()
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df5['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df5[['ModerateNegativeSkew']]))
sm.qqplot(df5['ModerateNegativeSkew_2'],line='45')
plt.show()
```
# OUTPUT:
## Feature Transformation - Data_to_Transform.csv
![image](https://github.com/kavyasenthamarai/Ex-06-Feature-Transformation/assets/118668727/721f8381-6def-46eb-bc62-afdb3355ae65)
![6 2](https://github.com/kavyasenthamarai/Ex-06-Feature-Transformation/assets/118668727/0980f83a-3e30-4c45-8470-bfa28b98253d)
![6 3](https://github.com/kavyasenthamarai/Ex-06-Feature-Transformation/assets/118668727/659983ad-39ca-44a9-86b3-eff7e1ca0141)
![6 4](https://github.com/kavyasenthamarai/Ex-06-Feature-Transformation/assets/118668727/9dff1f46-e606-4bd5-9d4d-3c1f22770825)
![6 5](https://github.com/kavyasenthamarai/Ex-06-Feature-Transformation/assets/118668727/c10420d8-8e90-4703-a4b7-da2370356a0a)

## Log Transformation
![6 6](https://github.com/kavyasenthamarai/Ex-06-Feature-Transformation/assets/118668727/05694bbd-dc48-49ab-b5eb-d942a02243c5)


## Reciprocal Transformation

![6 7](https://github.com/kavyasenthamarai/Ex-06-Feature-Transformation/assets/118668727/07069add-5190-40dc-975c-237beee37124)


## SquareRoot Transformation

![6 8](https://github.com/kavyasenthamarai/Ex-06-Feature-Transformation/assets/118668727/5482b045-e722-4a90-bc13-5ac9da55a84b)


## Power Transformation!
[6 9](https://github.com/kavyasenthamarai/Ex-06-Feature-Transformation/assets/118668727/35a767f2-fffc-49f5-9781-d8101e06f629)

![6 10](https://github.com/kavyasenthamarai/Ex-06-Feature-Transformation/assets/118668727/9cf1fd9e-f85d-4876-a26a-db20f1af4d2b)
## Quantile Transformation
![6 11](https://github.com/kavyasenthamarai/Ex-06-Feature-Transformation/assets/118668727/fce1c176-c8ff-4d9a-8091-cc0030216374)

## RESULT:
Thus the Feature Transformation for the given datasets had been executed successfully




















