# EXNO2DS
# AIM:
      To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```
```
dt=pd.read_csv("/content/titanic_dataset.csv")
```
```
dt
```



<img width="775" height="324" alt="423029384-9b158723-8628-4afb-960f-719269c88c0b" src="https://github.com/user-attachments/assets/6f1dd487-a557-4915-b3d4-6068983530b5" />



```
dt.info()
```



<img width="576" height="324" alt="423025017-e5fcb6ae-60fc-4fd5-aae2-28df4fabd154" src="https://github.com/user-attachments/assets/78f76a1b-05c2-4a23-bdfe-46b0471eacb3" />


```
dt.set_index("PassengerId",inplace=True)
```


```
dt.describe()
```


<img width="576" height="324" alt="423025343-6c88937b-6c10-4b0b-9a7f-dc95773909bd" src="https://github.com/user-attachments/assets/02468b27-e348-46cc-a8e6-ed7158862231" />


```
dt.shape
```


<img width="576" height="324" alt="423025702-b2e5ef01-4aaf-48c3-b0ad-77e665e9806d" src="https://github.com/user-attachments/assets/c0bb2285-f799-455f-9900-3737edd0439c" />


```
dt.nunique()
```


<img width="576" height="324" alt="423025776-a7f0ba82-c599-4de7-96ed-6182a09601a0" src="https://github.com/user-attachments/assets/eb4ebff9-b28b-4209-8f70-9fec5814f548" />


```
dt["Survived"].value_counts()
```


<img width="576" height="324" alt="423026060-dca9efce-9416-4d7b-abd4-5efb53b39a89" src="https://github.com/user-attachments/assets/25a1ea97-eac4-47ed-91a7-18071ec813cb" />


```
per=(dt["Survived"].value_counts()/dt.shape[0]*100).round(2)
per
```


<img width="576" height="324" alt="423026116-0da85651-4e43-4fc9-9200-6164f9cf8478" src="https://github.com/user-attachments/assets/3f170435-55e7-4ede-b810-387cbf428257" />


```
sns.countplot(data=dt,x="Survived")
```


<img width="576" height="337" alt="423027987-130621d1-4207-4a22-97f4-ce7c451710c3" src="https://github.com/user-attachments/assets/e367dc3b-62db-4836-acc8-e968562f0a00" />


```
dt.Pclass.unique()
```


<img width="576" height="324" alt="423028041-fb10fcb7-11bc-4b90-bd41-47858b76214f" src="https://github.com/user-attachments/assets/de262e58-d74d-41b5-88b7-6b63edac0227" />


```
dt.rename(columns={'Sex':'Gender'},inplace=True)
dt
```


<img width="835" height="336" alt="423028097-c9bdd853-c093-47f3-a5ee-3623897b999a" src="https://github.com/user-attachments/assets/7b0e9242-c1f5-494b-a703-b95f50ea7b9c" />



```
sns.catplot(x="Gender",col="Survived",kind="count",data=dt,height=5,aspect=.7)
```


<img width="576" height="388" alt="423028203-7ca54b04-6c19-4335-a0c3-808440320026" src="https://github.com/user-attachments/assets/e6368964-d602-4b9b-8e9e-5be1f9856b54" />


```
sns.catplot(x="Survived",hue="Gender",data=dt,kind="count")
```


<img width="576" height="386" alt="423028213-9b78dfd4-3942-4411-a3fd-13841e5e7d1e" src="https://github.com/user-attachments/assets/84db3de8-1a9b-4d76-98b8-b5b4fa4095da" />


```
sns.scatterplot(x=dt["Age"],y=dt['Fare'])
```


<img width="576" height="346" alt="423028240-4024b120-846d-4e46-a948-a1304a9a0411" src="https://github.com/user-attachments/assets/1ba7a44f-931a-430b-927e-78f28291a9ec" />


```
sns.jointplot(x="Age",y="Fare",data=dt)
```


<img width="576" height="463" alt="423028270-5b00a35d-41b1-49e2-b463-7d9ab7147515" src="https://github.com/user-attachments/assets/25c411a8-2121-48ad-8fa8-2fa9f3ba1371" />


```
fig,ax1=plt.subplots(figsize=(8,5))
pt=sns.boxplot(ax=ax1,x='Pclass',y='Age',hue='Gender',data=dt)
```


<img width="576" height="341" alt="423028292-c05d8a65-9456-4cff-90e7-424a1434036c" src="https://github.com/user-attachments/assets/6765c05e-38b2-45f0-be66-8cd073f1dfdf" />


```
sns.catplot(data=dt,col="Survived",x="Gender",hue="Pclass",kind="count")
```


<img width="804" height="390" alt="423028310-eb800629-edcd-4d7b-ac61-49f118225296" src="https://github.com/user-attachments/assets/fe1371de-dd41-4e93-8dee-73b3d1eff05d" />


```
corr=dt.select_dtypes(include=['number']).corr()
sns.heatmap(corr,annot=True)
```


<img width="576" height="337" alt="423028318-da6f1838-6cd0-475b-b720-d131f4dd870b" src="https://github.com/user-attachments/assets/2995e5b3-c8d1-4d8a-8f4f-756e5d042c4d" />


```
sns.pairplot(dt)
```


<img width="960" height="540" alt="423028335-7e413260-c7e5-42b7-b295-e6b76e898b44" src="https://github.com/user-attachments/assets/d3a0cfdb-fd0f-474f-8831-f9f9209c1149" />
<img width="960" height="540" alt="423028346-ba85dd4d-8d2f-4264-bbad-60b42b8d0a2b" src="https://github.com/user-attachments/assets/f7573188-7b9e-43b6-84c2-f338e1005f59" />





# RESULT
    To perform Exploratory Data Analysis on the given data is successfully completed.
