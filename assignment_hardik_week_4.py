import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

### loading titanic data
titanic_data= sns.load_dataset('titanic')
print(titanic_data.head())

### data summary
print("Shape of dataset: ", titanic_data.shape)
print("\nColumn in dataset:\n", titanic_data.columns)
print("Data types:", titanic_data.dtypes)
print("Summary statistics:\n ",titanic_data.describe(include='all'))


### missing values
print("Missing values:\n", titanic_data.isnull().sum().sort_values(ascending=False))
print("\nPercentage of missing values:\n", (titanic_data.isnull().sum()/len(titanic_data))*100)

### graphs 
sns.heatmap(titanic_data.isnull(),cbar=True,cmap="magma",yticklabels=False)
plt.title("Missing values heatmap")
plt.show()


### finding missing values  

titanic_data_copy=titanic_data.copy()
numerical_columns=titanic_data_copy.select_dtypes(include=["number"]).columns
categorical_columns=titanic_data_copy.select_dtypes(exclude=["number"]).columns

### for numerical_columns
imputer_num= IterativeImputer(estimator=RandomForestRegressor(n_estimators=5), random_state=0,max_iter=10)
titanic_data_copy[numerical_columns]= imputer_num.fit_transform(titanic_data_copy[numerical_columns])

### for catagoriacl_columns
for col in categorical_columns:
    titanic_data_copy[col]= titanic_data_copy[col].fillna(titanic_data_copy[col].mode()[0])
print("\nMissing values after computing:\n",titanic_data_copy.isnull().sum())

### box plot for outliners

plt.subplot(1,2,1)
sns.boxplot(y='age', data=titanic_data)
plt.title("Boxplot of Age")

plt.subplot(1,2,2)
sns.boxplot(y='fare', data=titanic_data)
plt.title("Boxplot of Fare")
plt.tight_layout()
plt.show()

### Univariate Analysis

print("\n Survived:",titanic_data_copy['survived'].value_counts())
plt.subplot(2,2,1)
sns.countplot(x="survived",data=titanic_data_copy)
plt.title("Survived Count")

print("\n Passenger Class Distribution:\n",titanic_data_copy['pclass'].value_counts())
plt.subplot(2,2,2)
sns.countplot(x="pclass",data=titanic_data_copy)
plt.title("Passenger Class Distribution")

print("\nAge Statistics:\n",titanic_data_copy['age'].describe())
plt.subplot(2,2,3)
sns.histplot(x="age",data=titanic_data_copy,kde=True)
plt.title("Age Distribution")

print("\nSex Distribution\n",titanic_data_copy["sex"].value_counts())
plt.subplot(2,2,4)
sns.countplot(x="sex",data=titanic_data_copy)
plt.title("Sex Distribution")

plt.tight_layout()
plt.show()

###  Bivariate Analysis

print("\nSurvival by Gender:\n",titanic_data_copy.groupby("sex")["survived"].value_counts())
plt.subplot(2,2,1)
sns.countplot(x="sex",hue="survived",data=titanic_data_copy)
plt.title("Survived by Gender")

print("\nSurvival by Passenger Class:\n",titanic_data_copy.groupby("pclass")["survived"].value_counts())
plt.subplot(2,2,2)
sns.countplot(x="pclass", hue="survived",data=titanic_data_copy)
plt.title("Survived by Passenger Class")

titanic_data_copy["age_group"]= pd.cut(titanic_data_copy["age"], bins=[0, 18, 65, 100], labels=["Child","Adult","Senior"])
print("\nSurvival by Age Group:\n", titanic_data_copy.groupby("age_group")["survived"].value_counts())
plt.subplot(2,2,3)
sns.countplot(x="age_group", hue="survived", data=titanic_data_copy)
plt.title("Survival by Age Group")

print("\nSurvival by Gender and Class:\n", titanic_data_copy.groupby(["sex", "pclass"])["survived"].value_counts())
sns.catplot(x="sex", hue="survived", col="pclass", kind="count", data=titanic_data_copy)
plt.tight_layout()
plt.show()

### Survival Analysis

sns.countplot(x="embarked",hue="survived", data=titanic_data,palette="Spectral")
plt.title("Survey based on Embarkation Point")
plt.xlabel("Embarkation Point")
plt.ylabel("Count")
plt.legend(title="Survived",labels=["No","Yes"])
plt.tight_layout()
plt.show()

### FEATURE ENGINEERING

bins = [0, 12, 18, 40, 60, 80]
labels = ["Child", "Teenager", "Adult", "Middle-aged", "Senior"]
titanic_data["age_group"] = pd.cut(titanic_data["age"], bins=bins, labels=labels)

sns.countplot(x="age_group", hue="survived", data=titanic_data, palette="magma")
plt.title("Survival Based on Age Group")
plt.xlabel("Age Group")
plt.ylabel("Count")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.tight_layout()
plt.show()

### usign numerical feature for correlation

numeric_features = titanic_data.select_dtypes(include=np.number)
sns.heatmap(numeric_features.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()