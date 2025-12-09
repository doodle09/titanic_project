Titanic Dataset – Exploratory Data Analysis, Missing Value Imputation & Feature Engineering

This project performs an in-depth exploratory data analysis (EDA) on the famous Titanic dataset to understand passenger demographics, survival patterns, missing value behavior, and correlations between key features. The analysis includes visualizing missing data, cleaning the dataset using advanced imputation techniques, studying outliers, and exploring relationships between survival and variables like gender, passenger class, age, and embarkation point.

The goal of the project is to build a strong foundational understanding of data preprocessing, visualization, and feature engineering—skills essential for any data analyst or machine learning workflow.

Key Highlights of the Analysis
1. Missing Value Detection & Imputation

The dataset contains missing values in several columns such as age, deck, and embarkation.
The project includes:

A heatmap to visualize all missing values

Separate handling of numerical and categorical features

Iterative Imputer with RandomForestRegressor for numerical columns

Mode imputation for categorical variables

After imputation, all missing values are successfully resolved.

2. Outlier Analysis

To identify outliers and understand spread:

Boxplots are created for Age and Fare

Visuals help detect extreme values affecting model performance or interpretation

3. Univariate Analysis

The project examines:

Survival distribution

Passenger class distribution

Age distribution (with KDE curve)

Gender distribution

These insights reveal how passengers were grouped across various categories.

4. Bivariate Analysis

To understand deeper relationships, the project compares:

Survival by Gender

Survival by Passenger Class

Survival by Age Group (Child, Adult, Senior)

Survival differences across Gender + Class combinations

The results align with historic accounts—women, children, and higher-class passengers had better survival chances.

5. Feature Engineering

A new feature, Age Group, is created by binning age into categories:

Child, Teenager, Adult, Middle-aged, Senior

This helps simplify age analysis and reveals clearer trends in survival.

6. Correlation Analysis

A correlation heatmap is generated for all numerical features to study relationships such as:

Fare and class

Age and survival

SibSp, Parch, and family structure

These insights support understanding of which features may contribute to prediction models.

Conclusion

This project demonstrates the complete lifecycle of data preparation and exploratory analysis:

Understanding the dataset

Handling missing values

Detecting outliers

Conducting univariate and bivariate EDA

Engineering new features

Examining numerical correlations

It serves as a strong example of practical, real-world EDA work and forms the foundation for future machine learning modeling on the Titanic dataset.
