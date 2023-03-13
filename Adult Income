import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("adult.csv")

# 1.Display Top 10 Rows of The Dataset
print(data.head(10))

# 2. Check Last 10 Rows of The Dataset
print(data.tail(10))

# 3. Find Shape of Our Dataset (Number of Rows And Number of Columns)
print(f"Number of Rows: {data.shape[0]} \nNumber of Columns: {data.shape[1]}")

# 4. Getting Information About Our Dataset Like Total Number Rows, Total Number of Columns, Datatypes of Each Column
# And Memory Requirement
print(data.info())

# 5. Fetch Random Sample From the Dataset (50%)
print(data.sample(frac=0.50))

# 6.Check Null Values In The Dataset
print(sns.heatmap(data.isnull()))
plt.show()

# 7.Perform Data Cleaning [ Replace '?' with NaN ]
print(data.isin(['?']).sum())
data['workclass'] = data['workclass'].replace('?', np.nan)
data['occupation'] = data['occupation'].replace('?', np.nan)
data['native-country'] = data['native-country'].replace('?', np.nan)
print(data.isin(['?']).sum())
print(data.isnull().sum())
sns.heatmap(data.isnull())
plt.show()

# 8. Drop all The Missing Values
missing_perc = (data.isnull().sum())*100/len(data)
print(missing_perc)
data.dropna(how='any', inplace=True)
print(data.shape)

# 9. Check For Duplicate Data and Drop Them
print(data.duplicated().any(), data.shape)
data = data.drop_duplicates()
print(data.shape)

# 10. Get Overall Statistics About The Dataframe
print(data.describe())


# 11. Drop The Columns education-num, capital-gain, and capital-loss
data = data.drop(['educational-num', 'capital-gain', 'capital-loss'], axis=1)
print(data)


# 12. What Is The Distribution of Age Column?
data['age'].hist()
plt.show()
# 13. Find Total Number of Persons Having Age Between 17 To 48 (Inclusive) Using Between Method
print(data[(data['age'] >= 17) & (data['age']) < 49])
print(sum(data['age'].between(17, 49)))

# 14. What is The Distribution of Workclass Column?
plt.figure(figsize=(15, 15))
data['workclass'].hist()
plt.show()

# 15. How Many Persons Having Bachelors and Masters Degree?
print(((data['education'] == 'Bachelors') | (data['education'] == 'Masters')).sum())
print(sum(data['education'].isin(['Bachelors', 'Masters'])))

# 16. Bivariate Analysis
sns.boxplot(x=data['income'], y=data['age'], data=data)
plt.show()

# 17. Replace Salary Values With 0 and 1
print(data['income'].unique(), data['income'].value_counts())


def sal(salary):
    if salary == '<=50K':
        return 0
    else:
        return 1


data['end'] = data['income'].apply(sal)
data.replace(to_replace=['<=50K', '>50K'], value=[0, 1], inplace=True)
print(data.head())
sns.countplot(x=data['income'])
plt.show()

# 18. Which Workclass Getting The Highest Salary?

print(data.groupby('workclass')['end'].mean())
print(data)
# 19.How Has Better Chance To Get Salary greater than 50K Male or Female?
print(data.groupby('gender')['end'].mean().sort_values(ascending=True))

# 20. Covert workclass Columns Datatype To Category Datatype
print(data.info())
data['workclass'] = data['workclass'].astype('category')
print(data.info())
