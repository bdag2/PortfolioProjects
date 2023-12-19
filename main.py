import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib

plt.style.use('ggplot')
from matplotlib.pyplot import figure

matplotlib.rcParams['figure.figsize'] = (12, 8)

pd.options.mode.chained_assignment = None

# Read data from csv file
df = pd.read_csv('movies.csv')

# Detection of missing data
for col in df.columns:
    pct_missing = (df[col].isnull()).mean()
    number_missing = (df[col].isnull()).sum()
    # print(f"{col} - {pct_missing}")
    # print(f"{col} - {number_missing}")

# Deleting the rows which have missing values
df = df.dropna()
for col in df.columns:
    pct_missing = (df[col].isnull()).mean()
    number_missing = (df[col].isnull()).sum()
    # print(f"{col} - {pct_missing}")
    # print(f"{col} - {number_missing}")

# DATA CLEANING
# change data type of columns
df['budget'] = df['budget'].astype('int64')
df['gross'] = df['gross'].astype('int64')
# obtaining the correct year from released column
df['year_correct'] = df['released'].astype(str).str.split().str[2]
df = df.sort_values(by=['gross'], inplace=False, ascending=False)
pd.set_option('display.max_rows', None)
# detecting and dropping duplicates
unique_companies = df.company.drop_duplicates()
# print(len(df.company))
# print(len(unique_companies))
df = df.drop_duplicates()  # drop duplicates by checking entire table

# Hypothesis1 : budget has high correlation with gross.
# Hypothesis2 : company has high correlation with gross.

# Scatter plot budget vs gross

plt.scatter(x=df.budget, y=df.gross)
plt.title('Budget versus Gross Earnings')
plt.xlabel('Budget')
plt.ylabel('Gross Earnings')
# Plot the budget vs gross using seaborn
sns.regplot(x='budget', y='gross', data=df, scatter_kws={"color": "red"}, line_kws={"color": "blue"})
# plt.show()
numeric_columns = df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_columns.corr()  # high correlation between budget and gross
# sns.heatmap(correlation_matrix, annot=True)  # visualization of correlation matrix
# plt.title('Correlation Matrix')
# plt.xlabel('Movie Features')
# plt.ylabel('Movie Features')
# plt.show()

# Hypothesis 2 : there might be high correlation between company and gross earnings


df_categorized = df.copy()

for col_name in df_categorized.columns:
    if df_categorized[col_name].dtype == 'object':
        df_categorized[col_name] = df_categorized[col_name].astype('category')
        df_categorized[col_name] = df_categorized[col_name].cat.codes

df_categorized = df_categorized.sort_values(by='gross', inplace=False, ascending=False)
correlation_matrix2 = df_categorized.corr()  # high correlation between budget and gross
sns.heatmap(correlation_matrix2, annot=True)  # visualization of correlation matrix
plt.title('Correlation Matrix')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')
# plt.show()

corr_pairs = correlation_matrix2.unstack()
sorted_pairs = corr_pairs.sort_values()
high_correlation = sorted_pairs[sorted_pairs > 0.5]
# print(high_correlation)

# Findings : vote and budge has the highest correlation, company has low correlation