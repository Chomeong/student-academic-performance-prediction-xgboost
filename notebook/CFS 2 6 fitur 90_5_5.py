# %% [markdown]
# ### **Import Library**

# %%
!pip install openml

# %%
import openml

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
from scipy.stats import spearmanr
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

# %% [markdown]
# ### **Load Dataset**

# %%
# Fetching the list of all available datasets on OpenML
d = openml.datasets.list_datasets(output_format='dataframe')
print(d.shape)

# Listing column names or attributes that OpenML offers
for name in d.columns:
    print(name)

# %%
print(d.head())

# %%
# Filtering dataset list to have 'students' in the 'name' column
# then sorting the list based on the 'version'
d[d['name'].str.contains('Students')].sort_values(by='version').head()

# %%
students = openml.datasets.get_dataset(43415)
students

# %%
students.features

# %%
print(students.description)

# %%
df = pd.DataFrame(students.get_data()[0], columns=students.get_data()[1])
df.head()

# %%
df.info()

# %%
df.describe()

# %%
df.describe(include=object)

# %% [markdown]
# # **Pre-processing**

# %% [markdown]
# ### **Missing Values**

# %%
df.isna().sum()

# %% [markdown]
# ### **Duplication**

# %%
#Check duplicate value in dataframe
df.duplicated().sum()

# %%
df.drop_duplicates(keep='first', inplace=True)

# %%
df.shape

# %% [markdown]
# ### **Outliers**

# %%
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion']])
plt.title('Boxplot of Numerical Features')
plt.xticks(rotation=45)
plt.show()

# IQR method for outlier detection and removal (example for 'raisedhands')
Q1 = df['raisedhands'].quantile(0.25)
Q3 = df['raisedhands'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_no_outliers = df[(df['raisedhands'] >= lower_bound) & (df['raisedhands'] <= upper_bound)]

Q1 = df['VisITedResources'].quantile(0.25)
Q3 = df['VisITedResources'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_no_outliers = df_no_outliers[(df_no_outliers['VisITedResources'] >= lower_bound) & (df_no_outliers['VisITedResources'] <= upper_bound)]

Q1 = df['AnnouncementsView'].quantile(0.25)
Q3 = df['AnnouncementsView'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_no_outliers = df_no_outliers[(df_no_outliers['AnnouncementsView'] >= lower_bound) & (df_no_outliers['AnnouncementsView'] <= upper_bound)]

Q1 = df['Discussion'].quantile(0.25)
Q3 = df['Discussion'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_no_outliers = df_no_outliers[(df_no_outliers['Discussion'] >= lower_bound) & (df_no_outliers['Discussion'] <= upper_bound)]

print(f"Original shape: {df.shape}")
print(f"Shape after outlier removal: {df_no_outliers.shape}")

# %% [markdown]
# # **EDA**

# %% [markdown]
# ### **Distribution of Qualitative Variables**

# %%
count = df['gender'].value_counts()
percent = 100*df['gender'].value_counts(normalize=True)
df_gender = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df_gender)
count.plot(kind='bar', title='Gender Distribution', xlabel='Gender', ylabel='Count')
plt.xticks(rotation=45, ha='right')
plt.show()

# %%
count = df['NationalITy'].value_counts()
percent = 100*df['NationalITy'].value_counts(normalize=True)
df_nationality = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df_nationality)
count.plot(kind='bar', title='Nationality Distribution', xlabel='Nationality', ylabel='Count')
plt.xticks(rotation=45, ha='right')
plt.show()

# %%
count = df['PlaceofBirth'].value_counts()
percent = 100*df['PlaceofBirth'].value_counts(normalize=True)
df_pob = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df_pob)
count.plot(kind='bar', title='Place of Birth Distribution', xlabel='Place of Birth', ylabel='Count')
plt.xticks(rotation=45, ha='right')
plt.show()

# %%
count = df['StageID'].value_counts()
percent = 100*df['StageID'].value_counts(normalize=True)
df_stage = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df_stage)
count.plot(kind='bar', title='Educational Level Distribution', xlabel='Educational Level', ylabel='Count')
plt.xticks(rotation=45, ha='right')
plt.show()

# %%
count = df['GradeID'].value_counts()
percent = 100*df['GradeID'].value_counts(normalize=True)
df_grade = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df_grade)
count.plot(kind='bar', title='Grade Distribution', xlabel='Grade', ylabel='Count')
plt.xticks(rotation=45, ha='right')
plt.show()

# %%
#terdapat ketidaksesuaian antara jumlah eduacational level dan grade di mana seharusnya
#lower ~ G01 - G05
#middle ~ G06 - G08
#high ~ G09 - G12

# Replace values in column 'StageID' where values in column 'GradeID' are equal to 'G-07'
df.loc[df['GradeID'] == 'G-07', 'StageID'] = 'MiddleSchool'

# %%
count = df['StageID'].value_counts()
percent = 100*df['StageID'].value_counts(normalize=True)
df_stage = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df_stage)
count.plot(kind='bar', title='Educational Level Distribution', xlabel='Educational Level', ylabel='Count')
plt.xticks(rotation=45, ha='right')
plt.show()

# %%
count = df['SectionID'].value_counts()
percent = 100*df['SectionID'].value_counts(normalize=True)
df_class = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df_class)
count.plot(kind='bar', title='Class Distribution', xlabel='Class', ylabel='Count')
plt.xticks(rotation=45, ha='right')
plt.show()

# %%
count = df['Topic'].value_counts()
percent = 100*df['Topic'].value_counts(normalize=True)
df_topic = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df_topic)
count.plot(kind='bar', title='Topic Distribution', xlabel='Topic', ylabel='Count')
plt.xticks(rotation=45, ha='right')
plt.show()

# %%
count = df['Semester'].value_counts()
percent = 100*df['Semester'].value_counts(normalize=True)
df_semester = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df_semester)
count.plot(kind='bar', title='Semester Distribution', xlabel='Semester', ylabel='Count')
plt.xticks(rotation=45, ha='right')
plt.show()

# %%
count = df['Relation'].value_counts()
percent = 100*df['Relation'].value_counts(normalize=True)
df_relation = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df_relation)
count.plot(kind='bar', title='Relation Distribution', xlabel='Relation', ylabel='Count')
plt.xticks(rotation=45, ha='right')
plt.show()

# %%
count = df['ParentAnsweringSurvey'].value_counts()
percent = 100*df['ParentAnsweringSurvey'].value_counts(normalize=True)
df_survey = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df_survey)
count.plot(kind='bar', title='Parent Answering Survey Distribution', xlabel='Parent Answering Survey', ylabel='Count')
plt.xticks(rotation=45, ha='right')
plt.show()

# %%
count = df['ParentschoolSatisfaction'].value_counts()
percent = 100*df['ParentschoolSatisfaction'].value_counts(normalize=True)
df_satisfaction = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df_satisfaction)
count.plot(kind='bar', title='Parent School Satisfaction Distribution', xlabel='Parent School Satisfaction', ylabel='Count')
plt.xticks(rotation=45, ha='right')
plt.show()

# %%
count = df['StudentAbsenceDays'].value_counts()
percent = 100*df['StudentAbsenceDays'].value_counts(normalize=True)
df_absence = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df_absence)
count.plot(kind='bar', title='Student AbsenceDays Distribution', xlabel='Student Absence Days', ylabel='Count')
plt.xticks(rotation=45, ha='right')
plt.show()

# %% [markdown]
# ### **Bivariate Analysis of Qualitative Variables**

# %%
categorical_features = ['gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID', 'SectionID', 'Topic', 'Semester',
                        'Relation', 'ParentAnsweringSurvey', 'ParentschoolSatisfaction', 'StudentAbsenceDays']

# %%
for feature in categorical_features:
    counts = df.groupby([feature, 'Class'])['Class'].count().reset_index(name='Counts')

    sns.catplot(data=df, x=feature, hue='Class', kind="count", dodge=True)
    plt.title(f"Count of 'Class' Relatif terhadap - {feature}")
    plt.xticks(rotation=45, ha='right')
    plt.show()

# %% [markdown]
# ### **Correlation Between Quantitative Variables**

# %%
correlation_matrix = df[['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion']].corr()

# %%
plt.figure(figsize=(8, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation between Quantitative Variables')
plt.show()

# %% [markdown]
# # **Feature Selection**

# %% [markdown]
# ### **Feature Selection for Categorical Value**

# %%
#List of chi value p-value & cramer's v
chi_value = []
p_value = []
cramer_v = []

# %% [markdown]
# **Gender & Class**

# %%
#Contingency Table
contingency_table1 = pd.crosstab(df['gender'], df['Class'])
print(contingency_table1)

# %%
#Chi-square
stat, p, dof, expected = chi2_contingency(contingency_table1)
chi_value.append(stat)

# interpret p-value
alpha = 0.05
print(f"P-value is {p}")
p_value.append(p)
if p <= alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (H0 holds true)')

# %%
#Cramer's V
N = np.sum(contingency_table1.values)
minimum_dimension = min(contingency_table1.shape)-1

# Calculate Cramer's V
result = np.sqrt((stat/N) / minimum_dimension)

# Print the result
print(f"Cramér's Coefficient V is {result}")
cramer_v.append(result)

# %% [markdown]
# **Nationality & Class**

# %%
#Contingency Table
contingency_table2 = pd.crosstab(df['NationalITy'], df['Class'])
print(contingency_table2)

# %%
#Chi-square
stat, p, dof, expected = chi2_contingency(contingency_table2)
chi_value.append(stat)

# interpret p-value
alpha = 0.05
print(f"P-value is {p}")
p_value.append(p)
if p <= alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (H0 holds true)')

# %%
#Cramer's V
N = np.sum(contingency_table2.values)
minimum_dimension = min(contingency_table2.shape)-1

# Calculate Cramer's V
result = np.sqrt((stat/N) / minimum_dimension)

# Print the result
print(f"Cramér's Coefficient V is {result}")
cramer_v.append(result)

# %% [markdown]
# **Place of Birth & Class**

# %%
#Contingency Table
contingency_table3 = pd.crosstab(df['PlaceofBirth'], df['Class'])
print(contingency_table3)

# %%
#Chi-square
stat, p, dof, expected = chi2_contingency(contingency_table3)
chi_value.append(stat)

# interpret p-value
alpha = 0.05
print(f"P-value is {p}")
p_value.append(p)
if p <= alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (H0 holds true)')

# %%
#Cramer's V
N = np.sum(contingency_table3.values)
minimum_dimension = min(contingency_table3.shape)-1

# Calculate Cramer's V
result = np.sqrt((stat/N) / minimum_dimension)

# Print the result
print(f"Cramér's Coefficient V is {result}")
cramer_v.append(result)

# %% [markdown]
# **Educational Level & Class**

# %%
#Contingency Table
contingency_table4 = pd.crosstab(df['StageID'], df['Class'])
print(contingency_table4)

# %%
#Chi-square
stat, p, dof, expected = chi2_contingency(contingency_table4)
chi_value.append(stat)

# interpret p-value
alpha = 0.05
print(f"P-value is {p}")
p_value.append(p)
if p <= alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (H0 holds true)')

# %%
#Cramer's V
N = np.sum(contingency_table4.values)
minimum_dimension = min(contingency_table4.shape)-1

# Calculate Cramer's V
result = np.sqrt((stat/N) / minimum_dimension)

# Print the result
print(f"Cramér's Coefficient V is {result}")
cramer_v.append(result)

# %% [markdown]
# **Grade & Class**

# %%
#Contingency Table
contingency_table5 = pd.crosstab(df['GradeID'], df['Class'])
print(contingency_table5)

# %%
#Chi-square
stat, p, dof, expected = chi2_contingency(contingency_table5)
chi_value.append(stat)

# interpret p-value
alpha = 0.05
print(f"P-value is {p}")
p_value.append(p)
if p <= alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (H0 holds true)')

# %%
#Cramer's V
N = np.sum(contingency_table5.values)
minimum_dimension = min(contingency_table5.shape)-1

# Calculate Cramer's V
result = np.sqrt((stat/N) / minimum_dimension)

# Print the result
print(f"Cramér's Coefficient V is {result}")
cramer_v.append(result)

# %% [markdown]
# **Classroom & Class**

# %%
#Contingency Table
contingency_table6 = pd.crosstab(df['SectionID'], df['Class'])
print(contingency_table6)

# %%
#Chi-square
stat, p, dof, expected = chi2_contingency(contingency_table6)
chi_value.append(stat)

# interpret p-value
alpha = 0.05
print(f"P-value is {p}")
p_value.append(p)
if p <= alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (H0 holds true)')

# %%
#Cramer's V
N = np.sum(contingency_table6.values)
minimum_dimension = min(contingency_table6.shape)-1

# Calculate Cramer's V
result = np.sqrt((stat/N) / minimum_dimension)

# Print the result
print(f"Cramér's Coefficient V is {result}")
cramer_v.append(result)

# %% [markdown]
# **Topic & Class**

# %%
#Contingency Table
contingency_table7 = pd.crosstab(df['Topic'], df['Class'])
print(contingency_table7)

# %%
#Chi-square
stat, p, dof, expected = chi2_contingency(contingency_table7)
chi_value.append(stat)

# interpret p-value
alpha = 0.05
print(f"P-value is {p}")
p_value.append(p)
if p <= alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (H0 holds true)')

# %%
#Cramer's V
N = np.sum(contingency_table7.values)
minimum_dimension = min(contingency_table7.shape)-1

# Calculate Cramer's V
result = np.sqrt((stat/N) / minimum_dimension)

# Print the result
print(f"Cramér's Coefficient V is {result}")
cramer_v.append(result)

# %% [markdown]
# **Semester & Class**

# %%
#Contingency Table
contingency_table8 = pd.crosstab(df['Semester'], df['Class'])
print(contingency_table8)

# %%
#Chi-square
stat, p, dof, expected = chi2_contingency(contingency_table8)
chi_value.append(stat)

# interpret p-value
alpha = 0.05
print(f"P-value is {p}")
p_value.append(p)
if p <= alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (H0 holds true)')

# %%
#Cramer's V
N = np.sum(contingency_table8.values)
minimum_dimension = min(contingency_table8.shape)-1

# Calculate Cramer's V
result = np.sqrt((stat/N) / minimum_dimension)

# Print the result
print(f"Cramér's Coefficient V is {result}")
cramer_v.append(result)

# %% [markdown]
# **Relation & Class**

# %%
#Contingency Table
contingency_table9 = pd.crosstab(df['Relation'], df['Class'])
print(contingency_table9)

# %%
#Chi-square
stat, p, dof, expected = chi2_contingency(contingency_table9)
chi_value.append(stat)

# interpret p-value
alpha = 0.05
print(f"P-value is {p}")
p_value.append(p)
if p <= alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (H0 holds true)')

# %%
#Cramer's V
N = np.sum(contingency_table9.values)
minimum_dimension = min(contingency_table9.shape)-1

# Calculate Cramer's V
result = np.sqrt((stat/N) / minimum_dimension)

# Print the result
print(f"Cramér's Coefficient V is {result}")
cramer_v.append(result)

# %% [markdown]
# **Parents Answering Survey & Class**

# %%
#Contingency Table
contingency_table10 = pd.crosstab(df['ParentAnsweringSurvey'], df['Class'])
print(contingency_table10)

# %%
#Chi-square
stat, p, dof, expected = chi2_contingency(contingency_table10)
chi_value.append(stat)

# interpret p-value
alpha = 0.05
print(f"P-value is {p}")
p_value.append(p)
if p <= alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (H0 holds true)')

# %%
#Cramer's V
N = np.sum(contingency_table10.values)
minimum_dimension = min(contingency_table10.shape)-1

# Calculate Cramer's V
result = np.sqrt((stat/N) / minimum_dimension)

# Print the result
print(f"Cramér's Coefficient V is {result}")
cramer_v.append(result)

# %% [markdown]
# **Parents School Satisfaction & Class**

# %%
#Contingency Table
contingency_table11 = pd.crosstab(df['ParentschoolSatisfaction'], df['Class'])
print(contingency_table11)

# %%
#Chi-square
stat, p, dof, expected = chi2_contingency(contingency_table11)
chi_value.append(stat)

# interpret p-value
alpha = 0.05
print(f"P-value is {p}")
p_value.append(p)
if p <= alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (H0 holds true)')

# %%
#Cramer's V
N = np.sum(contingency_table11.values)
minimum_dimension = min(contingency_table11.shape)-1

# Calculate Cramer's V
result = np.sqrt((stat/N) / minimum_dimension)

# Print the result
print(f"Cramér's Coefficient V is {result}")
cramer_v.append(result)

# %% [markdown]
# **Student's Absence Days & Class**

# %%
#Contingency Table
contingency_table12 = pd.crosstab(df['StudentAbsenceDays'], df['Class'])
print(contingency_table12)

# %%
#Chi-square
stat, p, dof, expected = chi2_contingency(contingency_table12)
chi_value.append(stat)

# interpret p-value
alpha = 0.05
print(f"P-value is {p}")
p_value.append(p)
if p <= alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (H0 holds true)')

# %%
#Cramer's V
N = np.sum(contingency_table12.values)
minimum_dimension = min(contingency_table12.shape)-1

# Calculate Cramer's V
result = np.sqrt((stat/N) / minimum_dimension)

# Print the result
print(f"Cramér's Coefficient V is {result}")
cramer_v.append(result)

# %%
results_df = pd.DataFrame({
    'Variable': ['gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID', 'SectionID', 'Topic', 'Semester',
                 'Relation', 'ParentAnsweringSurvey', 'ParentschoolSatisfaction', 'StudentAbsenceDays'],
    'Chi-square': chi_value,
    'P-value': p_value,
    "Cramer's V": cramer_v
})

# Rank the variables based on Cramer's V in descending order
results_df = results_df.sort_values(by="Cramer's V", ascending=False)
results_df = results_df.reset_index(drop=True)

results_df

# %% [markdown]
# ### **Feature Selection for Numerical Value**

# %% [markdown]
# **Convert ordinal data (target) to numeric value**

# %%
dataMapping = {
    "L": 1,
    "M": 2,
    "H": 3
}

# %%
df['ClassNum'] = df['Class'].map(dataMapping)

# %%
df

# %%
spearman = []
pvalue = []

# %% [markdown]
# **Raised Hands & Class**

# %%
corr, pval = spearmanr(df['raisedhands'], df['ClassNum'])
spearman.append(corr)
pvalue.append(pval)

# print the result
print("Spearman's correlation coefficient:", corr)
print("p-value:", pval)
alpha = 0.05
if pval <= alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (H0 holds true)')

# %% [markdown]
# **Visited Resources & Class**

# %%
corr, pval = spearmanr(df['VisITedResources'], df['ClassNum'])
spearman.append(corr)
pvalue.append(pval)

# print the result
print("Spearman's correlation coefficient:", corr)
print("p-value:", pval)
alpha = 0.05
if pval <= alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (H0 holds true)')

# %% [markdown]
# **Announcements View & Class**

# %%
corr, pval = spearmanr(df['AnnouncementsView'], df['ClassNum'])
spearman.append(corr)
pvalue.append(pval)

# print the result
print("Spearman's correlation coefficient:", corr)
print("p-value:", pval)
alpha = 0.05
if pval <= alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (H0 holds true)')

# %% [markdown]
# **Discussion & Class**

# %%
corr, pval = spearmanr(df['Discussion'], df['ClassNum'])
spearman.append(corr)
pvalue.append(pval)

# print the result
print("Spearman's correlation coefficient:", corr)
print("p-value:", pval)
alpha = 0.05
if pval <= alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (H0 holds true)')

# %%
results_df2 = pd.DataFrame({
    'Variable': ['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion'],
    'P-value': pvalue,
    'Corr': spearman
})

# Rank the variables based on ANOVA in descending order
results_df2 = results_df2.sort_values(by='Corr', ascending=False)
results_df2 = results_df2.reset_index(drop=True)

results_df2

# %% [markdown]
# # **6 Features (Difference) 90:5:5**

# %% [markdown]
# ### **New DataFrame**

# %%
new_df3 = df[['StudentAbsenceDays','ParentAnsweringSurvey', 'Relation',
             'AnnouncementsView', 'raisedhands', 'VisITedResources', 'Class']].copy()

# %%
new_df3.head()

# %% [markdown]
# ### **Encoding Categorical Features**

# %%
categorical_columns = ['StudentAbsenceDays','ParentAnsweringSurvey', 'Relation']

# %%
# Convert categorical columns to 'category' data type
for col in categorical_columns:
    new_df3[col] = new_df3[col].astype('category')

# %%
new_df3.info()

# %% [markdown]
# ### **Encoding Categorical Target**

# %%
encoder = OrdinalEncoder(categories=[['L', 'M', 'H']])
new_df3['Class_encoded'] = encoder.fit_transform(new_df3[['Class']])

# %%
new_df3.head(11)

# %%
X_train, X_temp, y_train, y_temp = train_test_split(new_df3.drop(['Class', 'Class_encoded'], axis=1), new_df3['Class_encoded'], test_size=0.1, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# %%
print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Validation data shape: {X_val.shape}, {y_val.shape}")
print(f"Testing data shape: {X_test.shape}, {y_test.shape}")

# %% [markdown]
# ### **Classification with XGBoost (Initial Model)**

# %%
# Initialize and train the model
model = XGBClassifier(enable_categorical=True,
                      objective='multi:softmax',
                      num_class=len(encoder.categories_[0])
                      )

# %%
model.fit(X_train, y_train)

# %%
# Make predictions
y_pred = model.predict(X_test)
print("Predictions:", y_pred)

# %%
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score on Test Data: {accuracy}")

# %% [markdown]
# ### **Classification with XGBoost (Hyperparameter Tuning)**
# 
# 

# %%
param_grid1 = {
    'max_depth':range(1,13,1)
}

# %%
# Initialize and train the model
model1 = XGBClassifier(enable_categorical=True,
                      objective='multi:softmax',
                      num_class=len(encoder.categories_[0]),
                      seed=42,
                      learning_rate=0.01
                      )

# %%
grid_search1 = GridSearchCV(model1, param_grid1, cv=5, scoring='neg_log_loss')

# %%
grid_search1.fit(X_train, y_train,
                 eval_set=[(X_val, y_val)],
                 verbose=3)

# %%
grid_search1.cv_results_, grid_search1.best_params_, grid_search1.best_score_

# %%
param_grid2 = {
    'min_child_weight':range(1,13,1)
}

# %%
# Initialize and train the model
model2 = XGBClassifier(enable_categorical=True,
                       objective='multi:softmax',
                       num_class=len(encoder.categories_[0]),
                       seed=42,
                       learning_rate=0.01,
                       max_depth=9
                      )

# %%
grid_search2 = GridSearchCV(model2, param_grid2, cv=5, scoring='neg_log_loss')

# %%
grid_search2.fit(X_train, y_train,
                 eval_set=[(X_val, y_val)],
                 verbose=3)

# %%
grid_search2.cv_results_, grid_search2.best_params_, grid_search2.best_score_

# %%
param_grid3 = {
    'gamma': [i/10.0 for i in range(0,5)]
}

# %%
# Initialize and train the model
model3 = XGBClassifier(enable_categorical=True,
                      objective='multi:softmax',
                      num_class=len(encoder.categories_[0]),
                      max_depth=9,
                      min_child_weight=1,
                      seed=42,
                      learning_rate=0.01
                      )

# %%
grid_search3 = GridSearchCV(model3, param_grid3, cv=5, scoring='neg_log_loss')

# %%
grid_search3.fit(X_train, y_train,
                 eval_set=[(X_val, y_val)],
                 verbose=3)

# %%
grid_search3.cv_results_, grid_search3.best_params_, grid_search3.best_score_

# %%
param_grid4 = {
    'subsample':[i/10.0 for i in range(6,10)]
}

# %%
# Initialize and train the model
model4 = XGBClassifier(enable_categorical=True,
                      objective='multi:softmax',
                      num_class=len(encoder.categories_[0]),
                      max_depth=9,
                      min_child_weight=1,
                      seed=42,
                      learning_rate=0.01,
                      gamma=0.3
                      )

# %%
grid_search4 = GridSearchCV(model4, param_grid4, cv=5, scoring='neg_log_loss')

# %%
grid_search4.fit(X_train, y_train,
                 eval_set=[(X_val, y_val)],
                 verbose=3)

# %%
grid_search4.cv_results_, grid_search4.best_params_, grid_search4.best_score_

# %%
param_grid5 = {
    'colsample_bytree':[i/10.0 for i in range(6,10)]
}

# %%
# Initialize and train the model
model5 = XGBClassifier(enable_categorical=True,
                      objective='multi:softmax',
                      num_class=len(encoder.categories_[0]),
                      max_depth=9,
                      min_child_weight=1,
                      seed=42,
                      learning_rate=0.01,
                      gamma=0.3,
                      subsample=0.7
                      )

# %%
grid_search5 = GridSearchCV(model5, param_grid5, cv=5, scoring='neg_log_loss')

# %%
grid_search5.fit(X_train, y_train,
                 eval_set=[(X_val, y_val)],
                 verbose=3)

# %%
grid_search5.cv_results_, grid_search5.best_params_, grid_search5.best_score_

# %%
param_grid6 = {
    'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]
}

# %%
# Initialize and train the model
model6 = XGBClassifier(enable_categorical=True,
                      objective='multi:softmax',
                      num_class=len(encoder.categories_[0]),
                      max_depth=9,
                      min_child_weight=1,
                      seed=42,
                      learning_rate=0.01,
                      gamma=0.3,
                      subsample=0.7,
                      colsample_bytree=0.9
                      )

# %%
grid_search6 = GridSearchCV(model6, param_grid6, cv=5, scoring='neg_log_loss')

# %%
grid_search6.fit(X_train, y_train,
                 eval_set=[(X_val, y_val)],
                 verbose=3)

# %%
grid_search6.cv_results_, grid_search6.best_params_, grid_search6.best_score_

# %%
param_grid7 = {
    'reg_lambda': [1e-5, 1e-2, 0.1, 1, 100]
}

# %%
# Initialize and train the model
model7 = XGBClassifier(enable_categorical=True,
                      objective='multi:softmax',
                      num_class=len(encoder.categories_[0]),
                      max_depth=9,
                      min_child_weight=1,
                      seed=42,
                      learning_rate=0.01,
                      gamma=0.3,
                      subsample=0.7,
                      colsample_bytree=0.9,
                      reg_alpha=1e-05
                      )

# %%
grid_search7 = GridSearchCV(model7, param_grid7, cv=5, scoring='neg_log_loss')

# %%
grid_search7.fit(X_train, y_train,
                 eval_set=[(X_val, y_val)],
                 verbose=3)

# %%
grid_search7.cv_results_, grid_search7.best_params_, grid_search7.best_score_

# %%
param_grid8 = {
    'n_estimators':range(100,1200,100)
}

# %%
# Initialize and train the model
model8 = XGBClassifier(enable_categorical=True,
                      objective='multi:softmax',
                      num_class=len(encoder.categories_[0]),
                      max_depth=9,
                      min_child_weight=1,
                      seed=42,
                      learning_rate=0.01,
                      gamma=0.3,
                      subsample=0.7,
                      colsample_bytree=0.9,
                      reg_alpha=1e-05,
                      reg_lambda=1e-05
                      )

# %%
grid_search8 = GridSearchCV(model8, param_grid8, cv=5, scoring='neg_log_loss')

# %%
grid_search8.fit(X_train, y_train,
                 eval_set=[(X_val, y_val)],
                 verbose=3)

# %%
grid_search8.cv_results_, grid_search8.best_params_, grid_search8.best_score_

# %%
# Make predictions
best_model = grid_search8.best_estimator_
y_pred = best_model.predict(X_test)
print("Predictions:", y_pred)

# %%
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score on Test Data: {accuracy}")

# %% [markdown]
# ### **Evaluation**

# %%
# Generate the classification report
print(classification_report(y_test, y_pred))

# Generate and plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=encoder.categories_[0], yticklabels=encoder.categories_[0])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# %% [markdown]
# # **Imbalanced(?)**

# %%
print(new_df3['Class'].value_counts())

# %%
df_newimb = pd.concat([
    new_df3[new_df3['Class'] == 'M'].sample(n=170, random_state=42),  # Select 170 random samples from class M
    new_df3[new_df3['Class'] == 'L'],  # Select all samples from class L
    new_df3[new_df3['Class'] == 'H']   # Select all samples from class H
])

# Now df_new contains the selected data
df_newimb

# %%
print(df_newimb['Class'].value_counts())

# %%
df_newimb.info()

# %%
X_train2, X_temp2, y_train2, y_temp2 = train_test_split(df_newimb.drop(['Class', 'Class_encoded'], axis=1), df_newimb['Class_encoded'], test_size=0.1, random_state=42)
X_val2, X_test2, y_val2, y_test2 = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# %%
print(f"Training data shape: {X_train2.shape}, {y_train2.shape}")
print(f"Validation data shape: {X_val2.shape}, {y_val2.shape}")
print(f"Testing data shape: {X_test2.shape}, {y_test2.shape}")

# %% [markdown]
# ### **Classification with XGBoost (Initial Model)**

# %%
# Initialize and train the model
imb_model = XGBClassifier(enable_categorical=True,
                          objective='multi:softmax',
                          num_class=len(encoder.categories_[0])
                         )

# %%
imb_model.fit(X_train2, y_train2)

# %%
# Make predictions
y_pred2 = imb_model.predict(X_test2)
print("Predictions:", y_pred2)

# %%
accuracy2 = accuracy_score(y_test2, y_pred2)
print(f"Accuracy Score on Test Data: {accuracy2}")

# %% [markdown]
# ### **Classification with XGBoost (Hyperparameter Tuning)**
# 
# 

# %%
imb_param_grid1 = {
    'max_depth':range(1,13,1)
}

# %%
# Initialize and train the model
imb_model1 = XGBClassifier(enable_categorical=True,
                           objective='multi:softmax',
                           num_class=len(encoder.categories_[0]),
                           seed=42,
                           learning_rate=0.01
                          )

# %%
imb_grid_search1 = GridSearchCV(imb_model1, imb_param_grid1, cv=5, scoring='neg_log_loss')

# %%
imb_grid_search1.fit(X_train2, y_train2,
                     eval_set=[(X_val2, y_val2)],
                     verbose=3)

# %%
imb_grid_search1.cv_results_, imb_grid_search1.best_params_, imb_grid_search1.best_score_

# %%
imb_param_grid2 = {
    'min_child_weight':range(1,13,1)
}

# %%
# Initialize and train the model
imb_model2 = XGBClassifier(enable_categorical=True,
                       objective='multi:softmax',
                       num_class=len(encoder.categories_[0]),
                       seed=42,
                       learning_rate=0.01,
                       max_depth=7
                      )

# %%
imb_grid_search2 = GridSearchCV(imb_model2, imb_param_grid2, cv=5, scoring='neg_log_loss')

# %%
imb_grid_search2.fit(X_train2, y_train2,
                     eval_set=[(X_val2, y_val2)],
                     verbose=3)

# %%
imb_grid_search2.cv_results_, grid_search2.best_params_, grid_search2.best_score_

# %%
imb_param_grid3 = {
    'gamma': [i/10.0 for i in range(0,5)]
}

# %%
# Initialize and train the model
imb_model3 = XGBClassifier(enable_categorical=True,
                      objective='multi:softmax',
                      num_class=len(encoder.categories_[0]),
                      max_depth=7,
                      min_child_weight=1,
                      seed=42,
                      learning_rate=0.01
                      )

# %%
imb_grid_search3 = GridSearchCV(imb_model3, imb_param_grid3, cv=5, scoring='neg_log_loss')

# %%
imb_grid_search3.fit(X_train2, y_train2,
                     eval_set=[(X_val2, y_val2)],
                     verbose=3)

# %%
imb_grid_search3.cv_results_, imb_grid_search3.best_params_, imb_grid_search3.best_score_

# %%
imb_param_grid4 = {
    'subsample':[i/10.0 for i in range(6,10)]
}

# %%
# Initialize and train the model
imb_model4 = XGBClassifier(enable_categorical=True,
                      objective='multi:softmax',
                      num_class=len(encoder.categories_[0]),
                      max_depth=7,
                      min_child_weight=1,
                      seed=42,
                      learning_rate=0.01,
                      gamma=0.3
                      )

# %%
imb_grid_search4 = GridSearchCV(imb_model4, imb_param_grid4, cv=5, scoring='neg_log_loss')

# %%
imb_grid_search4.fit(X_train2, y_train2,
                     eval_set=[(X_val2, y_val2)],
                     verbose=3)

# %%
imb_grid_search4.cv_results_, imb_grid_search4.best_params_, imb_grid_search4.best_score_

# %%
imb_param_grid5 = {
    'colsample_bytree':[i/10.0 for i in range(6,10)]
}

# %%
# Initialize and train the model
imb_model5 = XGBClassifier(enable_categorical=True,
                      objective='multi:softmax',
                      num_class=len(encoder.categories_[0]),
                      max_depth=7,
                      min_child_weight=1,
                      seed=42,
                      learning_rate=0.01,
                      gamma=0.3,
                      subsample=0.8
                      )

# %%
imb_grid_search5 = GridSearchCV(imb_model5, imb_param_grid5, cv=5, scoring='neg_log_loss')

# %%
imb_grid_search5.fit(X_train2, y_train2,
                     eval_set=[(X_val2, y_val2)],
                     verbose=3)

# %%
imb_grid_search5.cv_results_, imb_grid_search5.best_params_, imb_grid_search5.best_score_

# %%
imb_param_grid6 = {
    'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]
}

# %%
# Initialize and train the model
imb_model6 = XGBClassifier(enable_categorical=True,
                      objective='multi:softmax',
                      num_class=len(encoder.categories_[0]),
                      max_depth=7,
                      min_child_weight=1,
                      seed=42,
                      learning_rate=0.01,
                      gamma=0.3,
                      subsample=0.8,
                      colsample_bytree=0.9
                      )

# %%
imb_grid_search6 = GridSearchCV(imb_model6, imb_param_grid6, cv=5, scoring='neg_log_loss')

# %%
imb_grid_search6.fit(X_train2, y_train2,
                     eval_set=[(X_val2, y_val2)],
                     verbose=3)

# %%
imb_grid_search6.cv_results_, imb_grid_search6.best_params_, imb_grid_search6.best_score_

# %%
imb_param_grid7 = {
    'reg_lambda': [1e-5, 1e-2, 0.1, 1, 100]
}

# %%
# Initialize and train the model
imb_model7 = XGBClassifier(enable_categorical=True,
                      objective='multi:softmax',
                      num_class=len(encoder.categories_[0]),
                      max_depth=7,
                      min_child_weight=1,
                      seed=42,
                      learning_rate=0.01,
                      gamma=0.3,
                      subsample=0.8,
                      colsample_bytree=0.9,
                      reg_alpha=1e-05
                      )

# %%
imb_grid_search7 = GridSearchCV(imb_model7, imb_param_grid7, cv=5, scoring='neg_log_loss')

# %%
imb_grid_search7.fit(X_train2, y_train2,
                     eval_set=[(X_val2, y_val2)],
                     verbose=3)

# %%
imb_grid_search7.cv_results_, imb_grid_search7.best_params_, imb_grid_search7.best_score_

# %%
imb_param_grid8 = {
    'n_estimators':range(100,1200,100)
}

# %%
# Initialize and train the model
imb_model8 = XGBClassifier(enable_categorical=True,
                      objective='multi:softmax',
                      num_class=len(encoder.categories_[0]),
                      max_depth=7,
                      min_child_weight=1,
                      seed=42,
                      learning_rate=0.01,
                      gamma=0.3,
                      subsample=0.8,
                      colsample_bytree=0.9,
                      reg_alpha=1e-05,
                      reg_lambda=1e-05
                      )

# %%
imb_grid_search8 = GridSearchCV(imb_model8, imb_param_grid8, cv=5, scoring='neg_log_loss')

# %%
imb_grid_search8.fit(X_train2, y_train2,
                     eval_set=[(X_val2, y_val2)],
                     verbose=3)

# %%
imb_grid_search8.cv_results_, imb_grid_search8.best_params_, imb_grid_search8.best_score_

# %%
# Make predictions
best_model2 = imb_grid_search8.best_estimator_
y_pred2 = best_model2.predict(X_test2)
print("Predictions:", y_pred2)

# %%
accuracy2 = accuracy_score(y_test2, y_pred2)
print(f"Accuracy Score on Test Data: {accuracy2}")

# %% [markdown]
# ### **Evaluation**

# %%
# Generate the classification report
print(classification_report(y_test2, y_pred2))

# Generate and plot the confusion matrix
cm = confusion_matrix(y_test2, y_pred2)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=encoder.categories_[0], yticklabels=encoder.categories_[0])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


