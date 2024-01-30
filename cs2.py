import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('bankdata.csv')
print("CMSC 176 Case Study 2")

# age_mean = df['age'].mean()
# age_sd = df['age'].std()
# age_med = df['age'].median()
# age_var = df['age'].var()
# age_min = df['age'].min()
# age_max = df[ 'age'].max()
# age_range = age_max - age_min
# age_sum = df['age'].sum()
# age_count = len (df)

# print()
# print ("Mean:", age_mean)
# print ("Standard Deviation:", "{:.2f}". format (age_sd)) 
# print("Median:" , age_med)
# print("Variance:","{:.2f}".format(age_var))
# print ("Range:" , age_ran)
# print ("Minimum: " , age_min)
# print ("Maximum:" , age_max)
# print ("Sum:" , age_sum)
# print ("Count:", age_count)

# income_mean = df['income'].mean()
# income_sd = df['income'].std()
# income_med = df['income'].median()
# income_var = df['income'].var()
# income_min = df['income'].min()
# income_max = df['income'].max()
# income_range = income_max - income_min
# income_sum = df['income'].sum()
# income_count = len (df)

# print()
# print ("Mean:" , inc_mean)
# print("Standard Deviation:","{:2f}".format(inc_sd))
# print ("Median:" , inc_med)
# print("Variance:","{:.2f}".format(inc_var))
# print ("Range:" , inc_ran)
# print ("Minimum:" , inc_min)
# print ("Maximum:" , inc_max)
# print ("Sum: " , inc_sum) 
# print ("Count:", inc_count)

# children_mean = df['children'].mean()
# children_sd = df['children'].std()
# children_med = df['children'].median()
# children_var = df['children'].var()
# children_min = df['children'].min()
# children_max = df['children'].max()
# children_range = children_max - children_min
# children_sum = df['children'].sum()
# children_count = len(df)

# print()
# print ("Mean:",chi_mean)
# print("Standard Deviation:","{:.2f}".format (chi_sd))
# print ("Median:", chi_med)
# print ("Variance:","{:.2f}".format(chi_var))
# print ("Range:" , chi_ran)
# print("Minimum: " , chi_min)
# print ("Maximum:", chi_max)
# print ("Sum: " , chi_sum)
# print ("Count:", chi_count)

# table = pd.pivot_table(df, index=['married'], values=['children'], aggfunc ='sum')
# table2 = df.groupby( 'children' )['married'].value_counts()
# print(table2)
# print()
# print (table)

# df = df.sort_values('income')
# point1 = df.iloc[0]['income']
# point2 = df.iloc[199]['income']
# point3 = df.iloc[399]['income']
# point4 = df.iloc[599]['income']
# print()
# print("Q1:" , point1)
# print("Q2:" , point2)
# print ("Q3:", point3)
# print ("Q4:" , point4)

# pd. set_option('display.max_rows', None)
# df = pd. read_csv ('bankdata.csv')
# print()
# print(pd.get_dummies(df['region ']))

##################################################################################################################################
# Main Code For Processing the Data per Question
##################################################################################################################################

# What is the range of values of the Age variable? 
# What is the minimum, maximum and middle value?
age_var = df['age'].var()
age_min = df['age'].min()
age_max = df[ 'age'].max()
age_med = df['age'].median()
age_range = age_max - age_min

print()
print ("Range:", age_range) 
print("Minimum:" , age_min)
print("Maximum:", age_max)
print ("Median:", age_med)
print()


# How many customers have a savings account? Current account?
save_no = df['save_act'].value_counts()
current_no = df['current_act'].value_counts()

print()
print("Savings account info:", save_no)
print("Current account info:", current_no)
print()

# Create a pivot report for the relationship of the variable “Married” and the Number of Children
#text
table = pd.pivot_table(df, index=['married'], values=['children'], aggfunc ='sum')
table2 = df.groupby( 'children' )['married'].value_counts()
print(table2)
print()
print(table)

# Calculate the means of Age, Income and Children by PEP, Married and has Car.
grouped_data = df.groupby(['pep', 'married', 'car'])
age_sd = df['age'].std()
income_sd = df['income'].std()
children_sd = df['children'].std()

for group, data in grouped_data:
    print(f"Group: {group}")
   
    sorted_age = data.sort_values(by='age')
    mean_age = sorted_age['age'].mean()
    print("Mean of Age:", round(mean_age, 2), end="")
    print(" ±", round(mean_age, 2))
    
    sorted_income = data.sort_values(by='income')
    mean_income = sorted_income['income'].mean()
    print("Mean of Income:", round(mean_income, 2), end="")
    print(" ±", round(mean_age, 2))

    sorted_children = data.sort_values(by='children')
    mean_children = sorted_children['children'].mean()
    print("Mean of Children:", round(mean_children, 2), end="")
    print(" ±", round(mean_age, 2))
    print()

# As Age increases, what pattern do you see in terms of buying a PEP?
# text
age_bins = [18, 26, 33, 40, 47, 54, 61, 68]
age_labels = ['18-25', '26-32', '33-39', '40-46', '47-53', '54-60', '61-67']

df['custom_age_group'] = pd.cut(df['age'], bins = age_bins, labels = age_labels, right = False)
grouped_data = df.groupby(['custom_age_group'], observed=False)  

for age_group, data in grouped_data:
    print(f"Age Group: {age_group}")
    
    count_yes = (data['pep'] == 'YES').sum()
    count_no = (data['pep'] == 'NO').sum()
    
    print("YES:", count_yes)
    print("NO:", count_no)
    print()

#graph
age_groups = ['18-25', '26-32', '33-39', '40-46', '47-53', '54-60', '61-67']
yes_counts = [33, 28, 38, 45, 39, 37, 54]
no_counts = [64, 50, 48, 51, 43, 34, 36]

bar_width = 0.35
index = range(len(age_groups))

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(index, yes_counts, bar_width, label='YES', alpha=0.7)

ax.bar([i + bar_width for i in index], no_counts, bar_width, label='NO', alpha=0.7)

ax.set_xlabel('Age Group')
ax.set_ylabel('Count')
ax.set_title('Counts of YES and NO by Age Group')
ax.set_xticks([i + bar_width / 2 for i in index])
ax.set_xticklabels(age_groups)
ax.legend()

plt.tight_layout()
plt.show()

# # In terms of the number of children what pattern do you see in terms of buying a PEP? For Being Married?
# text-children
children_bins = [0, 1, 2, 3, 4]
children_labels = ['0', '1', '2', '3']

df['num_of_children'] = pd.cut(df['children'], bins = children_bins, labels = children_labels, right = False)
grouped_data = df.groupby(['num_of_children'], observed=False)  

for num_of_children, data in grouped_data:
    print(f"No. of children: {num_of_children}")
    
    count_yes = (data['pep'] == 'YES').sum()
    count_no = (data['pep'] == 'NO').sum()
    
    print("YES:", count_yes)
    print("NO:", count_no)
    print()

#chart-children
grouped_data = df.groupby(['num_of_children'], observed=False)

yes_counts = []
no_counts = []
children_labels = []
bar_width = 0.35  # Set the width for each bar

for i, (num_of_children, data) in enumerate(grouped_data):
    children_labels.append(num_of_children)
    count_yes = (data['pep'] == 'YES').sum()
    yes_counts.append(count_yes)
    count_no = (data['pep'] == 'NO').sum()
    no_counts.append(count_no)

x = range(len(children_labels))
x_yes = [x_i - bar_width/2 for x_i in x]  # Position for 'YES' bars
x_no = [x_i + bar_width/2 for x_i in x]   # Position for 'NO' bars

fig, ax = plt.subplots(figsize=(10, 6))

plt.bar(x_yes, yes_counts, bar_width, label='YES')
plt.bar(x_no, no_counts, bar_width, label='NO')

plt.xlabel('Number of Children')
plt.ylabel('Count')
plt.title('Counts of YES and NO by Number of Children')
plt.xticks(x, children_labels)
plt.legend()

plt.tight_layout()
plt.show()

#text-married
grouped_data = df.groupby(['married', 'pep'])

counts = {}

for (married_status, pep_status), data in grouped_data:
    if married_status not in counts:
        counts[married_status] = {}
    counts[married_status][pep_status] = len(data)

for married_status, pep_counts in counts.items():
    print(f"Married {married_status}")
    for pep_status, count in pep_counts.items():
        print(f"PEP {pep_status}: {count}")
    print()

#chart-married
grouped_data = df.groupby(['married', 'pep'])

counts = {}

for (married_status, pep_status), data in grouped_data:
    if married_status not in counts:
        counts[married_status] = {}
    counts[married_status][pep_status] = len(data)

marital_statuses = df['married'].unique()
pep_statuses = df['pep'].unique()

data = np.array([[counts[marital_status].get(pep_status, 0) for pep_status in pep_statuses] for marital_status in marital_statuses])

bar_width = 0.35
index = np.arange(len(marital_statuses))

fig, ax = plt.subplots(figsize=(10, 6))
for i, pep_status in enumerate(pep_statuses):
    plt.bar(index + i * bar_width, data[:, i], bar_width, label=f'PEP {pep_status}')

plt.xlabel('Marital Status')
plt.ylabel('Count')
plt.title('Counts of PEP (YES/NO) by Marital Status')
plt.xticks(index + bar_width, marital_statuses)
plt.legend(title='PEP')

plt.tight_layout()
plt.show()

# To Normalize the Income Column into a [0,1] scale.
income_column = df['income']

min_income = income_column.min()
max_income = income_column.max()

    # It then normalizes the 'income' column to a [0, 1] scale by applying the 
    # min-max scaling formula to create a new column called 'normalized_income' 
    # in the DataFrame. This transformation ensures that all 'income' values are 
    # rescaled to fall within the [0, 1] range.
df['normalized_income'] = (income_column - min_income) / (max_income - min_income)

print(df[['income', 'normalized_income']])

result_df = df[['income', 'normalized_income']]

csv_file_path = 'normalized_income.csv'
result_df.to_csv(csv_file_path, index=False)

# Suppose that we want to create an equal depth (frequency) variable for Income 
# where the new variable could take in “Low”, “Medium” and “High.”
quantiles = df['income'].quantile([0, 1/3, 2/3, 1])

bins = [quantiles.iloc[0], quantiles.iloc[1], quantiles.iloc[2], quantiles.iloc[3]]
labels = ['Low', 'Medium', 'High']

df['income_Category'] = pd.cut(df['income'], bins=bins, labels=labels, include_lowest=True)

print(df[['income', 'income_Category']])

intervals = {}
for i in range(len(bins) - 1):
    interval_label = f'{bins[i]} - {bins[i+1]}'
    intervals[interval_label] = labels[i]

# Print the intervals and corresponding labels
print("\nIntervals and Labels:")
for interval, label in intervals.items():
    print(f"{interval}: {label}")

# Count the number of incomes under each label
income_counts = df['income_Category'].value_counts()

print("\nIncome Counts:")
print(income_counts)
print()

# Suppose that we want to create dummy variables for the four values of Region
# Create dummy variables for the 'region' column
region_dummies = pd.get_dummies(df['region'], prefix='Region')

# Replace True with 1 and False with 0
region_dummies = region_dummies.astype(int)

df = pd.concat([df, region_dummies], axis=1)

columns_to_keep = ['id'] + list(region_dummies.columns)
df = df[columns_to_keep]

df = df.iloc[:, 2:]

# Export to a CSV file
df.to_csv('dummy_variables.csv', index=False)
print(df)
