<h1>IBM Attrition HR Data Analytics</h1>
<br>
<h2>Importing Resources</h2>

```Python

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
df= pd.read_csv('IBM HR Dataset.csv')
df

```
<br>

<img width="1746" height="592" alt="image" src="https://github.com/user-attachments/assets/4aee8e87-24ab-48c9-b330-486aff85923a" />

<br>

<h2>Data Cleaning</h2>

```python

df.isna().sum()

```
<br>
<img width="300" height="781" alt="image" src="https://github.com/user-attachments/assets/59e3c244-c7cb-4239-89fa-894ff4dd7fa4" />

<br>
<br>

```Python

#Checking if there's any outlier
rules = {
    "Age": (18, 60),
    "MonthlyIncome": (1, None),
    "DistanceFromHome": (1, 30),
    "PercentSalaryHike": (0, 100),
    "TotalWorkingYears": (0, 40),
    "YearsAtCompany": (0, 40),
    "JobLevel": (1, 5),
    "JobSatisfaction": (1, 4),
    "PerformanceRating": (1, 4)
}

out_of_range = {}

for col, (low, high) in rules.items():
    if col in df.columns:
        condition = False
        if low is not None:
            condition |= df[col] < low
        if high is not None:
            condition |= df[col] > high
        
        out_of_range[col] = df[condition]

for col, data in out_of_range.items():
    print(col, "→", data.shape[0], "out-of-range rows")
    
#No outlier proceed with EDA
```
<br>

<img width="392" height="204" alt="image" src="https://github.com/user-attachments/assets/0861cb76-85a6-4834-b621-af9f74886942" />

<br>

<h2>Exploratory Data Analysis</h2>

<br>

<p>
  Univariate Analysis has already been done In sql Ref : 'SQL Analytics.sql'

By which Following Parameters are telling some stories

1] Job Role

2] Marital Status

3] Age Group

4] Business Travel

5] Monthly Income

6] Years At Company

7] Total Experience Group

8] Years Since Current Role

9] Number of companies changed before

Insights :

1.Sales representatives exhibit a higher risk of attrition compared to other roles.

2.Single employees show a higher likelihood of attrition than married employees.

3.Employees aged above 31 demonstrate a higher tendency toward attrition.

4.Employees who travel frequently are more susceptible to attrition.

5.Employees in lower monthly income brackets have a higher attrition risk.

6.Newly hired employees are more likely to leave the organization.

7.Employees with limited industry experience are more vulnerable to attrition.

8.Attrition is most prevalent within the first three years of an employee’s current role.

9.Employees with frequent job changes, especially recent joiners, show higher attrition susceptibility.
</p>
<h3>Now Let's move ahead with bivariate and multivariate analysis to deepen the understanding</h3>
<br>
<h2>Hypothesis 1 : Single and divorced employees with low age tend to cause more attrition than married ones</h2>
<br>

```Python

#finding the low age threshold
low_age_threshold = df['Age'].median()   # or 30
#Segregating Low and High Values
df['AgeGroup'] = df['Age'].apply(lambda x: 'Low' if x <= df['Age'].median() else 'High')
#finding mean of attrition rate for different marital status
attrition_rate = (
    df.groupby('MaritalStatus')['Attrition']
      .apply(lambda x: (x == 'Yes').mean())
      .sort_values(ascending=False)
)

print(attrition_rate)

```
<br>

<img width="283" height="107" alt="image" src="https://github.com/user-attachments/assets/f236b446-b987-402f-a7f5-5233d79d02c4" />

<br>
<br>

```python

pivot = pd.pivot_table(
    df,
    values='Attrition',
    index=['MaritalStatus', 'AgeGroup'],
    aggfunc=lambda x: (x == 'Yes').mean()
)

print(pivot)

```
<br>
<img width="301" height="166" alt="image" src="https://github.com/user-attachments/assets/05e1c68d-2c25-47b8-926b-fbf34de63d6e" />
<br>
<br>

```python

# Chi-square Test to validate the hypothesis for marital status and Attrition
from scipy.stats import chi2_contingency

table = pd.crosstab(df['MaritalStatus'], df['Attrition'])
chi2, p, dof, expected = chi2_contingency(table)

p
```

Output : np.float64(9.45551106034083e-11)
<br>
<br>

```Python

# Chi-square Test to validate the hypothesis

low_age_df = df[df['AgeGroup'] == 'Low']
table_low = pd.crosstab(low_age_df['MaritalStatus'], low_age_df['Attrition'])
chi2, p, _, _ = chi2_contingency(table_low)

p

```
Output : np.float64(2.7407564224454254e-08)

<p>We got

p= 2.740756 x 10-8

Compare with Common thresholds

p < 0.05 -> Statstically Significant

p < 0.01 -> Significant enough to to take action

“The chi-square test indicates a statistically significant association between marital status and attrition (p < 0.001). This association remains significant within the low-age employee segment, supporting the hypothesis that younger single and divorced employees experience higher attrition than married employees.”
</p>
<h4>Thus Hypothesis 1 is true</h4>
<br>
<h2>Hypothesis 2 : Sales people who travel frequently are associated with high risk of attrition</h2>
<br>

```Python

sales_df=df[df['Department']=='Sales']
# Finding Mean of Attrition rate for Different frequency of Business Travel
attrition_rate=(
    sales_df.groupby('BusinessTravel')['Attrition']
    .apply(lambda x : ( x=='Yes').mean())
    .sort_values(ascending=False)
)
attrition_rate

```
<br>
<br>

<img width="294" height="123" alt="image" src="https://github.com/user-attachments/assets/f3ffc587-5cb9-4318-a734-61c12abca5fe" />

<br>
<br>

```python

# Visual representation of Attrition Rate and Business travel Frequency
attrition_rate.plot(kind='bar')
plt.title('Attrition Rate by Business Travel (Sales Department)')
plt.ylabel('Attrition Rate')
plt.xlabel('Business Travel')
plt.show()

```

<br>
<br>  

<img width="724" height="703" alt="image" src="https://github.com/user-attachments/assets/51552499-c463-4094-abb0-23975f852c0b" />

<br>
<br>

```Python

# The chi- Square Test to validate the hypothesis 2
table = pd.crosstab(
    sales_df['BusinessTravel'],
    sales_df['Attrition']
)

chi2, p, dof, expected = chi2_contingency(table)
p

```

Output : np.float64(0.0015205437920981172)

<p>
  We got

p= 0.0015205437920981172

Compare with Common thresholds

p < 0.05 -> Statstically Significant

p < 0.01 -> Significant enough to to take action

“Within the Sales department, business travel frequency is significantly associated with attrition (χ² test, p ≈ 0.0015). Employees who travel frequently exhibit the highest attrition risk compared to those who travel rarely or not at all.”
</p>
<h4>Hypothesis 2 is True</h4>
<br>
<h2>Hypothesis 3 : Employees with relatively low monthly income and significant number of years in company are at risk of attrition.</h2>

```Python

# Segregating employees into high and low tenures and Monthly income
income_threshold = df['MonthlyIncome'].median()
tenure_threshold = df['YearsAtCompany'].median()

df['IncomeGroup'] = df['MonthlyIncome'].apply(
    lambda x: 'Low Income' if x <= income_threshold else 'High Income'
)

df['TenureGroup'] = df['YearsAtCompany'].apply(
    lambda x: 'High Tenure' if x >= tenure_threshold else 'Low Tenure'
)
risk_df = df[
    (df['IncomeGroup'] == 'Low Income') &
    (df['TenureGroup'] == 'High Tenure')
]
# Creating Pivot table for Income Group, Tenure Group and Attrition
pivot = pd.pivot_table(
    df,
    values='Attrition',
    index=['IncomeGroup', 'TenureGroup'],
    aggfunc=lambda x: (x == 'Yes').mean()
)

pivot

```
<br>
<br>

<img width="336" height="229" alt="image" src="https://github.com/user-attachments/assets/133a7ea2-e244-41ef-ad61-f6242a9f7ead" />

<br>
<br>

```Python

#Visualizing Monthly Income vs Attrition for high Tenure Employees

high_tenure_df = df[df['TenureGroup'] == 'High Tenure']

sns.boxplot(x='Attrition', y='MonthlyIncome', data=high_tenure_df)
plt.title('Monthly Income vs Attrition (High Tenure Employees)')
plt.show()

```


<img width="739" height="569" alt="image" src="https://github.com/user-attachments/assets/b7d500ed-6f32-4696-9d29-3a445af8101f" />

<br>
<br>

```Python

# Scatter plot of Income vs Tenure with Attrition
sns.scatterplot(
    data=df,
    x='YearsAtCompany',
    y='MonthlyIncome',
    hue='Attrition'
)
plt.title('Income vs Tenure with Attrition')
plt.show()

```

<br>


<img width="739" height="576" alt="image" src="https://github.com/user-attachments/assets/062cf296-b532-4e58-8d6e-a71173614533" />

<br>
<br>

```Python

df['RiskGroup'] = (
    (df['IncomeGroup'] == 'Low Income') &
    (df['TenureGroup'] == 'High Tenure')
)
# The Chi-Square Test to validate Hypothesis 3
table = pd.crosstab(df['RiskGroup'], df['Attrition'])
chi2, p, dof, expected = chi2_contingency(table)
p

```
Output : np.float64(0.08591350686435932)

<p>
  We got:

p = 0.0859

Compare with common thresholds:

0.05 → not significant

0.10 → marginally significant (weak evidence)

Hypothesis:

Employees with low monthly income and high tenure are at higher risk of attrition

Result:

NOT strongly supported statistically

This is important:

Directionally true, but not statistically strong enough.
</p>

<h4>Hypothesis 3 is not True</h4>

<br>

<h2> Hypothesis 4 : Employees with relative young age who have changes significant number of companies before are at high attrition risk.</h2>

```Python

# Seggregating employees based on Age and Number of companies worked before

age_threshold = df['Age'].median()
company_threshold = df['NumCompaniesWorked'].median()

df['AgeGroup'] = df['Age'].apply(
    lambda x: 'Young' if x <= age_threshold else 'Older'
)

df['CompanyChangeGroup'] = df['NumCompaniesWorked'].apply(
    lambda x: 'High Changes' if x >= company_threshold else 'Low Changes'
)
risk_df = df[
    (df['AgeGroup'] == 'Young') &
    (df['CompanyChangeGroup'] == 'High Changes')
]
# Creating pivot table for Attrition based on Age Group and Company Changed Group

pivot = pd.pivot_table(
    df,
    values='Attrition',
    index=['AgeGroup', 'CompanyChangeGroup'],
    aggfunc=lambda x: (x == 'Yes').mean()
)

pivot

```
<br>
<br>

<img width="380" height="243" alt="image" src="https://github.com/user-attachments/assets/559b8b33-8f06-4383-87bb-736b51e5c41d" />

<br>
<br>

```Python

# Visializing pivot Table
pivot.plot(kind='bar')
plt.title('Attrition Rate by Age & Company Change History')
plt.ylabel('Attrition Rate')
plt.xlabel('Employee Segment')
plt.xticks(rotation=45)
plt.show()

```

<img width="728" height="695" alt="image" src="https://github.com/user-attachments/assets/a767462e-7ba0-4ccc-b6ad-aa3c24956a2c" />
<br>
<br>

```Python

df['RiskGroup'] = (
    (df['AgeGroup'] == 'Young') &
    (df['CompanyChangeGroup'] == 'High Changes')
)
# The Chi-Square test to validate Hypothesis 4
table = pd.crosstab(df['RiskGroup'], df['Attrition'])
chi2, p, dof, expected = chi2_contingency(table)
p

```

Output : np.float64(0.0009968897509854534)

<p>
  We got:

p = 0.0009968897509854534

This is:

< 0.05 → statistically significant

< 0.01 → strongly significant

So:

We reject the null hypothesis with high confidence.

️Final decision on Hypothesis 4 Hypothesis

Employees with relatively young age and a history of frequent company changes are at higher attrition risk

Result

SUPPORTED (strong evidence)

The probability that this relationship occurred by chance is < 0.1%.

“The analysis shows a statistically significant association between age, prior company changes, and attrition (χ² test, p < 0.001). Younger employees with a history of frequent job changes exhibit the highest attrition risk.”
</p>
<h4>Hypothesis 4 is True</h4>
<br>

<h2>Hypothesis 5 : Employees with low job satisfaction, low environment satisfaction, low job involvement and low percentage hike are at high risk of attrition.</h2>

```Python
# Seggregating Employees based on job satisfaction, Environment, Job involvement and low percentage hike
df['LowJobSatisfaction'] = df['JobSatisfaction'] <= 2
df['LowEnvSatisfaction'] = df['EnvironmentSatisfaction'] <= 2
df['LowJobInvolvement'] = df['JobInvolvement'] <= 2
df['LowHike'] = df['PercentSalaryHike'] <= df['PercentSalaryHike'].median()
df['DissatisfactionRisk'] = (
    df['LowJobSatisfaction'] &
    df['LowEnvSatisfaction'] &
    df['LowJobInvolvement'] &
    df['LowHike']
)
# Finding attrition rate based on dissatisfaction Risk
attrition_rate = (
    df.groupby('DissatisfactionRisk')['Attrition']
      .apply(lambda x: (x == 'Yes').mean())
)

attrition_rate

```

<img width="300" height="106" alt="image" src="https://github.com/user-attachments/assets/1a1df102-2b7c-4422-a2bc-79d58479d992" />

<br>
<br>

```python

# Visual Representation of Attrition Rate Dissatisfied Employeee
attrition_rate.plot(kind='bar')
plt.title('Attrition Risk for Dissatisfied Employees')
plt.ylabel('Attrition Rate')
plt.xlabel('High Dissatisfaction Group')
plt.show()

```

<img width="728" height="694" alt="image" src="https://github.com/user-attachments/assets/f38344c4-9cdf-4030-b366-a40201ae1434" />

<br>
<br>

```Python

# The chi- square test to validate hypothesis 5
table = pd.crosstab(df['DissatisfactionRisk'], df['Attrition'])
chi2, p, dof, expected = chi2_contingency(table)
p

```
Output : np.float64(3.419729024669389e-05)

<p>We got:

p = 3.42 × 10⁻⁵ (≈ 0.000034)

This is:

≪ 0.05 → statistically significant

≪ 0.01 → very strongly significant

So:

We reject the null hypothesis with high confidence.

Final decision on Hypothesis 5 Hypothesis 5

Employees with low job satisfaction, low environment satisfaction, low job involvement, and low salary hike are at high risk of attrition

Result

STRONGLY SUPPORTED

The probability that this pattern occurred by chance is less than 0.004%.

“The combined dissatisfaction indicators—job satisfaction, environment satisfaction, job involvement, and salary hike—show a statistically significant association with attrition (χ² test, p < 0.001). Employees with consistently low engagement and compensation growth exhibit substantially higher attrition risk.”</p>

<h4>Hypothesis 5 is True</h4>
<br>

<h1>Recommendations</h1>
<p>

  Following are recommendations for IBM:

1] To retain Single emplyees of young age, organization should present personalized growth plan to such employees and instill sense of purpose.

2] Sales Representatives' travel frequency shall be reduced because this tends to develope fatigue and medical issues in them, moreover they are more likely find better offers outside as traveling enables their network go large.

3] Organization should pitch better offers in terms of hikes and promotions to high performing employees who have changed work too many times in past.

4] Organization should bring a positive change in environment of employees who have reviewed low job satisfaction, low job involvement, low environment satisfaction and relatively low monthly income, by Adopting following methods

a) Team Change.

b) Offering Flexibility of remote work.

c) Team Bonding Activities

d) Delivery - Reward Frameworks .
</p>
