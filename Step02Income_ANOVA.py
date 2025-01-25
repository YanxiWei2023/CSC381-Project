import pandas as pd
from scipy.stats import shapiro, levene, f_oneway
import matplotlib.pyplot as plt

# Step 1: Load the cleaned CSV file
data = pd.read_csv('BRFSS_data_cleaned.csv')

# Step 2: Filter for "Obesity / Weight Status" in the Class column and "Percent of adults aged 18 years and older who have obesity" in the Question column
data_filtered = data[(data['Class'] == 'Obesity / Weight Status') &
                     (data['Question'] == 'Percent of adults aged 18 years and older who have obesity')]

# Step 3: Filter for "Income" in StratificationCategory1, and exclude "Data not reported" in Stratification1
data_income = data_filtered[(data_filtered['StratificationCategory1'] == 'Income') &
                            (data_filtered['Stratification1'] != 'Data not reported')]

# Step 4: Pivot table to get obesity rates by year and income group
data_pivot_income = data_income.pivot_table(index='YearStart', columns='Stratification1', values='Data_Value')

# Drop any rows with missing values (years without data for all income groups)
data_pivot_income = data_pivot_income.dropna()

# Reorder columns to have income groups from low to high
income_order = ['Less than $15,000', '$15,000 - $24,999', '$25,000 - $34,999',
                '$35,000 - $49,999', '$50,000 - $74,999', '$75,000 or greater']
data_pivot_income = data_pivot_income[income_order]

# Print the yearly obesity rate comparison table by income group
print("Yearly Obesity Rates by Income Group (Ordered by Income Level):")
print(data_pivot_income)

# Save the yearly obesity rate comparison to a CSV file
data_pivot_income.to_csv('Yearly_Obesity_Rates_by_Income_Group_Ordered.csv')

# Step 5: Perform normality tests for each income group
normality_results = {}
for income_group in data_pivot_income.columns:
    normality_test = shapiro(data_pivot_income[income_group])
    normality_results[income_group] = normality_test
    print(f"Normality test for {income_group}: {normality_test}")

# Step 6: Perform homogeneity of variances test (Levene's test) across income groups
levene_test = levene(*[data_pivot_income[income_group] for income_group in data_pivot_income.columns])
print(f"Levene’s test for equality of variances across income groups: {levene_test}")

# Step 7: Perform ANOVA to compare obesity rates across different income groups
anova_test = f_oneway(*[data_pivot_income[income_group] for income_group in data_pivot_income.columns])
print(f"ANOVA test result - F-statistic: {anova_test.statistic}, P-value: {anova_test.pvalue}")

# Step 8: Visualization of obesity rates over time by income group
plt.figure(figsize=(12, 8))

# Plot a line for each income group
for income_group in data_pivot_income.columns:
    plt.plot(data_pivot_income.index, data_pivot_income[income_group], marker='o', label=income_group)

# Add titles and labels
plt.title('Obesity Rates Over Time by Income Group (2011-2023)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Obesity Rate (%)', fontsize=12)
plt.legend(title='Income Group', loc='upper left', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the visualization as an image
plt.savefig('Obesity_Rates_by_Income_Group_Over_Time.png')

# Display the plot
plt.show()
