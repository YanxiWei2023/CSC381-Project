import pandas as pd
from scipy.stats import shapiro, levene, f_oneway
import matplotlib.pyplot as plt

# Step 1: Load the cleaned CSV file
data = pd.read_csv('BRFSS_data_cleaned.csv')

# Step 2: Filter for "Obesity / Weight Status" in the Class column and "Percent of adults aged 18 years and older who have obesity" in the Question column
data_filtered = data[(data['Class'] == 'Obesity / Weight Status') &
                     (data['Question'] == 'Percent of adults aged 18 years and older who have obesity')]

# Step 3: Filter for "Age (years)" in StratificationCategory1
data_age = data_filtered[data_filtered['StratificationCategory1'] == 'Age (years)']

# Step 4: Pivot table to get obesity rates by year and age group
data_pivot_age = data_age.pivot_table(index='YearStart', columns='Stratification1', values='Data_Value')

# Drop any rows with missing values (years without data for all age groups)
data_pivot_age = data_pivot_age.dropna()

# Print the yearly obesity rate comparison table by age group
print("Yearly Obesity Rates by Age Group:")
print(data_pivot_age)

# Save the yearly obesity rate comparison to a CSV file
data_pivot_age.to_csv('Yearly_Obesity_Rates_by_Age_Group.csv')

# Step 5: Perform normality tests for each age group
normality_results = {}
for age_group in data_pivot_age.columns:
    normality_test = shapiro(data_pivot_age[age_group])
    normality_results[age_group] = normality_test
    print(f"Normality test for {age_group}: {normality_test}")

# Step 6: Perform homogeneity of variances test (Levene's test) across age groups
levene_test = levene(*[data_pivot_age[age_group] for age_group in data_pivot_age.columns])
print(f"Levene’s test for equality of variances across age groups: {levene_test}")

# Step 7: Perform ANOVA to compare obesity rates across different age groups
anova_test = f_oneway(*[data_pivot_age[age_group] for age_group in data_pivot_age.columns])
print(f"ANOVA test result - F-statistic: {anova_test.statistic}, P-value: {anova_test.pvalue}")

# Step 8: Visualization of obesity rates over time by age group
plt.figure(figsize=(12, 8))

# Plot a line for each age group
for age_group in data_pivot_age.columns:
    plt.plot(data_pivot_age.index, data_pivot_age[age_group], marker='o', label=age_group)

# Add titles and labels
plt.title('Obesity Rates Over Time by Age Group (2011-2023)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Obesity Rate (%)', fontsize=12)
plt.legend(title='Age Group', loc='upper left', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the visualization as an image
plt.savefig('Obesity_Rates_by_Age_Group_Over_Time.png')

# Display the plot
plt.show()
