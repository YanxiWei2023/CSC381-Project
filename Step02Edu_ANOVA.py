import pandas as pd
from scipy.stats import shapiro, levene, f_oneway
import matplotlib.pyplot as plt

# Step 1: Load the cleaned CSV file
data = pd.read_csv('BRFSS_data_cleaned.csv')

# Step 2: Filter for "Obesity / Weight Status" in the Class column and "Percent of adults aged 18 years and older who have obesity" in the Question column
data_filtered = data[(data['Class'] == 'Obesity / Weight Status') &
                     (data['Question'] == 'Percent of adults aged 18 years and older who have obesity')]

# Step 3: Filter for "Education" in StratificationCategory1, and exclude "Data not reported" in Stratification1
data_education = data_filtered[(data_filtered['StratificationCategory1'] == 'Education') &
                               (data_filtered['Stratification1'] != 'Data not reported')]

# Step 4: Pivot table to get obesity rates by year and education level
data_pivot_education = data_education.pivot_table(index='YearStart', columns='Stratification1', values='Data_Value')

# Drop any rows with missing values (years without data for all education groups)
data_pivot_education = data_pivot_education.dropna()

# Reorder columns to have education levels from low to high
education_order = ['Less than high school', 'High school graduate', 'Some college or technical school', 'College graduate']
data_pivot_education = data_pivot_education[education_order]

# Print the yearly obesity rate comparison table by education level
print("Yearly Obesity Rates by Education Level (Ordered from Low to High):")
print(data_pivot_education)

# Save the yearly obesity rate comparison to a CSV file
data_pivot_education.to_csv('Yearly_Obesity_Rates_by_Education_Level_Ordered_Low_to_High.csv')

# Step 5: Perform normality tests for each education group
normality_results = {}
for education_level in data_pivot_education.columns:
    normality_test = shapiro(data_pivot_education[education_level])
    normality_results[education_level] = normality_test
    print(f"Normality test for {education_level}: {normality_test}")

# Step 6: Perform homogeneity of variances test (Levene's test) across education groups
levene_test = levene(*[data_pivot_education[education_level] for education_level in data_pivot_education.columns])
print(f"Levene’s test for equality of variances across education groups: {levene_test}")

# Step 7: Perform ANOVA to compare obesity rates across different education levels
anova_test = f_oneway(*[data_pivot_education[education_level] for education_level in data_pivot_education.columns])
print(f"ANOVA test result - F-statistic: {anova_test.statistic}, P-value: {anova_test.pvalue}")

# Step 8: Visualization of obesity rates over time by education level
plt.figure(figsize=(12, 8))

# Plot a line for each education level
for education_level in data_pivot_education.columns:
    plt.plot(data_pivot_education.index, data_pivot_education[education_level], marker='o', label=education_level)

# Add titles and labels
plt.title('Obesity Rates Over Time by Education Level (2011-2023)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Obesity Rate (%)', fontsize=12)
plt.legend(title='Education Level', loc='upper left', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the visualization as an image
plt.savefig('Obesity_Rates_by_Education_Level_Over_Time.png')

# Display the plot
plt.show()
