import pandas as pd
from scipy.stats import shapiro, levene, ttest_ind
import matplotlib.pyplot as plt

# Step 1: Load the cleaned CSV file
data = pd.read_csv('BRFSS_data_cleaned.csv')

# Step 2: Filter for "Obesity / Weight Status" in the Class column and "Percent of adults aged 18 years and older who have obesity" in the Question column
data_filtered = data[(data['Class'] == 'Obesity / Weight Status') &
                     (data['Question'] == 'Percent of adults aged 18 years and older who have obesity')]

# Step 3: Filter for "Gender" in StratificationCategory1 and keep only rows with "Female" and "Male" in Stratification1
data_gender = data_filtered[(data_filtered['StratificationCategory1'] == 'Gender') &
                            (data_filtered['Stratification1'].isin(['Female', 'Male']))]

# Step 4: Pivot table to get separate columns for Male and Female obesity rates by year
data_pivot = data_gender.pivot_table(index='YearStart', columns='Stratification1', values='Data_Value')

# Drop any rows with missing values (years without both Male and Female data)
data_pivot = data_pivot.dropna()

# Print the yearly obesity rate comparison table
print("Yearly Male and Female Obesity Rates:")
print(data_pivot)

# Save the yearly obesity rate comparison to a CSV file
data_pivot.to_csv('Yearly_Male_Female_Obesity_Rates.csv')

# Step 5: Perform normality tests for Male and Female data
male_normality = shapiro(data_pivot['Male'])
female_normality = shapiro(data_pivot['Female'])
print(f"Normality test for Male: {male_normality}")
print(f"Normality test for Female: {female_normality}")

# Step 6: Perform homogeneity of variances test (Levene's test)
levene_test = levene(data_pivot['Male'], data_pivot['Female'])
print(f"Levene’s test for equality of variances: {levene_test}")

# Step 7: Perform independent t-test to compare Male and Female obesity rates
t_stat, p_value = ttest_ind(data_pivot['Male'], data_pivot['Female'], equal_var=levene_test.pvalue >= 0.05)
print(f"T-test result - T-statistic: {t_stat}, P-value: {p_value}")

# Step 8: Visualization of obesity rates over time for Male and Female
plt.figure(figsize=(10, 6))
plt.plot(data_pivot.index, data_pivot['Male'], marker='o', label='Male', linestyle='-', color='blue')
plt.plot(data_pivot.index, data_pivot['Female'], marker='o', label='Female', linestyle='-', color='red')

# Add titles and labels
plt.title('Obesity Rates Over Time by Gender (2011-2023)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Obesity Rate (%)', fontsize=12)
plt.legend(title='Gender')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the visualization as an image
plt.savefig('Obesity_Rates_by_Gender_Over_Time.png')

# Display the plot
plt.show()
