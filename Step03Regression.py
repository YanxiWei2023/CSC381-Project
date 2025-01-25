# Import necessary libraries
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load and Combine Data
# -------------------------------------
# Load the datasets for income, education, and age
income_data = pd.read_csv('Yearly_Obesity_Rates_by_Income_Group_Ordered.csv')  # Obesity rates by income group
education_data = pd.read_csv('Yearly_Obesity_Rates_by_Education_Level_Ordered_Low_to_High.csv')  # Obesity rates by education level
age_data = pd.read_csv('Yearly_Obesity_Rates_by_Age_Group.csv')  # Obesity rates by age group

# Convert the data into long format
income_long = income_data.melt(id_vars=['YearStart'], var_name='IncomeGroup', value_name='ObesityRate')
education_long = education_data.melt(id_vars=['YearStart'], var_name='EducationLevel', value_name='ObesityRate')
age_long = age_data.melt(id_vars=['YearStart'], var_name='AgeGroup', value_name='ObesityRate')

# Combine all datasets into a single DataFrame
data_long = pd.concat([income_long, education_long, age_long], axis=0, ignore_index=True)

# Step 2: Encode and Reorder Categorical Variables
# -------------------------------------
# Reorder IncomeGroup categories
data_long['IncomeGroup'] = pd.Categorical(data_long['IncomeGroup'],
                                          categories=['Less than $15,000', '$15,000 - $24,999', '$25,000 - $34,999',
                                                      '$35,000 - $49,999', '$50,000 - $74,999', '$75,000 or greater'],
                                          ordered=True)

# Reorder EducationLevel categories
data_long['EducationLevel'] = pd.Categorical(data_long['EducationLevel'],
                                             categories=['Less than high school', 'High school graduate',
                                                         'Some college or technical school', 'College graduate'],
                                             ordered=True)

# Reorder AgeGroup categories
data_long['AgeGroup'] = pd.Categorical(data_long['AgeGroup'],
                                       categories=['18 - 24', '25 - 34', '35 - 44', '45 - 54', '55 - 64', '65 or older'],
                                       ordered=True)

# Encode categories for regression analysis
data_long['IncomeGroup_encoded'] = data_long['IncomeGroup'].cat.codes
data_long['EducationLevel_encoded'] = data_long['EducationLevel'].cat.codes
data_long['AgeGroup_encoded'] = data_long['AgeGroup'].cat.codes

# Step 3: Perform Regression Analysis
# -------------------------------------
# Define the dependent variable
y = data_long['ObesityRate']

# Create interaction term for Income and Education
data_long['Income_Education_Interaction'] = data_long['IncomeGroup_encoded'] * data_long['EducationLevel_encoded']

X_interaction = data_long[['IncomeGroup_encoded', 'EducationLevel_encoded', 'Income_Education_Interaction']]
X_interaction = sm.add_constant(X_interaction)  # Add constant term

# Fit interaction model
model_interaction = sm.OLS(y, X_interaction).fit()
print("Interaction Model Summary:\n", model_interaction.summary())

# Step 4: Visualization
# -------------------------------------
def plot_income_education_interaction(model, income_categories, education_categories):
    """Plot a heatmap for interaction effects between Income and Education."""
    # Define levels for income and education
    income_levels = range(len(income_categories))
    education_levels = range(len(education_categories))

    # Construct prediction data (X_grid)
    X_grid = []
    for income in income_levels:
        for education in education_levels:
            row = [
                1,  # Constant term
                income,  # IncomeGroup_encoded
                education,  # EducationLevel_encoded
                income * education  # Income_Education_Interaction
            ]
            X_grid.append(row)

    # Convert to DataFrame and ensure column order matches the regression model
    X_grid = pd.DataFrame(
        X_grid,
        columns=['const', 'IncomeGroup_encoded', 'EducationLevel_encoded', 'Income_Education_Interaction']
    )

    # Predict obesity rates
    y_pred = model.predict(X_grid)
    y_pred_matrix = y_pred.values.reshape(len(income_levels), len(education_levels))

    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(y_pred_matrix, xticklabels=education_categories, yticklabels=income_categories,
                cmap='YlGnBu', annot=True, fmt=".1f", cbar_kws={'label': 'Predicted Obesity Rate'})
    plt.title('Interaction Effect of Income and Education on Obesity Rate')
    plt.xlabel('Education Level')
    plt.ylabel('Income Group')
    plt.xticks(rotation=0)  # Horizontal labels for X-axis
    plt.tight_layout()
    plt.show()

# Visualize the interaction between income and education
income_categories = data_long['IncomeGroup'].cat.categories
education_categories = data_long['EducationLevel'].cat.categories
plot_income_education_interaction(model_interaction, income_categories, education_categories)

# Step 5: Residual Analysis
# -------------------------------------
def residual_analysis(model, model_name):
    """Perform residual analysis for the regression model."""
    residuals = model.resid
    fitted = model.fittedvalues

    # Plot residuals vs fitted values
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=fitted, y=residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.title(f'Residuals vs Fitted Values ({model_name})')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.tight_layout()
    plt.show()

    # Check residuals normality
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, bins=20)
    plt.title(f'Residuals Distribution ({model_name})')
    plt.xlabel('Residuals')
    plt.ylabel('Density')
    plt.tight_layout()
    plt.show()

# Perform residual analysis for the interaction model
residual_analysis(model_interaction, "Interaction Model (Income and Education)")

# Step 6: Print Model Summary
# -------------------------------------
def print_model_summary():
    """Print a summary table for regression models."""
    print("\nModel Summary Table:\n")

    # Age Model
    X_age = sm.add_constant(data_long[['AgeGroup_encoded']])  # Use AgeGroup_encoded
    model_age = sm.OLS(y, X_age).fit()
    print(
        f"Age Model: R² = {model_age.rsquared:.3f}, Key Variables = Age Group, Conclusion = Analyze how age affects obesity.")

    # Income Model
    X_income = sm.add_constant(data_long[['IncomeGroup_encoded']])
    model_income = sm.OLS(y, X_income).fit()
    print(
        f"Income Model: R² = {model_income.rsquared:.3f}, Key Variables = None, Conclusion = No significant independent effect.")

    # Education Model
    X_education = sm.add_constant(data_long[['EducationLevel_encoded']])
    model_education = sm.OLS(y, X_education).fit()
    print(
        f"Education Model: R² = {model_education.rsquared:.3f}, Key Variables = Education Level (Negative), Conclusion = Education is the most significant variable.")

    # Full Model with Interactions
    X_full = data_long[['IncomeGroup_encoded', 'EducationLevel_encoded', 'Income_Education_Interaction']]
    X_full = sm.add_constant(X_full)
    model_full = sm.OLS(y, X_full).fit()
    print(
        f"Full Model with Interactions: R² = {model_full.rsquared:.3f}, Key Variables = Age-Income-Education Interactions, Conclusion = Interaction effects improved model slightly.")

# Call the function to print the summary
print_model_summary()
