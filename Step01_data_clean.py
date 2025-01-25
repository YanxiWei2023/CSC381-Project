import pandas as pd

# Load the CSV file
data = pd.read_csv('BRFSS_data.csv')

# Drop rows where 'Data_Value' column has missing values
data_cleaned = data.dropna(subset=['Data_Value'])

# Save the cleaned data to a new CSV file
data_cleaned.to_csv('BRFSS_data_cleaned.csv', index=False)

# Display the first few rows of the cleaned data
print(data_cleaned.head())
print(f"Number of rows after cleaning: {data_cleaned.shape[0]}, Number of columns: {data_cleaned.shape[1]}")
