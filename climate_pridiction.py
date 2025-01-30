import pandas as pd  
import numpy as np 
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the dataset  
df = pd.read_csv("climate_change_dataset.csv")  

# Display basic info  
"""print(df.info())  
print(df.head())  """

# Check for missing values  
"""print(df.isnull().sum())  """
# Fill numerical missing values with mean
df.fillna(df.select_dtypes(include=['number']).mean(), inplace=True)

# Fill categorical missing values with mode (most frequent value)
df["Country"].fillna(df["Country"].mode()[0], inplace=True)

# Drop rows with too many missing values (optional) 

# Select only numeric columns
numeric_cols = df.select_dtypes(include=[np.number])

# Compute IQR only for numeric columns
Q1 = numeric_cols.quantile(0.25)
Q3 = numeric_cols.quantile(0.75)
IQR = Q3 - Q1

# Apply IQR filter only on numeric data
df_filtered = df[~((numeric_cols < (Q1 - 1.5 * IQR)) | (numeric_cols > (Q3 + 1.5 * IQR))).any(axis=1)]

# Display the new dataframe
# Global Temperature Trends
print(df_filtered.shape)  # Check if rows were removed
plt.figure(figsize=(12, 6))
sns.lineplot(x=df["Year"], y=df["Avg Temperature (°C)"], ci=None)
plt.xlabel("Year")
plt.ylabel("Avg Temperature (°C)")
plt.title("Global Temperature Trends Over Time")
#plt.show()

#CO₂ Emissions by Country
top_countries = df.groupby("Country")["CO2 Emissions (Tons/Capita)"].mean().sort_values(ascending=False).head(10)
top_countries.plot(kind="bar", figsize=(12,6), color="red")
plt.xlabel("Country")
plt.ylabel("Average CO2 Emissions (Tons/Capita)")
plt.title("Top 10 Countries by Average CO2 Emissions")
#plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Climate Factors")
plt.show()
