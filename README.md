# ASSIGNMENT-WEEK-6
TASK 1
LOADING AND CLEANING DATA SET
import pandas as pd

# Load the dataset
try:
    # For this example, we'll use the Iris dataset from sklearn for demonstration purposes
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

    # Display the first few rows of the dataset
    print("First few rows of the dataset:")
    print(df.head())

except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")

# Check the structure of the dataset (data types, missing values)
print("\nDataset structure:")
print(df.info())

# Checking for missing values
print("\nMissing values in the dataset:")
print(df.isnull().sum())

# Cleaning the dataset: Dropping rows with missing values (if any)
df_cleaned = df.dropna()

# Alternatively, you can fill missing values if needed:
# df_filled = df.fillna(df.mean()) # Use mean for numerical columns

print("\nCleaned Dataset:")
print(df_cleaned.head())


TASK 2
BASIC DATA ANALYSIS
# Compute basic statistics for numerical columns
print("\nBasic statistics of numerical columns:")
print(df_cleaned.describe())

# Perform groupings by 'species' and compute the mean of a numerical column (e.g., 'sepal length')
grouped = df_cleaned.groupby('species').mean()

print("\nAverage of numerical columns for each species:")
print(grouped)


TASK 3
DATA VISUALIZATION
import matplotlib.pyplot as plt
import seaborn as sns

# Line chart: Let's assume we have a 'date' column for sales data (not present in the iris dataset, just a placeholder)
# We'll plot a line chart of sepal length over the indices (acting as time for demonstration)
plt.figure(figsize=(10, 6))
plt.plot(df_cleaned.index, df_cleaned['sepal length (cm)'], label='Sepal Length')
plt.title('Line Chart: Sepal Length Over Time')
plt.xlabel('Index (Time)')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.show()

# Bar chart: Average petal length per species
plt.figure(figsize=(10, 6))
sns.barplot(x='species', y='petal length (cm)', data=df_cleaned)
plt.title('Bar Chart: Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.show()

# Histogram: Distribution of sepal length
plt.figure(figsize=(10, 6))
plt.hist(df_cleaned['sepal length (cm)'], bins=15, edgecolor='black')
plt.title('Histogram: Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.show()

# Scatter plot: Sepal length vs Petal length
plt.figure(figsize=(10, 6))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', data=df_cleaned, hue='species')
plt.title('Scatter Plot: Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()
