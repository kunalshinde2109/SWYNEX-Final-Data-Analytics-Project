import pandas as pd
import numpy as np

print("Titanic Data Cleaning Project Started!")

# Download Titanic Dataset

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"

df = pd.read_csv(url)

print("Dataset Downloaded Successfully!")

print(df.head())

import os

os.makedirs("data", exist_ok=True)

df.to_csv("data/titanic_raw.csv", index=False)

print("Raw Dataset Saved Successfully!")

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

# Check Duplicate Records

print("\nDuplicate Records:")

print(df.duplicated().sum())

# Check Unique Values

print("\nSex Values:")
print(df["sex"].unique())

print("\nEmbarked Values:")
print(df["embarked"].unique())

print("\nClass Values:")
print(df["class"].unique())

print("\nAge Data Type:")
print(df["age"].dtype)

# Display Duplicate Records

duplicate_rows = df[df.duplicated(keep=False)]

print("\nTotal Duplicate Rows:")
print(len(duplicate_rows))

print("\nFirst 10 Duplicate Records:")
print(duplicate_rows.head(10))

# Missing Values Percentage

missing_percentage = (df.isnull().sum() / len(df)) * 100

print("\nMissing Values Percentage:")
print(missing_percentage)

# Calculate Median and Mode

age_median = df["age"].median()

embarked_mode = df["embarked"].mode()[0]

town_mode = df["embark_town"].mode()[0]

print("\nAge Median:")
print(age_median)

print("\nEmbarked Mode:")
print(embarked_mode)

print("\nEmbark Town Mode:")
print(town_mode)

# Handle Missing Values

# Fill Age with Median
df["age"] = df["age"].fillna(df["age"].median())

# Fill Embarked with Mode
df["embarked"] = df["embarked"].fillna(
    df["embarked"].mode()[0]
)

# Fill Embark Town with Mode
df["embark_town"] = df["embark_town"].fillna(
    df["embark_town"].mode()[0]
)

print("\nMissing Values After Cleaning:")

print(df.isnull().sum())

# Save Cleaned Dataset
df.to_csv("data/titanic_cleaned.csv", index=False)

print("\nCleaned Dataset Saved Successfully!")

print("\nFinal Dataset Shape:")
print(df.shape)

# Duplicate Records Validation

duplicate_count = df.duplicated().sum()

print("\nDuplicate Records After Cleaning:")
print(duplicate_count)

duplicate_rows = df[df.duplicated(keep=False)]

print("\nTotal Rows Involved in Duplicates:")
print(len(duplicate_rows))

print("\nDuplicate Records Sample:")
print(duplicate_rows.head(10))

# Data Validation

print("\nData Validation Results:")

# Check Age
print("\nInvalid Age Records:")
print(df[(df["age"] < 0) | (df["age"] > 100)].shape[0])

# Check Fare
print("\nNegative Fare Records:")
print(df[df["fare"] < 0].shape[0])

# Check Passenger Class
print("\nInvalid Passenger Class:")
print(df[~df["pclass"].isin([1, 2, 3])].shape[0])

# Check Survival Values
print("\nInvalid Survival Values:")
print(df[~df["survived"].isin([0, 1])].shape[0])

# Check Gender
print("\nInvalid Gender Values:")
print(df[~df["sex"].isin(["male", "female"])].shape[0])