import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("data/titanic_cleaned.csv")

print("Dataset Loaded Successfully!")

# Display First 5 Rows
print("\nFirst 5 Rows:")
print(df.head())

# Dataset Information
print("\nDataset Information:")
df.info()

# Dataset Shape
print("\nRows and Columns:")
print(df.shape)

# Statistical Summary
print("\nStatistical Summary:")
print(df.describe())

# Missing Values Analysis
print("\nMissing Values:")
print(df.isnull().sum())

# Missing Values Percentage
print("\nMissing Values Percentage:")
print((df.isnull().sum() / len(df)) * 100)

# Duplicate Records
print("\nDuplicate Records:")
print(df.duplicated().sum())

# Survival Count
print("\nSurvival Count:")
print(df["alive"].value_counts())

# Gender Count
print("\nGender Count:")
print(df["sex"].value_counts())

# Passenger Class Count
print("\nPassenger Class Count:")
print(df["class"].value_counts())

# -------------------------------
# DATA VISUALIZATION
# -------------------------------

# Chart 1: Survival Count
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="alive")
plt.title("Titanic Survival Count")
plt.xlabel("Survival Status")
plt.ylabel("Number of Passengers")
plt.savefig("charts/survival_count.png")
plt.show()


# Chart 2: Survival Rate by Gender
gender_survival = df.groupby("sex")["survived"].mean() * 100

plt.figure(figsize=(7, 5))
sns.barplot(x=gender_survival.index, y=gender_survival.values)
plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate (%)")
plt.savefig("charts/survival_by_gender.png")
plt.show()


# Chart 3: Survival Rate by Passenger Class
class_survival = (
    df.groupby("class")["survived"].mean() * 100
).reindex(["First", "Second", "Third"])

plt.figure(figsize=(7, 5))
sns.barplot(x=class_survival.index, y=class_survival.values)
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate (%)")
plt.savefig("charts/survival_by_class.png")


print("\nAll Charts Generated Successfully!")


# -------------------------------
# DATA VISUALIZATION
# -------------------------------

# Chart 1: Survival Count
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="alive")
plt.title("Titanic Survival Count")
plt.xlabel("Survival Status")
plt.ylabel("Number of Passengers")
plt.savefig("charts/survival_count.png")
plt.close()

# Chart 2: Survival Rate by Gender
gender_survival = df.groupby("sex")["survived"].mean() * 100

plt.figure(figsize=(7, 5))
sns.barplot(x=gender_survival.index, y=gender_survival.values)
plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate (%)")
plt.savefig("charts/survival_by_gender.png")
plt.close()

# Chart 3: Survival Rate by Passenger Class
class_survival = (
    df.groupby("class")["survived"].mean() * 100
).reindex(["First", "Second", "Third"])

plt.figure(figsize=(7, 5))
sns.barplot(x=class_survival.index, y=class_survival.values)
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate (%)")
plt.savefig("charts/survival_by_class.png")
plt.close()

print("\nAll Charts Generated Successfully!")

# Chart 4: Age Distribution

plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="age", bins=20, kde=True)
plt.title("Passenger Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.savefig("charts/age_distribution.png")
plt.close()


# Chart 5: Fare Distribution

plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="fare", bins=30, kde=True)
plt.title("Passenger Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")
plt.savefig("charts/fare_distribution.png")
plt.close()

print("All 5 Charts Generated Successfully!")

# SURVIVAL RATE ANALYSIS

print("\nSurvival Rate by Gender (%):")
print(df.groupby("sex")["survived"].mean() * 100)

print("\nSurvival Rate by Passenger Class (%):")
print(df.groupby("class")["survived"].mean() * 100)

# Survival Rate by Age Group
df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 12, 18, 35, 60, 100],
    labels=["Child", "Teenager", "Young Adult", "Adult", "Senior"]
)

print("\nSurvival Rate by Age Group (%):")
print(df.groupby("age_group", observed=False)["survived"].mean() * 100)