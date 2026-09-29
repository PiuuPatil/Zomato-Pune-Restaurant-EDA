import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# ZOMATO PUNE RESTAURANT - EXPLORATORY DATA ANALYSIS
# ============================================================

# 1. LOAD DATASET
df = pd.read_csv("zomato_pune_V002.csv")

print("=" * 60)
print("ZOMATO PUNE RESTAURANT EDA")
print("=" * 60)


# 2. BASIC DATASET INFORMATION
print("\n1. FIRST 5 ROWS")
print(df.head())

print("\n2. DATASET SHAPE")
print("Number of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])

print("\n3. COLUMN NAMES")
print(df.columns.tolist())

print("\n4. DATA TYPES")
print(df.dtypes)

print("\n5. DATASET INFORMATION")
df.info()

print("\n6. STATISTICAL SUMMARY")
print(df.describe())


# 3. MISSING VALUE ANALYSIS
print("\n7. MISSING VALUES")

missing_values = df.isnull().sum()

if missing_values.sum() == 0:
    print("No missing values found.")
else:
    print(missing_values[missing_values > 0])


# 4. DUPLICATE ANALYSIS
print("\n8. DUPLICATE ANALYSIS")

duplicate_count = df.duplicated().sum()

print("Duplicate rows before cleaning:", duplicate_count)

if duplicate_count > 0:
    print("\nSample Duplicate Records:")
    print(df[df.duplicated()].head())

print("\nRows before removing duplicates:", df.shape[0])

df = df.drop_duplicates()

print("Rows after removing duplicates:", df.shape[0])
print("Duplicate rows remaining:", df.duplicated().sum())


# 5. RATING DATA CLEANING
print("\n9. RATING ANALYSIS")

df["Ratings_out_of_5"] = pd.to_numeric(
    df["Ratings_out_of_5"],
    errors="coerce"
)

average_rating = df["Ratings_out_of_5"].mean()
highest_rating = df["Ratings_out_of_5"].max()
lowest_rating = df["Ratings_out_of_5"].min()

print("Average Rating:", round(average_rating, 2))
print("Highest Rating:", highest_rating)
print("Lowest Rating:", lowest_rating)

print("\nRating Summary:")
print(df["Ratings_out_of_5"].describe())


# 6. RATING DISTRIBUTION
print("\n10. CREATING RATING DISTRIBUTION GRAPH")

plt.figure(figsize=(8, 5))

plt.hist(
    df["Ratings_out_of_5"].dropna(),
    bins=10
)

plt.xlabel("Restaurant Rating")
plt.ylabel("Number of Restaurants")
plt.title("Distribution of Restaurant Ratings")

plt.tight_layout()
plt.show()


# 7. TOP-RATED RESTAURANTS
print("\n11. TOP 10 RATED RESTAURANTS")

top_rated = df.sort_values(
    by="Ratings_out_of_5",
    ascending=False
)

print(
    top_rated[
        [
            "Restaurant_Name",
            "Ratings_out_of_5",
            "Locality"
        ]
    ].head(10)
)


# 8. LOCALITY-WISE RESTAURANT ANALYSIS
print("\n12. TOP 10 LOCALITIES BY NUMBER OF RESTAURANTS")

locality_count = df["Locality"].value_counts()

print(locality_count.head(10))


# 9. LOCALITY-WISE GRAPH
print("\n13. CREATING LOCALITY-WISE GRAPH")

plt.figure(figsize=(10, 6))

locality_count.head(10).plot(kind="bar")

plt.xlabel("Locality")
plt.ylabel("Number of Restaurants")
plt.title("Top 10 Localities by Number of Restaurants")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 10. CUISINE ANALYSIS
print("\n14. CUISINE ANALYSIS")

cuisine_count = df["Cuisines"].value_counts()

print("Top 10 Most Common Cuisine Entries:")
print(cuisine_count.head(10))


# 11. COST ANALYSIS
print("\n15. COST ANALYSIS")

df["Charges_for_two"] = pd.to_numeric(
    df["Charges_for_two"],
    errors="coerce"
)

print("Average Cost for Two:",
      round(df["Charges_for_two"].mean(), 2))

print("Maximum Cost for Two:",
      df["Charges_for_two"].max())

print("Minimum Cost for Two:",
      df["Charges_for_two"].min())


# 12. COST DISTRIBUTION GRAPH
print("\n16. CREATING COST DISTRIBUTION GRAPH")

plt.figure(figsize=(8, 5))

plt.hist(
    df["Charges_for_two"].dropna(),
    bins=10
)

plt.xlabel("Charges for Two")
plt.ylabel("Number of Restaurants")
plt.title("Distribution of Restaurant Charges for Two")

plt.tight_layout()
plt.show()


# 13. FINAL DATASET SUMMARY
print("\n" + "=" * 60)
print("FINAL DATASET SUMMARY")
print("=" * 60)

print("Original Rows: 12189")
print("Rows After Duplicate Removal:", df.shape[0])
print("Columns:", df.shape[1])

print("Average Restaurant Rating:",
      round(df["Ratings_out_of_5"].mean(), 2))

print("Highest Restaurant Rating:",
      df["Ratings_out_of_5"].max())

print("Lowest Restaurant Rating:",
      df["Ratings_out_of_5"].min())

print("\nTop Locality:")
print(locality_count.idxmax())

print("Number of Restaurants in Top Locality:",
      locality_count.max())

print("\nEDA COMPLETED SUCCESSFULLY!")