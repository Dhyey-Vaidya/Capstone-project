import pandas as pd

# 1. Load Version A (all historical years)
df = pd.read_csv("Indian_crop_production_VersionA_1997_2019.csv")

# 2. Record original row count and year range
original_rows = len(df)
original_years = sorted(df["crop_year"].dropna().unique())

print("Original rows:", original_rows)
print("Original year range:", min(original_years), "-", max(original_years))

# 3. Remove rows measured in Nuts
df = df[df["unit"] != "Nuts"].copy()

# 4. Save the cleaned Version A dataset
output_file = "Indian_crop_production_VersionA_no_nuts.csv"
df.to_csv(output_file, index=False)

# 5. Verify the cleaned dataset
print("\n--- CLEANING COMPLETE ---")
print("Original rows:", original_rows)
print("Rows after removing Nuts:", len(df))
print("Years retained:", sorted(df["crop_year"].dropna().unique()))
print("Units remaining:")
print(df["unit"].value_counts())
print("Saved as:", output_file)

print("\nPreview:")
print(df.head())