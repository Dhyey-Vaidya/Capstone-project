import pandas as pd

df = pd.read_csv("Indian_crop_production_VersionB_2017_2019_snapshot.csv")

# Remove rows measured in Nuts
df = df[df["unit"] != "Nuts"].copy()

# Save cleaned dataset
df.to_csv("Indian_crop_production_VersionB_no_nuts.csv", index=False)

print("Original rows:", 17710)
print("Rows after removing Nuts:", len(df))
print(df["unit"].value_counts())