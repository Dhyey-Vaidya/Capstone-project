import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

output_folder = r"C:\Users\HP\OneDrive\Desktop\Capstone"
os.makedirs(output_folder, exist_ok=True)

# =========================================================
# 1. LOAD CLEAN DATASET
# =========================================================
df = pd.read_csv(os.path.join(output_folder, "Indian_crop_production_VersionB_no_nuts.csv"))
print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# 2. BASIC DATA CHECK
print("\n--- DATA TYPES ---")
print(df.dtypes)
print("\n--- MISSING VALUES ---")
print(df.isnull().sum())
print("\n--- DUPLICATES ---")
print(df.duplicated().sum())

# 3. REMOVE DUPLICATES
df = df.drop_duplicates().copy()

# 4. MAKE SURE NUMERICAL COLUMNS ARE NUMERIC
numeric_columns = ["area_3yr_avg", "production_3yr_avg", "yield_3yr_avg"]
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")

# 5. CHECK NUMERICAL DATA
print("\n--- NUMERICAL SUMMARY ---")
print(df[numeric_columns].describe())

# 6. CHECK FOR ZERO / NEGATIVE VALUES
print("\n--- ZERO VALUES ---")
for column in numeric_columns:
    print(column, ":", (df[column] == 0).sum())

print("\n--- NEGATIVE VALUES ---")
for column in numeric_columns:
    print(column, ":", (df[column] < 0).sum())

# 7. CHECK CROPS
print("\n--- UNIQUE CROPS ---")
print("Number of crops:", df["crop"].nunique())
print(sorted(df["crop"].unique()))

# 8. MAJOR VS FRINGE CROPS
print("\n--- MAJOR VS FRINGE ---")
print(df["is_major_crop"].value_counts())

# 9. CREATE MAJOR-CROP DATASET
major_df = df[df["is_major_crop"] == True].copy()
print("\nMajor crop dataset:", major_df.shape)

# 10. CROP SUMMARY
crop_summary = (
    df.groupby("crop")
    .agg(
        total_production=("production_3yr_avg", "sum"),
        average_production=("production_3yr_avg", "mean"),
        total_area=("area_3yr_avg", "sum"),
        average_area=("area_3yr_avg", "mean"),
        average_yield=("yield_3yr_avg", "mean"),
    )
    .sort_values("total_production", ascending=False)
)
print("\n--- CROP SUMMARY ---")
print(crop_summary.head(20))

# 11. STATE SUMMARY
state_summary = (
    df.groupby("state_name")
    .agg(
        total_production=("production_3yr_avg", "sum"),
        total_area=("area_3yr_avg", "sum"),
        average_yield=("yield_3yr_avg", "mean"),
    )
    .sort_values("total_production", ascending=False)
)
print("\n--- STATE SUMMARY ---")
print(state_summary.head(10))

# 11B. DISTRICT SUMMARY
district_summary = (
    df.groupby(["state_name", "district_name"], dropna=False)
    .agg(
        total_production=("production_3yr_avg", "sum"),
        total_area=("area_3yr_avg", "sum"),
        average_yield=("yield_3yr_avg", "mean"),
        number_of_crops=("crop", "nunique"),
    )
    .sort_values("total_production", ascending=False)
)
print("\n--- TOP 20 DISTRICTS BY PRODUCTION ---")
print(district_summary.head(20))

# 11C. CROP DIVERSIFICATION BY STATE (SHANNON DIVERSITY INDEX)
# Uses each crop's share of positive production within a state.
# Higher Shannon index means production is spread more across crops.
diversification_rows = []
for state, state_group in df.groupby("state_name"):
    crop_production = (
        state_group.groupby("crop")["production_3yr_avg"]
        .sum(min_count=1)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    crop_production = crop_production[crop_production > 0]
    total_state_production = crop_production.sum()

    if len(crop_production) == 0 or total_state_production <= 0:
        continue

    shares = crop_production / total_state_production
    shannon_index = float(-(shares * np.log(shares)).sum())
    number_of_crops = int(len(shares))
    evenness = (
        float(shannon_index / np.log(number_of_crops))
        if number_of_crops > 1
        else 0.0
    )

    diversification_rows.append({
        "state_name": state,
        "number_of_crops_with_positive_production": number_of_crops,
        "shannon_diversity_index": shannon_index,
        "crop_production_evenness": evenness,
        "total_positive_crop_production": float(total_state_production),
    })

diversification_summary = pd.DataFrame(diversification_rows)
if not diversification_summary.empty:
    diversification_summary = diversification_summary.sort_values(
        "shannon_diversity_index", ascending=False
    )
    print("\n--- CROP DIVERSIFICATION BY STATE (SHANNON INDEX) ---")
    print(diversification_summary.to_string(index=False))
else:
    print("Could not calculate crop diversification: no positive production data found.")

# 12. MAJOR-CROP SUMMARY
major_crop_summary = (
    major_df.groupby("crop")
    .agg(
        total_production=("production_3yr_avg", "sum"),
        total_area=("area_3yr_avg", "sum"),
        average_yield=("yield_3yr_avg", "mean"),
    )
    .sort_values("total_production", ascending=False)
)
print("\n--- MAJOR CROP SUMMARY ---")
print(major_crop_summary)

# 13. SAVE PANDAS OUTPUTS
crop_summary.to_csv(os.path.join(output_folder, "crop_summary.csv"))
state_summary.to_csv(os.path.join(output_folder, "state_summary.csv"))
district_summary.to_csv(os.path.join(output_folder, "district_summary.csv"))
diversification_summary.to_csv(
    os.path.join(output_folder, "state_crop_diversification.csv"), index=False
)
major_crop_summary.to_csv(os.path.join(output_folder, "major_crop_summary.csv"))

print("\nPandas processing complete!")
print("Major crops actually present:")
print(sorted(major_df["crop"].unique()))
print("\nNumber of major crops:", major_df["crop"].nunique())
print("\nMajor crops with zero production:")
print(
    major_df[major_df["production_3yr_avg"] == 0][
        [
            "state_name",
            "district_name",
            "crop",
            "production_3yr_avg",
            "yield_3yr_avg",
        ]
    ]
)

# =========================================================
# NUMPY ANALYSIS
# =========================================================
print("\n\n================ NUMPY ANALYSIS ================")

# Convert to numeric arrays and remove missing/infinite values separately
area = pd.to_numeric(df["area_3yr_avg"], errors="coerce").to_numpy(dtype=float)
production = pd.to_numeric(df["production_3yr_avg"], errors="coerce").to_numpy(dtype=float)
yield_data = pd.to_numeric(df["yield_3yr_avg"], errors="coerce").to_numpy(dtype=float)

area_clean = area[np.isfinite(area)]
production_clean = production[np.isfinite(production)]
yield_clean = yield_data[np.isfinite(yield_data)]

def print_numeric_summary(label, values):
    print(f"\n--- {label} STATISTICS ---")
    if values.size == 0:
        print("No valid finite observations available.")
        return
    print("Valid observations:", values.size)
    print("Mean:", np.mean(values))
    print("Median:", np.median(values))
    print("Standard deviation:", np.std(values))
    print("Minimum:", np.min(values))
    print("Maximum:", np.max(values))

print_numeric_summary("AREA", area_clean)
print_numeric_summary("PRODUCTION", production_clean)
print_numeric_summary("YIELD", yield_clean)

print("\n--- YIELD PERCENTILES ---")
if yield_clean.size:
    for percentile in [25, 50, 75, 90, 95, 99]:
        print(f"{percentile}th percentile:", np.percentile(yield_clean, percentile))
else:
    print("No valid yield observations available.")

if yield_clean.size:
    q1 = np.percentile(yield_clean, 25)
    q3 = np.percentile(yield_clean, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = yield_clean[(yield_clean < lower_bound) | (yield_clean > upper_bound)]

    print("\n--- YIELD OUTLIER ANALYSIS ---")
    print("Q1:", q1)
    print("Q3:", q3)
    print("IQR:", iqr)
    print("Lower bound:", lower_bound)
    print("Upper bound:", upper_bound)
    print("Number of potential outliers:", len(outliers))
else:
    print("\nYield outlier analysis skipped: no valid yield observations.")

zero_production = np.isfinite(production) & (production == 0)
print("\n--- ZERO PRODUCTION ---")
print("Number of zero-production rows:", int(np.sum(zero_production)))
print("\nNumPy analysis complete!")

# =========================================================
# DATA VISUALIZATION SECTION
# =========================================================
print("\n\n================ GENERATING GRAPH VISUALIZATIONS ================")
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

# Load historical Version A timeline for trend plot
df_vA = pd.read_csv(os.path.join(output_folder, "Indian_crop_production_VersionA_no_nuts.csv"))
df_vA_no_nuts = df_vA[df_vA["unit"] != "Nuts"].copy()

# ---------------------------------------------------------
# Graph 1: Top 10 Crops by Production (Bar Chart)
# What it proves: Identifies national staple crops driving India's agricultural volume.
# ---------------------------------------------------------
top10_crops = df.groupby("crop")["production_3yr_avg"].sum().nlargest(10).reset_index()
plt.figure(figsize=(10, 5))
sns.barplot(data=top10_crops, x="production_3yr_avg", y="crop", palette="viridis")
plt.title("Top 10 Crops by Total 3-Year Average Production (Tonnes)")
plt.xlabel("Total Production (Tonnes)")
plt.ylabel("Crop")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "graph1_top10_crops.png"), dpi=300)
plt.show()
plt.close()

# ---------------------------------------------------------
# Graph 2: Top 10 States by Production (Bar Chart)
# What it proves: Pinpoints primary agricultural hub states powering national output.
# ---------------------------------------------------------
top10_states = df.groupby("state_name")["production_3yr_avg"].sum().nlargest(10).reset_index()
plt.figure(figsize=(10, 5))
sns.barplot(data=top10_states, x="production_3yr_avg", y="state_name", palette="magma")
plt.title("Top 10 States by Total Agricultural Production (Tonnes)")
plt.xlabel("Total Production (Tonnes)")
plt.ylabel("State")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "graph2_top10_states.png"), dpi=300)
plt.show()
plt.close()

# ---------------------------------------------------------
# Graph 3: Production Trend Over Time (Line Graph)
# What it proves: Evaluates historical production trajectories, growth rate, and macro shocks over 23 years.
# ---------------------------------------------------------
annual_prod = df_vA_no_nuts.groupby("crop_year")["production"].sum().reset_index()
plt.figure(figsize=(10, 5))
plt.plot(
    annual_prod["crop_year"],
    annual_prod["production"],
    marker="o",
    color="#2b5c8f",
    linewidth=2,
)
plt.title("National Agricultural Production Trend (1997–2019)")
plt.xlabel("Crop Year")
plt.ylabel("Total Production (Tonnes)")
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "graph3_production_trend.png"), dpi=300)
plt.show()
plt.close()

# ---------------------------------------------------------
# Graph 4: Average Yield by Crop (Bar Chart)
# What it proves: Compares bio-efficiency across crops to identify high-yield per hectare commodities.
# ---------------------------------------------------------
top_yield_crops = df.groupby("crop")["yield_3yr_avg"].mean().nlargest(10).reset_index()
plt.figure(figsize=(10, 5))
sns.barplot(data=top_yield_crops, x="yield_3yr_avg", y="crop", palette="crest")
plt.title("Average Yield by Top 10 Crop Categories (Tonnes/Hectare)")
plt.xlabel("Average Yield (Tonnes per Hectare)")
plt.ylabel("Crop")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "graph4_avg_yield_by_crop.png"), dpi=300)
plt.show()
plt.close()

# ---------------------------------------------------------
# Graph 5: Area vs Production (Scatter Plot)
# What it proves: Determines if land scale directly correlates with total yield across crops/districts.
# ---------------------------------------------------------
log_plot_data = df[["area_3yr_avg", "production_3yr_avg"]].replace(
    [np.inf, -np.inf], np.nan
).dropna()
log_plot_data = log_plot_data[
    (log_plot_data["area_3yr_avg"] > 0)
    & (log_plot_data["production_3yr_avg"] > 0)
]

plt.figure(figsize=(9, 6))
if not log_plot_data.empty:
    plt.scatter(
        log_plot_data["area_3yr_avg"],
        log_plot_data["production_3yr_avg"],
        alpha=0.5,
        color="#e74c3c",
        edgecolors="none",
        s=25,
    )
else:
    print("Graph 5 note: no positive finite area-production pairs for log-scale plot.")

plt.xscale("log")
plt.yscale("log")
plt.title("Cultivated Area vs. Production Output (Log-Log Scale)")
plt.xlabel("Cultivated Area (Hectares, Log Scale)")
plt.ylabel("Production (Tonnes, Log Scale)")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "graph5_area_vs_production.png"), dpi=300)
plt.show()
plt.close()

# ---------------------------------------------------------
# Graph 6: State x Crop Production Intensity (Heatmap)
# What it proves: Uncovers geographic specialization and crop clustering in specific Indian states.
# ---------------------------------------------------------
top10_state_names = top10_states["state_name"].tolist()
top10_crop_names = top10_crops["crop"].tolist()
heatmap_df = df[
    df["state_name"].isin(top10_state_names) & df["crop"].isin(top10_crop_names)
]
pivot_table = heatmap_df.pivot_table(
    index="state_name",
    columns="crop",
    values="production_3yr_avg",
    aggfunc="sum",
    fill_value=0,
)
plt.figure(figsize=(11, 7))
sns.heatmap(pivot_table, cmap="YlGnBu", annot=False, fmt=".0f", linewidths=0.5)
plt.title("State × Crop Production Intensity Heatmap (Top 10 States vs Top 10 Crops)")
plt.xlabel("Crop")
plt.ylabel("State")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "graph6_state_crop_heatmap.png"), dpi=300)
plt.show()
plt.close()
print("All 6 visualization plots generated and saved successfully!")

# =========================================================
# SCIPY STATISTICAL ANALYSIS
# =========================================================
print("\n\n================ SCIPY STATISTICAL ANALYSIS ================")

# 1. PEARSON AND SPEARMAN CORRELATIONS: AREA VS PRODUCTION
print("\n--- AREA VS PRODUCTION CORRELATIONS ---")

correlation_data = df[["area_3yr_avg", "production_3yr_avg"]].copy()
correlation_data = correlation_data.replace([np.inf, -np.inf], np.nan).dropna()
area_valid = correlation_data["area_3yr_avg"].to_numpy(dtype=float)
production_valid = correlation_data["production_3yr_avg"].to_numpy(dtype=float)

main_result_rows = []

if (
    len(correlation_data) >= 2
    and np.unique(area_valid).size > 1
    and np.unique(production_valid).size > 1
):
    pearson_result = stats.pearsonr(area_valid, production_valid)
    spearman_result = stats.spearmanr(area_valid, production_valid)

    print("Valid paired observations:", len(correlation_data))
    print("Pearson correlation:", round(float(pearson_result.statistic), 4))
    print("Pearson p-value:", float(pearson_result.pvalue))
    if pearson_result.pvalue < 0.05:
        print("Pearson result: statistically significant linear association (p < 0.05).")
    else:
        print("Pearson result: no statistically significant linear association detected (p >= 0.05).")

    print("Spearman correlation:", round(float(spearman_result.statistic), 4))
    print("Spearman p-value:", float(spearman_result.pvalue))

    main_result_rows.extend([
        {
            "test": "Pearson: Area vs Production",
            "statistic": pearson_result.statistic,
            "p_value": pearson_result.pvalue,
            "observations": len(correlation_data),
        },
        {
            "test": "Spearman: Area vs Production",
            "statistic": spearman_result.statistic,
            "p_value": spearman_result.pvalue,
            "observations": len(correlation_data),
        },
    ])
else:
    print(
        "Correlation skipped: need at least two valid paired observations "
        "and variation in both area and production."
    )

# 2. CROP-WISE PEARSON CORRELATION
print("\n--- CROP-WISE AREA VS PRODUCTION ---")
crop_results = []

for crop, group in df.groupby("crop"):
    values = (
        group[["area_3yr_avg", "production_3yr_avg"]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if len(values) < 10:
        continue
    if values["area_3yr_avg"].nunique() < 2 or values["production_3yr_avg"].nunique() < 2:
        continue

    crop_result = stats.pearsonr(
        values["area_3yr_avg"].to_numpy(dtype=float),
        values["production_3yr_avg"].to_numpy(dtype=float),
    )
    crop_results.append({
        "crop": crop,
        "observations": len(values),
        "correlation": crop_result.statistic,
        "p_value": crop_result.pvalue,
    })

crop_results_df = pd.DataFrame(
    crop_results, columns=["crop", "observations", "correlation", "p_value"]
)
if not crop_results_df.empty:
    crop_results_df = crop_results_df.sort_values("correlation", ascending=False)
    print(crop_results_df.to_string(index=False))
else:
    print("No crops met the minimum sample/variation requirements.")

crop_results_df.to_csv(
    os.path.join(output_folder, "scipy_crop_correlations.csv"), index=False
)
print("Saved scipy_crop_correlations.csv")

# 3. YIELD NORMALITY TEST
# The pooled test is descriptive only because different crops have different yield scales.
print("\n--- OVERALL YIELD NORMALITY TEST (POOLED CROPS) ---")
yield_values = (
    pd.to_numeric(df["yield_3yr_avg"], errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
    .to_numpy(dtype=float)
)

if len(yield_values) >= 8 and np.unique(yield_values).size > 1:
    rng = np.random.default_rng(42)
    sample_size = min(5000, len(yield_values))
    sample = rng.choice(yield_values, size=sample_size, replace=False)
    normality_result = stats.normaltest(sample)
    print("Sample size:", sample_size)
    print("Test statistic:", round(float(normality_result.statistic), 4))
    print("P-value:", float(normality_result.pvalue))
    if normality_result.pvalue < 0.05:
        print("Pooled yield values significantly deviate from normality (p < 0.05).")
    else:
        print("Insufficient evidence to reject normality for the pooled sample (p >= 0.05).")
    print("Caution: this combines different crops, so crop-specific tests are more interpretable.")
else:
    print("Overall normality test skipped: need at least 8 valid values with variation.")

# 4. CROP-WISE NORMALITY TESTS (MORE INTERPRETABLE THAN POOLING CROPS)
crop_normality_rows = []
for crop, group in df.groupby("crop"):
    crop_yield = (
        pd.to_numeric(group["yield_3yr_avg"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=float)
    )
    if len(crop_yield) < 8 or np.unique(crop_yield).size < 2:
        continue
    # normaltest is most useful with adequate samples; cap the sample for efficiency.
    if len(crop_yield) > 5000:
        rng = np.random.default_rng(42)
        crop_yield = rng.choice(crop_yield, size=5000, replace=False)
    crop_normality = stats.normaltest(crop_yield)
    crop_normality_rows.append({
        "crop": crop,
        "observations_tested": len(crop_yield),
        "test_statistic": crop_normality.statistic,
        "p_value": crop_normality.pvalue,
        "deviates_from_normality_at_0_05": bool(crop_normality.pvalue < 0.05),
    })

crop_normality_df = pd.DataFrame(
    crop_normality_rows,
    columns=[
        "crop", "observations_tested", "test_statistic", "p_value",
        "deviates_from_normality_at_0_05"
    ],
)
crop_normality_df.to_csv(
    os.path.join(output_folder, "scipy_crop_yield_normality.csv"), index=False
)
print("\nSaved scipy_crop_yield_normality.csv")
if not crop_normality_df.empty:
    print(crop_normality_df.to_string(index=False))
else:
    print("No crops met the sample/variation requirements for normality testing.")

# 5. 95% CONFIDENCE INTERVAL FOR MEAN YIELD
print("\n--- 95% CONFIDENCE INTERVAL FOR MEAN YIELD ---")
n = len(yield_values)
if n > 1:
    mean_yield = float(np.mean(yield_values))
    standard_error = float(stats.sem(yield_values))
    if np.isfinite(standard_error):
        lower, upper = stats.t.interval(
            confidence=0.95,
            df=n - 1,
            loc=mean_yield,
            scale=standard_error,
        )
        print("Observations:", n)
        print("Mean yield:", round(mean_yield, 4))
        print("Standard error:", round(standard_error, 4))
        print("95% confidence interval:", (float(lower), float(upper)))
        print("Interpretation: interval estimates the mean of the pooled row-level yield values.")
        print("Caution: observations from different crops/locations may not be independent.")
    else:
        print("Confidence interval skipped: standard error is not finite.")
else:
    print("Confidence interval skipped: need at least two valid yield observations.")

# 6. SAVE MAIN SCIPY RESULTS
main_results = pd.DataFrame(
    main_result_rows,
    columns=["test", "statistic", "p_value", "observations"],
)
main_results.to_csv(
    os.path.join(output_folder, "scipy_main_results.csv"), index=False
)
print("\nSaved scipy_main_results.csv")
print("\nSciPy statistical analysis complete!")

# =========================================================
# CROP PRODUCTION FORECASTING (VERSION A: HISTORICAL DATA)
# =========================================================
print("\n\n================ CROP PRODUCTION FORECAST ================")

selected_crop = input("Enter the crop name for forecasting (case-insensitive): ")
forecast_data = df_vA_no_nuts[
    df_vA_no_nuts["crop"].astype(str).str.strip().str.casefold()
    == selected_crop.casefold()
].copy()

if forecast_data.empty:
    print(f"Crop '{selected_crop}' was not found in Version A.")
    print("Available crop names include:")
    print(sorted(df_vA_no_nuts["crop"].dropna().astype(str).unique())[:50])
else:
    forecast_data["crop_year"] = pd.to_numeric(forecast_data["crop_year"], errors="coerce")
    forecast_data["production"] = pd.to_numeric(forecast_data["production"], errors="coerce")
    annual_crop = (
        forecast_data.dropna(subset=["crop_year", "production"])
        .groupby("crop_year")["production"]
        .sum()
        .sort_index()
    )
    annual_crop = annual_crop[np.isfinite(annual_crop.to_numpy())]
    years = annual_crop.index.to_numpy(dtype=float)
    values = annual_crop.to_numpy(dtype=float)

    if len(years) < 6:
        print(
            f"Not enough annual observations for {selected_crop} "
            "to evaluate a forecast. At least 6 are needed."
        )
    else:
        split = len(years) - 3
        train_years = years[:split]
        train_values = values[:split]
        test_years = years[split:]
        actual_values = values[split:]

        slope, intercept = np.polyfit(train_years, train_values, 1)
        predicted_values = slope * test_years + intercept
        mae = np.mean(np.abs(actual_values - predicted_values))

        print("Selected crop:", selected_crop)
        print("Training years:", int(train_years.min()), "to", int(train_years.max()))
        print("Test years:", test_years.astype(int).tolist())
        print("Actual production:", np.round(actual_values, 2).tolist())
        print("Predicted production:", np.round(predicted_values, 2).tolist())
        print("Mean Absolute Error (MAE):", round(float(mae), 2))

        slope_all, intercept_all = np.polyfit(years, values, 1)
        next_year = int(years.max()) + 1
        forecast_value = slope_all * next_year + intercept_all

        print(
            f"Estimated {selected_crop} production for {next_year}:",
            round(float(forecast_value), 2),
            "tonnes",
        )
        print(
            "Note: This is a simple linear-trend estimate, not a guarantee "
            "of future production."
        )

        forecast_results = pd.DataFrame({
            "year": list(test_years.astype(int)) + [next_year],
            "crop": [selected_crop] * (len(test_years) + 1),
            "actual_production": list(actual_values) + [np.nan],
            "predicted_production": list(predicted_values) + [forecast_value],
            "record_type": ["test_prediction"] * len(test_years) + ["next_year_forecast"],
        })
        forecast_results.to_csv(
            os.path.join(output_folder, "crop_production_forecast_results.csv"),
            index=False,
        )

        plt.figure(figsize=(10, 5))
        plt.plot(
            years, values, marker="o", linewidth=2,
            label="Actual annual production",
        )
        plt.scatter(
            [next_year], [forecast_value], color="red",
            label=f"{next_year} forecast", zorder=3,
        )
        plt.plot(
            [years[-1], next_year], [values[-1], forecast_value],
            linestyle="--", color="red", alpha=0.8,
        )
        plt.title(f"{selected_crop} Production Forecast")
        plt.xlabel("Year")
        plt.ylabel("Production (Tonnes)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_folder, "graph7_crop_forecast.png"),
            dpi=300,
        )
        plt.show()
        plt.close()
        print("Saved graph7_crop_forecast.png")
        print("Saved crop_production_forecast_results.csv")

print("\nForecasting section complete!")