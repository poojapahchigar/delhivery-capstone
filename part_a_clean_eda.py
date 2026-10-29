
import pandas as pd
import numpy as np
from src.shipment_prep import clean_shipments, build_features


# Load the datasets
raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")


# Check missing values before cleaning
raw_unique = raw.drop_duplicates().copy()

missing_distance_before = raw_unique["distance_km"].isnull().sum()

missing_weight_before = raw_unique["parcel_weight_kg"].isnull().sum()

sentinel_weight_count = (
    raw_unique["parcel_weight_kg"] > 100
).sum()

missing_payment_before = raw_unique["payment_mode"].isnull().sum()

# Convert declared value into numbers to check missing values
declared_value_before = pd.to_numeric(
    raw_unique["declared_value"]
    .str.replace("₹", "", regex=False)
    .str.replace(",", "", regex=False),
    errors="coerce"
)

missing_declared_before = declared_value_before.isnull().sum()


# Clean the shipment data
clean, medians = clean_shipments(raw)

# Create engineered features
modelling_table = build_features(clean, hubs)


print("Raw data shape:", raw.shape)
print("Clean data shape:", clean.shape)
print("Modelling table shape:", modelling_table.shape)


# Create the delay target
modelling_table["is_delayed"] = (
    modelling_table["delivery_hours"]
    > modelling_table["promised_hours"]
).astype(int)

print("\nDelay target counts:")
print(modelling_table["is_delayed"].value_counts())


# Save the modelling table
modelling_table.to_csv(
    "data/modelling_table.csv",
    index=False
)

print("\nModelling table saved successfully.")


# Validate the modelling table
print("\nMissing values in modelling table:")
print(modelling_table.isna().sum())

print("\nDuplicate rows in modelling table:",
      modelling_table.duplicated().sum())

print("\nTarget values:")
print(sorted(modelling_table["is_delayed"].unique()))


# Delay rate by service type
delay_by_service = (
    modelling_table.groupby("service_type")
    .agg(
        total_shipments=("is_delayed", "count"),
        delayed_shipments=("is_delayed", "sum"),
        delay_rate=("is_delayed", "mean")
    )
    .reset_index()
)

delay_by_service["delay_rate"] = (
    delay_by_service["delay_rate"] * 100
).round(2)

print("\nDelay rate by service type:")
print(delay_by_service)


# Delay rate by hub tier
delay_by_hub_tier = (
    modelling_table.groupby("hub_tier")
    .agg(
        total_shipments=("is_delayed", "count"),
        delayed_shipments=("is_delayed", "sum"),
        delay_rate=("is_delayed", "mean")
    )
    .reset_index()
)

delay_by_hub_tier["delay_rate"] = (
    delay_by_hub_tier["delay_rate"] * 100
).round(2)

print("\nDelay rate by hub tier:")
print(delay_by_hub_tier)


# Delay rate by payment mode
delay_by_payment = (
    modelling_table.groupby("payment_mode")
    .agg(
        total_shipments=("is_delayed", "count"),
        delayed_shipments=("is_delayed", "sum"),
        delay_rate=("is_delayed", "mean")
    )
    .reset_index()
)

delay_by_payment["delay_rate"] = (
    delay_by_payment["delay_rate"] * 100
).round(2)

print("\nDelay rate by payment mode:")
print(delay_by_payment)


# Correlation with delivery time
numeric_columns = [
    "delivery_hours",
    "promised_hours",
    "distance_km",
    "parcel_weight_kg",
    "n_stops",
    "rain_flag",
    "sale_event_flag",
    "hub_daily_capacity",
    "is_cod",
    "is_weekend",
    "is_delayed"
]

correlations = (
    modelling_table[numeric_columns]
    .corr(numeric_only=True)["delivery_hours"]
    .drop("delivery_hours")
    .sort_values(ascending=False)
    .reset_index()
)

correlations.columns = ["feature", "correlation_with_delivery_hours"]

print("\nCorrelation with delivery hours:")
print(correlations.round(3))


# Save the full numeric correlation matrix
correlation_matrix = (
    modelling_table
    .select_dtypes(include="number")
    .corr()
    .rename_axis("feature")
    .reset_index()
)

correlation_matrix = correlation_matrix.round(4)

correlation_matrix.to_csv(
    "outputs/part_a_correlations.csv",
    index=False
)

print("\nFull correlation matrix saved successfully.")
print("Correlation matrix shape:", correlation_matrix.shape)


# Prepare delay evidence by group
group_results = []

# Analyse service type and hub tier
for dimension in ["service_type", "hub_tier"]:
    grouped = (
        modelling_table.groupby(dimension)
        .agg(
            shipments=("is_delayed", "count"),
            delayed=("is_delayed", "sum"),
            delay_rate=("is_delayed", "mean"),
            mean_hours=("delivery_hours", "mean")
        )
        .reset_index()
    )

    grouped = grouped.rename(columns={dimension: "group"})
    grouped["dimension"] = dimension

    group_results.append(grouped)

print("\nService type and hub tier summaries prepared.")


# Create distance bands
modelling_table["distance_band"] = pd.cut(
    modelling_table["distance_km"],
    bins=[0, 200, 400, 600, 2000],
    labels=["0-200", "201-400", "401-600", "601+"],
    include_lowest=True
)

# Summarize delays by distance band
distance_grouped = (
    modelling_table.groupby("distance_band", observed=False)
    .agg(
        shipments=("is_delayed", "count"),
        delayed=("is_delayed", "sum"),
        delay_rate=("is_delayed", "mean"),
        mean_hours=("delivery_hours", "mean")
    )
    .reset_index()
    .rename(columns={"distance_band": "group"})
)

distance_grouped["dimension"] = "distance_band"
group_results.append(distance_grouped)

print("\nDistance-band summary prepared.")
print(distance_grouped)


# Combine all group summaries into one table
delay_by_group = pd.concat(group_results, ignore_index=True)

# Keep the required columns in order
delay_by_group = delay_by_group[
    ["dimension", "group", "shipments", "delayed", "delay_rate", "mean_hours"]
]

# Round the decimal values to 4 places
delay_by_group["delay_rate"] = delay_by_group["delay_rate"].round(4)
delay_by_group["mean_hours"] = delay_by_group["mean_hours"].round(4)

# Save the results
delay_by_group.to_csv("outputs/part_a_delay_by_group.csv", index=False)

print("\nDelay by group saved successfully.")
print(delay_by_group)


# Check the data cleaning results
checks = []

checks.append(["raw_rows", len(raw)])
checks.append(["duplicate_rows_removed", len(raw) - len(clean)])
checks.append(["rows_after_dedup", len(clean)])
checks.append(["rows_after_hub_merge", len(modelling_table)])
checks.append(["delayed_count", int(modelling_table["is_delayed"].sum())])
checks.append(["delay_rate", round(modelling_table["is_delayed"].mean(), 4)])

# Check missing values in the modelling table
checks.append([
    "nulls_in_features_and_target",
    modelling_table[[
        "promised_hours", "distance_km", "parcel_weight_kg",
        "log_declared_value", "n_stops", "rain_flag",
        "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
        "hub_tier", "hub_daily_capacity", "label_print_batch",
        "scanner_build_no", "tape_roll_no", "sticker_serial_mod",
        "shelf_slot_no", "delivery_hours", "is_delayed"
    ]].isnull().sum().sum()
])

# Convert the checks into a table
checks_df = pd.DataFrame(checks, columns=["metric", "value"])

print("\nBasic data checks:")
print(checks_df)


# Add missing-value checks
checks.append([
    "missing_distance_km_before",
    missing_distance_before
])

checks.append([
    "missing_parcel_weight_before",
    missing_weight_before
])

checks.append([
    "missing_declared_value_before",
    missing_declared_before
])

checks.append([
    "missing_payment_mode_before",
    missing_payment_before
])

checks.append([
    "sentinel_weight_rows",
    sentinel_weight_count
])

# Update and display the checks table
checks_df = pd.DataFrame(checks, columns=["metric", "value"])

print("\nUpdated data checks:")
print(checks_df.tail(5))


# Check service type spellings before and after cleaning
service_types_before = raw_unique["service_type"].nunique()
service_types_after = clean["service_type"].nunique()

checks.append([
    "service_type_spellings_before",
    service_types_before
])

checks.append([
    "service_type_spellings_after",
    service_types_after
])

# Update the checks table
checks_df = pd.DataFrame(checks, columns=["metric", "value"])

print("\nService type checks:")
print(checks_df.tail(2))


# Check the median values used for missing data
checks.append([
    "median_distance_km",
    round(medians["distance_km"], 2)
])

checks.append([
    "median_parcel_weight_kg",
    round(medians["parcel_weight_kg"], 2)
])

checks.append([
    "median_declared_value_inr",
    round(medians["declared_value_inr"], 2)
])

# Update the checks table
checks_df = pd.DataFrame(checks, columns=["metric", "value"])

print("\nMedian checks:")
print(checks_df.tail(3))


# Calculate delivery time quartiles
q1 = modelling_table["delivery_hours"].quantile(0.25)
q3 = modelling_table["delivery_hours"].quantile(0.75)

# Calculate the IQR limits
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# Count delivery time outliers
outlier_count = (
    (modelling_table["delivery_hours"] < lower_bound) |
    (modelling_table["delivery_hours"] > upper_bound)
).sum()

# Add outlier checks
checks.append(["hours_q1", round(q1, 2)])
checks.append(["hours_q3", round(q3, 2)])
checks.append(["hours_lower_bound", round(lower_bound, 2)])
checks.append(["hours_upper_bound", round(upper_bound, 2)])
checks.append(["hours_outlier_count", int(outlier_count)])

# Update the checks table
checks_df = pd.DataFrame(checks, columns=["metric", "value"])

print("\nDelivery time outlier checks:")
print(checks_df.tail(5))


# Check declared value statistics
checks.append([
    "declared_value_mean_inr",
    round(modelling_table["declared_value_inr"].mean(), 2)
])

checks.append([
    "declared_value_median_inr",
    round(modelling_table["declared_value_inr"].median(), 2)
])

checks.append([
    "log_declared_value_mean",
    round(modelling_table["log_declared_value"].mean(), 3)
])

checks.append([
    "log_declared_value_median",
    round(modelling_table["log_declared_value"].median(), 3)
])

# Update the checks table
checks_df = pd.DataFrame(checks, columns=["metric", "value"])

print("\nDeclared value checks:")
print(checks_df.tail(4))

# Check the log transformation
print("\nChecking log declared value:")
print("Mean:", modelling_table["log_declared_value"].mean())
print("Median:", modelling_table["log_declared_value"].median())
#print("Formula check:", (modelling_table["declared_value_inr"] + 1).apply(__import__("math").log).mean())

# Check the median declared value before taking the log
print("\nMedian declared value:")
print(modelling_table["declared_value_inr"].median())

print("Log of median declared value:")
print(np.log(modelling_table["declared_value_inr"].median()))

checks_df = pd.DataFrame(checks, columns=["metric", "value"])
checks_df.to_csv("outputs/part_a_checks.csv", index=False)

print("\nPart A checks saved successfully.")
print(checks_df.tail(6))