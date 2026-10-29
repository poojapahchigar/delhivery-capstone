
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Load the modelling table created in Part A
modelling_table = pd.read_csv("data/modelling_table.csv")

print("Modelling table shape:", modelling_table.shape)
print(modelling_table.head())


# Select the input features
FEATURES = [
    "promised_hours",
    "distance_km",
    "parcel_weight_kg",
    "log_declared_value",
    "n_stops",
    "rain_flag",
    "sale_event_flag",
    "is_cod",
    "is_weekend",
    "dispatch_hour",
    "hub_tier",
    "hub_daily_capacity",
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no"
]

# Select the input data and target
X = modelling_table[FEATURES]
y = modelling_table["delivery_hours"]

# Split data into training and temporary sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# Split the temporary set into validation and test sets
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))


# Scale the features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# Check the scaled data
print("Training scaled shape:", X_train_scaled.shape)
print("Validation scaled shape:", X_val_scaled.shape)
print("Test scaled shape:", X_test_scaled.shape)

train_scaled_max_abs_mean = abs(X_train_scaled.mean(axis=0)).max()
test_scaled_max_abs_mean = abs(X_test_scaled.mean(axis=0)).max()

print("Maximum absolute training mean:", train_scaled_max_abs_mean)
print("Maximum absolute test mean:", test_scaled_max_abs_mean)

# Save the Part B checks
part_b_checks = pd.DataFrame([{
    "train_rows": len(X_train),
    "val_rows": len(X_val),
    "test_rows": len(X_test),
    "train_scaled_max_abs_mean": train_scaled_max_abs_mean,
    "test_scaled_max_abs_mean": test_scaled_max_abs_mean
}])

part_b_checks.to_csv("outputs/part_b_checks.csv", index=False)
print("Part B checks saved.")

print("Maximum absolute test mean:",
      abs(X_test_scaled.mean(axis=0)).max())


# Create a simple baseline model
baseline_prediction = [y_train.mean()] * len(y_val)

# Calculate validation errors
baseline_mae = mean_absolute_error(y_val, baseline_prediction)
baseline_mse = mean_squared_error(y_val, baseline_prediction)
baseline_rmse = baseline_mse ** 0.5
baseline_r2 = r2_score(y_val, baseline_prediction)

print("\nBaseline Model Results")
print("MAE:", round(baseline_mae, 3))
print("MSE:", round(baseline_mse, 3))
print("RMSE:", round(baseline_rmse, 3))
print("R2:", round(baseline_r2, 3))


# Check different decision tree depths
depths = [1, 2, 3, 4, 6, 8, 12, None]
tree_results = []

for depth in depths:
    tree = DecisionTreeRegressor(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)

    train_prediction = tree.predict(X_train)
    val_prediction = tree.predict(X_val)

    train_rmse = mean_squared_error(
        y_train, train_prediction
    ) ** 0.5

    val_rmse = mean_squared_error(
        y_val, val_prediction
    ) ** 0.5

    if depth is None:
        depth_name = "unlimited"
    else:
        depth_name = depth

    tree_results.append({
        "max_depth": depth_name,
        "train_rmse": round(train_rmse, 3),
        "val_rmse": round(val_rmse, 3)
    })

tree_depth_table = pd.DataFrame(tree_results)
tree_depth_table.to_csv(
    "outputs/part_b_tree_depth.csv", index=False
)

print("\nDecision Tree Depth Results")
print(tree_depth_table)

# Select the depth with the lowest validation RMSE
best_depth_row = tree_depth_table.loc[
    tree_depth_table["val_rmse"].idxmin()
]

best_depth = best_depth_row["max_depth"]

if best_depth == "unlimited":
    best_depth = None
else:
    best_depth = int(best_depth)

print("\nBest tree depth:", best_depth)


# Create the regression models
models = {
    "linear_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "decision_tree": DecisionTreeRegressor(
        max_depth=best_depth, random_state=42
    ),
    "random_forest": RandomForestRegressor(
        n_estimators=100, random_state=42
    ),
    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
}

# Train and evaluate each model on validation data
model_results = []

for model_name, model in models.items():
    model.fit(X_train, y_train)
    val_prediction = model.predict(X_val)

    mae = mean_absolute_error(y_val, val_prediction)
    mse = mean_squared_error(y_val, val_prediction)
    rmse = mse ** 0.5
    r2 = r2_score(y_val, val_prediction)

    model_results.append({
        "model": model_name,
        "split": "val",
        "MAE": round(mae, 3),
        "MSE": round(mse, 3),
        "RMSE": round(rmse, 3),
        "R2": round(r2, 3)
    })

model_metrics = pd.DataFrame(model_results)

print("\nRegression Model Results")
print(model_metrics)

model_metrics.to_csv(
    "outputs/part_b_metrics.csv", index=False
)


# Check different learning rates for Gradient Boosting
learning_rates = [0.01, 0.1, 1.0]
gb_results = []

for rate in learning_rates:
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=rate,
        random_state=42
    )

    model.fit(X_train, y_train)
    val_prediction = model.predict(X_val)

    rmse = mean_squared_error(
        y_val, val_prediction
    ) ** 0.5

    r2 = r2_score(y_val, val_prediction)

    gb_results.append({
        "learning_rate": rate,
        "val_rmse": round(rmse, 3),
        "val_r2": round(r2, 3)
    })

gb_table = pd.DataFrame(gb_results)

gb_table.to_csv(
    "outputs/part_b_gb_learning_rate.csv",
    index=False
)

print("\nGradient Boosting Learning Rate Results")
print(gb_table)


# Combine training and validation data for cross-validation
X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

# Set up five-fold cross-validation
from sklearn.model_selection import KFold, cross_val_score

cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Compare the two best models
cv_models = {
    "linear_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
}

cv_results = []

for model_name, model in cv_models.items():
    scores = cross_val_score(
        model,
        X_train_val,
        y_train_val,
        cv=cv,
        scoring="r2"
    )

    cv_results.append({
        "model": model_name,
        "mean_cv_r2": round(scores.mean(), 3),
        "std_cv_r2": round(scores.std(), 3)
    })

cv_table = pd.DataFrame(cv_results)
cv_table.to_csv(
    "outputs/part_b_cv_results.csv", index=False
)

print("\nCross-Validation Results")
print(cv_table)


# Train the final model using training and validation data
final_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

final_model.fit(X_train_val, y_train_val)

# Evaluate the model on the test data
test_prediction = final_model.predict(X_test)

test_mae = mean_absolute_error(y_test, test_prediction)
test_mse = mean_squared_error(y_test, test_prediction)
test_rmse = test_mse ** 0.5
test_r2 = r2_score(y_test, test_prediction)

print("\nFinal Model Test Results")
print("MAE:", round(test_mae, 3))
print("MSE:", round(test_mse, 3))
print("RMSE:", round(test_rmse, 3))
print("R2:", round(test_r2, 3))

# Save the final test results
test_result = pd.DataFrame([{
    "model": "gradient_boosting",
    "split": "test",
    "MAE": round(test_mae, 3),
    "MSE": round(test_mse, 3),
    "RMSE": round(test_rmse, 3),
    "R2": round(test_r2, 3)
}])

model_metrics = pd.concat(
    [model_metrics, test_result], ignore_index=True
)

model_metrics.to_csv(
    "outputs/part_b_metrics.csv", index=False
)
