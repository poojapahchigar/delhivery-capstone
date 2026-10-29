
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.shipment_prep import FEATURES, clean_shipments


# Step 1: Read the saved Part C configuration
config = pd.read_csv(
    "outputs/part_c_final_config.csv"
).iloc[0]

k = int(config["selector_k"])
n_estimators = int(config["n_estimators"])
min_samples_leaf = int(config["min_samples_leaf"])
threshold = float(config["threshold"])


# Step 2: Load the same modelling table used in Part C
df = pd.read_csv("data/modelling_table.csv")

X = df[FEATURES]
y = df["is_delayed"].astype(int)


# Step 3: Recreate the exact Part C split
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))


# Step 4: Build the same final model as Part C
model = Pipeline([
    ("selector", SelectKBest(score_func=f_classif, k=k)),
    ("model", RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    ))
])


# Step 5: Train on the training set only
model.fit(X_train, y_train)


# Step 6: Create the models folder if needed
import os
os.makedirs("models", exist_ok=True)


# Step 7: Save the trained model
joblib.dump(model, "models/delay_model.pkl")


# Step 8: Save the threshold
pd.DataFrame({
    "threshold": [threshold]
}).to_csv(
    "models/delay_threshold.csv",
    index=False
)


# Step 9: Save the feature order
pd.DataFrame({
    "feature": FEATURES
}).to_csv(
    "models/feature_order.csv",
    index=False
)


# Step 10: Calculate medians using the shared cleaning function
raw_shipments = pd.read_csv("data/raw_shipments.csv")

cleaned_shipments, medians = clean_shipments(raw_shipments)

pd.DataFrame({
    "column": list(medians.keys()),
    "median": list(medians.values())
}).to_csv(
    "models/impute_values.csv",
    index=False
)


# Step 11: Reload the saved model
loaded_model = joblib.load("models/delay_model.pkl")


# Step 12: Compare predictions before and after reloading
original_probabilities = model.predict_proba(X_test)[:, 1]
reloaded_probabilities = loaded_model.predict_proba(X_test)[:, 1]

max_difference = float(np.max(
    np.abs(original_probabilities - reloaded_probabilities)
))


# Step 13: Save the reload check
reload_check = pd.DataFrame({
    "metric": [
        "test_rows_scored",
        "reload_max_abs_probability_diff",
        "threshold"
    ],
    "value": [
        len(X_test),
        max_difference,
        threshold
    ]
})

reload_check.to_csv(
    "outputs/part_d_reload_check.csv",
    index=False
)

print("\nReload check:")
print(reload_check.to_string(index=False))

print("\nModel packaging completed.")
