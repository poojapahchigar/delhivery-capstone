
import os
import pandas as pd
import numpy as np
import joblib

from src.shipment_prep import (
    FEATURES,
    clean_shipments,
    build_features
)


# Step 1: Load the new shipments, hub data, and saved model files
new_shipments = pd.read_csv("data/new_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

model = joblib.load("models/delay_model.pkl")

threshold = float(
    pd.read_csv("models/delay_threshold.csv").iloc[0]["threshold"]
)

feature_order = pd.read_csv(
    "models/feature_order.csv"
)["feature"].tolist()

impute_table = pd.read_csv("models/impute_values.csv")

medians = dict(zip(
    impute_table["column"],
    impute_table["median"]
))

hub_segments = pd.read_csv("outputs/hub_segments.csv")


# Step 2: Clean new shipments using the saved training medians
cleaned_shipments, _ = clean_shipments(
    new_shipments,
    medians=medians
)


# Step 3: Build the shipment features using shared preprocessing
featured_shipments = build_features(
    cleaned_shipments,
    hubs
)


# Step 4: Arrange the features in the saved model's expected order
X_new = featured_shipments[feature_order]


# Step 5: Predict the probability of delay
delay_probabilities = model.predict_proba(X_new)[:, 1]

featured_shipments["delay_probability"] = np.round(
    delay_probabilities, 4
)

featured_shipments["flagged"] = (
    delay_probabilities >= threshold
)


# Step 6: Attach hub segments
featured_shipments = featured_shipments.merge(
    hub_segments[["hub_id", "segment"]],
    on="hub_id",
    how="left",
    validate="many_to_one"
)


# Step 7: Recommend an operational action
def recommend_action(row):
    if not row["flagged"]:
        return "No action"

    if row["segment"] == "Escalate":
        return "Escalate to hub manager"

    return "Proactive customer call"


featured_shipments["action"] = featured_shipments.apply(
    recommend_action,
    axis=1
)


# Step 8: Select the required output columns
output_columns = [
    "shipment_id",
    "hub_id",
    "segment",
    "delay_probability",
    "flagged",
    "action"
]

output = featured_shipments[output_columns].copy()


# Step 9: Save the results
os.makedirs("outputs", exist_ok=True)

output.to_csv(
    "outputs/new_shipment_actions.csv",
    index=False
)


# Step 10: Display a summary
print("New shipments scored:", len(output))
print("\nFlagged shipments:", int(output["flagged"].sum()))

print("\nActions:")
print(output["action"].value_counts().to_string())

print("\nFirst five predictions:")
print(output.head().to_string(index=False))

print("\nSaved: outputs/new_shipment_actions.csv")
