
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Load the modelling table
modelling_table = pd.read_csv("data/modelling_table.csv")

print("Modelling table shape:", modelling_table.shape)
print(modelling_table["is_delayed"].value_counts())


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

# Select features and target
X = modelling_table[FEATURES]
y = modelling_table["is_delayed"]

# Split into training and temporary sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42,
    stratify=y
)

# Split temporary data into validation and test sets
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42,
    stratify=y_temp
)

print("\nTraining rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))

print("\nTraining class counts:")
print(y_train.value_counts())

print("\nValidation class counts:")
print(y_val.value_counts())

print("\nTest class counts:")
print(y_test.value_counts())


# Create the logistic regression model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

# Train the model
model.fit(X_train, y_train)

# Make predictions on validation data
val_prediction = model.predict(X_val)
val_probability = model.predict_proba(X_val)[:, 1]

# Calculate evaluation metrics
accuracy = accuracy_score(y_val, val_prediction)
precision = precision_score(y_val, val_prediction, zero_division=0)
recall = recall_score(y_val, val_prediction, zero_division=0)
f1 = f1_score(y_val, val_prediction, zero_division=0)
roc_auc = roc_auc_score(y_val, val_probability)

print("\nLogistic Regression Results")
print("Accuracy:", round(accuracy, 3))
print("Precision:", round(precision, 3))
print("Recall:", round(recall, 3))
print("F1-score:", round(f1, 3))
print("ROC-AUC:", round(roc_auc, 3))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, val_prediction))


# Create the classification models
models = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),
    "decision_tree": DecisionTreeClassifier(
        max_depth=6, random_state=42
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=100, random_state=42
    ),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
}

# Compare the models on validation data
model_results = []

for model_name, model in models.items():
    model.fit(X_train, y_train)

    val_prediction = model.predict(X_val)
    val_probability = model.predict_proba(X_val)[:, 1]

    model_results.append({
        "model": model_name,
        "accuracy": round(
            accuracy_score(y_val, val_prediction), 3
        ),
        "precision": round(
            precision_score(
                y_val, val_prediction, zero_division=0
            ), 3
        ),
        "recall": round(
            recall_score(
                y_val, val_prediction, zero_division=0
            ), 3
        ),
        "f1": round(
            f1_score(y_val, val_prediction, zero_division=0), 3
        ),
        "roc_auc": round(
            roc_auc_score(y_val, val_probability), 3
        )
    })

model_metrics = pd.DataFrame(model_results)

print("\nClassification Model Comparison")
print(model_metrics)

# Save classification model comparison results
model_metrics.to_csv(
    "outputs/part_c_metrics.csv",
    index=False
)

print("\nClassification metrics saved.")

# Evaluate the best model on validation data
best_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

best_model.fit(X_train, y_train)

val_prediction = best_model.predict(X_val)

cm = confusion_matrix(y_val, val_prediction)

print("\nGradient Boosting Confusion Matrix")
print(cm)

# Save the confusion matrix
cm_df = pd.DataFrame(
    cm,
    index=["actual_on_time", "actual_delayed"],
    columns=["predicted_on_time", "predicted_delayed"]
)

cm_df.to_csv("outputs/part_c_confusion_matrix.csv")

print("Confusion matrix saved.")


# Calculate feature scores using training data only
feature_selector = SelectKBest(score_func=f_classif, k="all")
feature_selector.fit(X_train, y_train)

feature_scores = pd.DataFrame({
    "feature": FEATURES,
    "f_score": feature_selector.scores_
})

feature_scores = feature_scores.sort_values(
    "f_score", ascending=False
).reset_index(drop=True)

feature_scores["f_rank"] = range(1, len(feature_scores) + 1)

feature_scores.to_csv(
    "outputs/part_c_feature_scores.csv",
    index=False
)

print("\nFeature scores saved.")
print(feature_scores)


# Compare different numbers of selected features
k_values = [5, 7, 10, 12, 17]
selectk_results = []

for k in k_values:
    model = Pipeline([
        ("selector", SelectKBest(score_func=f_classif, k=k)),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    val_prediction = model.predict(X_val)

    selectk_results.append({
        "k": k,
        "val_f1": round(
            f1_score(y_val, val_prediction, zero_division=0), 3
        )
    })

selectk_table = pd.DataFrame(selectk_results)

selectk_table.to_csv(
    "outputs/part_c_selectk.csv",
    index=False
)

print("\nFeature selection comparison")
print(selectk_table)
print("\nFeature selection results saved.")


# Select the top 7 features
selector = SelectKBest(score_func=f_classif, k=7)
X_train_7 = selector.fit_transform(X_train, y_train)
X_val_7 = selector.transform(X_val)

# Set the parameters to test
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10]
}

# Grid Search
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5
)
grid_search.fit(X_train_7, y_train)

grid_prediction = grid_search.predict(X_val_7)

# Randomized Search
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_grid,
    n_iter=9,
    cv=5,
    random_state=42
)
random_search.fit(X_train_7, y_train)

random_prediction = random_search.predict(X_val_7)

# Default Random Forest using the same 7 features
default_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
default_model.fit(X_train_7, y_train)

default_prediction = default_model.predict(X_val_7)

print("\nGrid Search best parameters:")
print(grid_search.best_params_)
print("Grid Search validation F1:",
      round(f1_score(y_val, grid_prediction), 3))

print("\nRandomized Search best parameters:")
print(random_search.best_params_)
print("Randomized Search validation F1:",
      round(f1_score(y_val, random_prediction), 3))

print("\nDefault Random Forest validation F1:",
      round(f1_score(y_val, default_prediction), 3))


# Save the tuning comparison results
tuning_results = pd.DataFrame([
    {
        "search": "grid_search",
        "candidates": 27,
        "cv_folds": 5,
        "total_fits": 135,
        "best_params": str(grid_search.best_params_),
        "best_cv_accuracy": round(grid_search.best_score_, 4),
        "val_f1": round(
            f1_score(y_val, grid_prediction, zero_division=0), 3
        )
    },
    {
        "search": "randomized_search",
        "candidates": 9,
        "cv_folds": 5,
        "total_fits": 45,
        "best_params": str(random_search.best_params_),
        "best_cv_accuracy": round(random_search.best_score_, 4),
        "val_f1": round(
            f1_score(y_val, random_prediction, zero_division=0), 3
        )
    },
    {
        "search": "default_forest_same_7_features",
        "candidates": 1,
        "cv_folds": 0,
        "total_fits": 1,
        "best_params": str({
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_leaf": 1
        }),
        "best_cv_accuracy": None,
        "val_f1": round(
            f1_score(y_val, default_prediction, zero_division=0), 3
        )
    }
])

tuning_results.to_csv(
    "outputs/part_c_tuning.csv",
    index=False
)

print("\nTuning results saved:")
print(tuning_results)


# Check different alert thresholds using validation probabilities
val_probability = default_model.predict_proba(X_val_7)[:, 1]

threshold_values = [
    0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9
]

threshold_results = []

for threshold in threshold_values:
    threshold_prediction = (
        val_probability >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        threshold_prediction,
        labels=[0, 1]
    ).ravel()

    threshold_results.append({
        "threshold": threshold,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "accuracy": round(
            accuracy_score(y_val, threshold_prediction), 3
        ),
        "precision": round(
            precision_score(
                y_val, threshold_prediction, zero_division=0
            ), 3
        ),
        "recall": round(
            recall_score(
                y_val, threshold_prediction, zero_division=0
            ), 3
        ),
        "f1": round(
            f1_score(
                y_val, threshold_prediction, zero_division=0
            ), 3
        )
    })

threshold_table = pd.DataFrame(threshold_results)

threshold_table.to_csv(
    "outputs/part_c_thresholds.csv",
    index=False
)

print("\nAlert threshold comparison")
print(threshold_table.to_string(index=False))

# Choose the highest threshold with recall >= 0.80
eligible_thresholds = threshold_table[
    threshold_table["recall"] >= 0.80
]

if len(eligible_thresholds) > 0:
    chosen_threshold = eligible_thresholds["threshold"].max()
else:
    chosen_threshold = 0.1

print("\nChosen alert threshold:", chosen_threshold)


# Build the final model using the selected 7 features
final_model = Pipeline([
    ("selector", SelectKBest(score_func=f_classif, k=7)),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

# Train using training data only
final_model.fit(X_train, y_train)

print("\nFinal model trained using 7 selected features.")
print("Chosen alert threshold:", chosen_threshold)


# Evaluate the final model on test data once
test_probability = final_model.predict_proba(X_test)[:, 1]

test_prediction = (
    test_probability >= chosen_threshold
).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_prediction,
    labels=[0, 1]
).ravel()

final_test_results = pd.DataFrame([{
    "threshold": chosen_threshold,
    "TN": tn,
    "FP": fp,
    "FN": fn,
    "TP": tp,
    "accuracy": round(
        accuracy_score(y_test, test_prediction), 3
    ),
    "precision": round(
        precision_score(
            y_test, test_prediction, zero_division=0
        ), 3
    ),
    "recall": round(
        recall_score(
            y_test, test_prediction, zero_division=0
        ), 3
    ),
    "f1": round(
        f1_score(
            y_test, test_prediction, zero_division=0
        ), 3
    )
}])

final_test_results.to_csv(
    "outputs/part_c_final_test.csv",
    index=False
)

print("\nFinal test results")
print(final_test_results.to_string(index=False))
print("\nFinal test results saved.")


# Save the final model configuration
final_config = pd.DataFrame([{
    "selector_k": 7,
    "n_estimators": 100,
    "max_depth": "unlimited",
    "min_samples_leaf": 1,
    "threshold": chosen_threshold,
    "chosen_from": "default_forest_same_7_features"
}])

final_config.to_csv(
    "outputs/part_c_final_config.csv",
    index=False
)

print("\nFinal configuration saved.")
print(final_config.to_string(index=False))


# Compare the blind baseline with the trained models
baseline_prediction = [0] * len(y_val)

baseline_tn, baseline_fp, baseline_fn, baseline_tp = confusion_matrix(
    y_val, baseline_prediction, labels=[0, 1]
).ravel()

model_comparison = [{
    "model": "blind_baseline",
    "accuracy": round(accuracy_score(y_val, baseline_prediction), 4),
    "precision": round(precision_score(
        y_val, baseline_prediction, zero_division=0
    ), 4),
    "recall": round(recall_score(
        y_val, baseline_prediction, zero_division=0
    ), 4),
    "f1": round(f1_score(
        y_val, baseline_prediction, zero_division=0
    ), 4),
    "roc_auc": None,
    "TN": baseline_tn,
    "FP": baseline_fp,
    "FN": baseline_fn,
    "TP": baseline_tp
}]

for model_name, model in models.items():
    model.fit(X_train, y_train)
    prediction = model.predict(X_val)
    probability = model.predict_proba(X_val)[:, 1]

    tn, fp, fn, tp = confusion_matrix(
        y_val, prediction, labels=[0, 1]
    ).ravel()

    model_comparison.append({
        "model": model_name,
        "accuracy": round(accuracy_score(y_val, prediction), 4),
        "precision": round(precision_score(
            y_val, prediction, zero_division=0
        ), 4),
        "recall": round(recall_score(
            y_val, prediction, zero_division=0
        ), 4),
        "f1": round(f1_score(
            y_val, prediction, zero_division=0
        ), 4),
        "roc_auc": round(roc_auc_score(y_val, probability), 4),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    })

model_comparison_df = pd.DataFrame(model_comparison)
model_comparison_df.to_csv("outputs/part_c_models.csv", index=False)

print("\nOfficial model comparison saved:")
print(model_comparison_df.to_string(index=False))
