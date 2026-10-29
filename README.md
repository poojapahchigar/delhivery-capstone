# Delhivery Last-Mile Delay Intelligence Pipeline

## How to Run

### 1. Install the required libraries

Open the terminal in the project folder and run:

```bash
pip install -r requirements.txt
```

### 2. Generate the dataset

Run the data generation script:

```bash
python generate_data.py
```

This creates the synthetic shipment data files in the `data` folder.

### 3. Run the data cleaning and EDA

```bash
python part_a_clean_eda.py
```

This cleans the shipment data, creates the modelling table and generates the Part A outputs.

### 4. Run the regression model

```bash
python part_b_regression.py
```

This compares regression models and saves the evaluation results.

### 5. Run the classification model

```bash
python part_c_classification.py
```

This compares classification models, evaluates feature selection and tuning, selects an alert threshold, and saves the final test results.

### 6. Run the remaining project scripts

Run the segmentation, model packaging and prediction scripts after completing their implementation.


## Data Findings

### Dataset Overview

- The original dataset contained 1,224 shipment records.
- After removing 24 duplicate records, 1,200 unique shipments remained.
- Missing values were handled using median imputation for numerical columns and `Unknown` for missing payment modes.
- Inconsistent service type names were standardized.
- Five parcel weight records contained unusually high sentinel values and were treated as missing.
- Hub information was merged with shipment data, and the final modelling table contained no missing values in the required features or target.

### Delivery Delay Patterns

- Out of 1,200 shipments, 278 were delayed, giving an overall delay rate of **23.17%**.
- Express shipments had the highest delay rate among service types, at approximately **48.97%**.
- Economy shipments had a delay rate of approximately **8.57%**, while Standard shipments had a delay rate of approximately **20.56%**.
- Hub Tier 3 had a delay rate of **44%**, compared with 21.5% for Tier 2 and 4% for Tier 1.
- Longer-distance shipments showed a higher delay rate. Shipments travelling more than 600 km had a delay rate of approximately **85.23%**.

### Correlation Findings

- `promised_hours` had a strong positive correlation with `delivery_hours` of approximately 0.84.
- `distance_km` had a positive correlation with `is_delayed` of approximately 0.55.
- `n_stops` also showed a positive correlation with delivery time.
- `hub_daily_capacity` had a negative correlation with delivery time, suggesting that higher-capacity hubs may be associated with shorter delivery times in this dataset.

These findings describe patterns in the generated dataset and should not be interpreted as proof of causation or as guaranteed patterns in real-world operations.

## Model Results (Regression)

### Objective

The regression model predicts the actual delivery time in hours for each shipment using shipment details, hub information, and operational features.

### Model Evaluation

The dataset was divided into training, validation, and test sets:

| Dataset | Number of shipments |
|---|---:|
| Training | 840 |
| Validation | 180 |
| Test | 180 |

Several regression algorithms were compared using validation data.

| Model | Validation MAE | Validation RMSE | Validation R² |
|---|---:|---:|---:|
| Linear Regression | 3.922 | 4.891 | 0.931 |
| Decision Tree | 5.097 | 6.280 | 0.887 |
| Random Forest | 4.251 | 5.468 | 0.914 |
| Gradient Boosting | 3.917 | 4.917 | 0.931 |

Gradient Boosting achieved the highest validation R², while Linear Regression had a very similar validation performance. Both performed considerably better than the baseline model.

### Final Model Performance

The final Gradient Boosting model was trained using the combined training and validation datasets and evaluated on the test dataset.

| Metric | Test Result |
|---|---:|
| Mean Absolute Error (MAE) | 3.431 hours |
| Mean Squared Error (MSE) | 19.993 |
| Root Mean Squared Error (RMSE) | 4.471 hours |
| R² Score | 0.950 |

The model achieved an R² score of approximately 0.95, indicating that it explained a large proportion of the variation in delivery time in the test dataset. Its MAE of approximately 3.43 hours means that its predictions differed from actual delivery times by about 3.43 hours on average.

These results are based on the generated dataset. Performance on real shipment data may differ.

## Operations Recommendation

Based on the analysis of the generated shipment data, the following actions could help improve last-mile delivery operations:

1. **Prioritize long-distance shipments:** Shipments travelling more than 600 km showed a high delay rate. These shipments should receive closer monitoring and earlier escalation when delays are likely.

2. **Review hub performance:** Hub Tier 3 had a higher delay rate than the other hub tiers. Operations teams should investigate capacity, workload, staffing, and processing efficiency at these hubs.

3. **Monitor Express shipments:** Express shipments showed a high delay rate in this dataset. Their promised delivery times and operational handling should be reviewed to identify possible bottlenecks.

4. **Use delay predictions for early alerts:** The regression model can estimate delivery time, while the classification model can flag shipments that may be delayed. Operations teams could use these predictions to prioritize follow-ups and customer notifications.

5. **Track performance continuously:** Monitor delay rate, prediction errors, false alerts, and missed delays across different hubs and distance bands. Use these results to improve operational decisions over time.

These recommendations are based on patterns in a generated dataset. They should be validated against real shipment records before being used for operational decisions.

## Production Plan and Limitations

### Production Plan

If this pipeline were deployed in a real delivery operation, the following steps would be required:

1. **Data integration:** Connect the pipeline to reliable shipment, hub, dispatch, and delivery-status data sources.
2. **Data validation:** Validate incoming records, handle missing values, and check for unexpected values before generating predictions.
3. **Model deployment:** Package the selected regression and classification models so they can be loaded and used for new shipments.
4. **Prediction and alerts:** Generate delivery-time estimates and delay-risk predictions for incoming shipments. Flag high-risk shipments for operational review.
5. **Monitoring:** Track model performance using actual delivery outcomes, including MAE for regression and precision, recall, and F1-score for classification.
6. **Model updates:** Retrain and validate the models periodically when shipment patterns or operational conditions change.

### Limitations

- The dataset is generated for this capstone project and may not represent real-world shipment behaviour.
- The classification model did not identify every delayed shipment in the test set. At the selected threshold of 0.5, it detected 31 of 41 delayed shipments and missed 10.
- False alerts can lead to unnecessary operational follow-ups, while missed delays can prevent teams from intervening early.
- Model performance may change when applied to new hubs, different seasons, unusual weather, or changing shipment volumes.
- Correlations in the dataset do not establish causation.
- Real-world deployment would require validation on historical operational data, ongoing monitoring, data-quality checks, and appropriate security controls.

The models should therefore be treated as decision-support tools rather than guarantees of delivery time or delay.


## Model Results (Classification)

### 1. Blind Baseline

The blind baseline predicts every shipment as on-time. It achieved 76.67% accuracy on the validation set, but its recall and F1-score were both 0. This shows that accuracy alone is not enough for this imbalanced classification problem because the model fails to identify delayed shipments.

### 2. Feature Scores

I used SelectKBest with the ANOVA F-test to score the 17 input features using the training data. The seven highest-ranked features were distance_km, hub_tier, hub_daily_capacity, n_stops, promised_hours, rain_flag and sale_event_flag.

All five noise features ranked outside the top seven and had F-scores below 10. This suggests that the main operational features are more useful for predicting delays than the added noise features.

### 3. Feature Selection

I compared Random Forest models using 5, 7, 10, 12 and 17 features. The best validation F1-score was 0.833 with seven features. Therefore, I used seven features for the tuning experiments.

### 4. Model Tuning

I compared Grid Search, Randomized Search and a default Random Forest using the same seven features.

Grid Search achieved a best cross-validation accuracy of 0.8952 and a validation F1-score of 0.775. Randomized Search achieved a best cross-validation accuracy of 0.8905 and a validation F1-score of 0.785. The default Random Forest achieved the highest validation F1-score of 0.833.

Tuning did not improve the validation F1-score in these experiments, so the default Random Forest was selected.

### 5. Alert Threshold

I tested thresholds from 0.1 to 0.9 and selected 0.5, the highest threshold with validation recall of at least 0.80. At this threshold, validation precision was 0.795 and recall was 0.833.

A lower threshold catches more delayed shipments but creates more false alerts. A higher threshold reduces false alerts but can miss more delayed shipments.

### 6. Final Test Results

The final model was evaluated on the test set using the selected seven features and a threshold of 0.5.

| Metric | Result |
|---|---:|
| Accuracy | 0.867 |
| Precision | 0.689 |
| Recall | 0.756 |
| F1-score | 0.721 |
| True negatives | 125 |
| False positives | 14 |
| False negatives | 10 |
| True positives | 31 |

The model identified 31 of the 41 delayed shipments in the test set and missed 10. Of the 45 shipments flagged as delayed, 31 were actually delayed. The test recall was lower than the validation recall, so model performance should be monitored on additional shipment data.