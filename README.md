Delhivery Last-Mile Delay Intelligence Pipeline
How to Run
1. Clone the repository
Open a terminal and run:
git clone https://github.com/poojapahchigar/delhivery-capstone.git
cd delhivery-capstone
2. Create and activate a virtual environment
For Windows PowerShell:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
3. Install the required libraries
pip install -r requirements.txt
4. Generate the dataset
python generate_data.py
This creates the synthetic shipment data in the data/ folder. The generator should produce 24 hubs, 1,224 raw shipment rows, and 48 new shipments.
5. Run Parts A, B, and C in order
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
Part A cleans the shipment data, creates the modelling table, and generates data-quality and exploratory-analysis outputs. Part B compares and evaluates regression models. Part C evaluates classification models, feature selection, tuning, and the alert threshold.
6. Run Part D, package the model, and score new shipments
python part_d_segments.py
python package_model.py
python predict_delay.py
- part_d_segments.py creates hub profiles, K-Means segments, cluster-separation results, and the tier-versus-segment mismatch report.
- package_model.py rebuilds the final classification pipeline using the configuration saved by Part C. It writes the model, threshold, feature order, imputation values, and model reload check.
- predict_delay.py scores the 48 new shipments and writes recommended actions to outputs/new_shipment_actions.csv.
Run these commands from the repository root and in this order. package_model.py regenerates models/delay_model.pkl. Use the compatible package versions pinned in requirements.txt, because a saved scikit-learn pickle may not be compatible across different versions.
Data Findings
Dataset Overview
- The original dataset contained 1,224 shipment records.
- After removing 24 duplicate records, 1,200 unique shipments remained in the modelling dataset.
- There were 278 delayed shipments, giving an overall delay rate of approximately 23.17%.
- Missing numeric values were handled using median imputation; missing payment modes were filled with Unknown.
- Inconsistent service-type names were standardized.
- Five parcel-weight records contained unusually high sentinel values and were treated as missing.
- Hub information was joined to shipment data, and the final modelling table had no missing values in the required features or target.
The new-shipment file contains 48 rows. Before cleaning, it has 3 missing distance_km values, 2 missing parcel_weight_kg values, 1 missing declared_value, 2 missing payment_mode values, and 8 service-type spelling variations. The prediction script applies the shared cleaning and feature-building functions and uses the saved training-data medians for numeric imputation.
Delivery Delay Patterns
- Express shipments had the highest delay rate among service types, at approximately 48.97%.
- Economy shipments had a delay rate of approximately 8.57%, while Standard shipments had a delay rate of approximately 20.56%.
- Hub Tier 3 had a delay rate of approximately 44%, compared with 21.5% for Tier 2 and 4% for Tier 1.
- Shipments travelling more than 600 km had a delay rate of approximately 85.23%.
These patterns suggest that service type, distance, and hub tier are useful dimensions for exploring delay risk in this dataset.
Correlation Findings
- promised_hours had a strong positive correlation with delivery_hours of approximately 0.84.
- distance_km had a positive correlation with is_delayed of approximately 0.55.
- n_stops also showed a positive correlation with delivery time.
- hub_daily_capacity had a negative correlation with delivery time, suggesting that higher-capacity hubs may be associated with shorter delivery times in this dataset.
These are patterns in a generated dataset and should not be interpreted as proof of causation or as guaranteed patterns in real-world operations.
Model Results (Regression)
Objective
The regression model predicts actual delivery time in hours for each shipment using shipment details, hub information, and operational features.
Data Split and Model Comparison
The dataset was divided into training, validation, and test sets:
Dataset	Number of shipments
Training	840
Validation	180
Test	180


The validation comparison was:
Model	Validation MAE	Validation RMSE	Validation R²
Linear Regression	3.922	4.891	0.931
Decision Tree	5.097	6.280	0.887
Random Forest	4.251	5.468	0.914
Gradient Boosting	3.917	4.917	0.931


Gradient Boosting and Linear Regression had very similar validation performance, and both performed better than the Decision Tree and Random Forest in this comparison.
Final Model Performance
The final Gradient Boosting model achieved the following test results:
Metric	Test result
Mean Absolute Error (MAE)	3.431 hours
Mean Squared Error (MSE)	19.993
Root Mean Squared Error (RMSE)	4.471 hours
R² score	0.950


The MAE means that predictions differed from actual delivery times by approximately 3.43 hours on average in the test set. The R² score indicates that the model explained a large proportion of delivery-time variation in this evaluation dataset.

These results are based on generated data. Performance on real shipment records may differ.
Model Results (Classification)
1. Blind Baseline
The blind baseline predicts every shipment as on-time. It achieved 76.67% accuracy on the validation set, but its recall and F1-score were both 0. This shows that accuracy alone is not enough for this imbalanced classification problem because the baseline fails to identify delayed shipments.
2. Feature Scores
SelectKBest with the ANOVA F-test was used to score the 17 input features using the training data. The seven highest-ranked features were:
- distance_km
- hub_tier
- hub_daily_capacity
- n_stops
- promised_hours
- rain_flag
- sale_event_flag
All five noise features ranked outside the top seven and had F-scores below 10. This suggests that the main operational features were more useful for predicting delays than the added noise features in this dataset.
3. Feature Selection
Random Forest models using 5, 7, 10, 12, and 17 features were compared. The best validation F1-score was 0.833 with seven features, so seven features were used for the tuning experiments.
4. Model Tuning
Grid Search, Randomized Search, and a default Random Forest were compared using the same seven features.
- Grid Search achieved a best cross-validation accuracy of 0.8952 and a validation F1-score of 0.775.
- Randomized Search achieved a best cross-validation accuracy of 0.8905 and a validation F1-score of 0.785.
- The default Random Forest achieved the highest validation F1-score of 0.833.
Tuning did not improve the validation F1-score in these experiments, so the default Random Forest was selected for the final classification pipeline.
5. Alert Threshold
Thresholds from 0.1 to 0.9 were tested. A threshold of 0.5 was selected as the highest threshold that met the validation recall requirement of at least 0.80. At this threshold, validation precision was 0.795 and recall was 0.833.

A lower threshold generally catches more delayed shipments but creates more false alerts. A higher threshold reduces alerts but may miss more delayed shipments.
6. Final Test Results
The final classification pipeline uses SelectKBest with seven features followed by a Random Forest with 100 trees. At the selected threshold of 0.5, the test results were:
Metric	Test result
Accuracy	0.867
Precision	0.689
Recall	0.756
F1-score	0.721
True negatives	125
False positives	14
False negatives	10
True positives	31


The model identified 31 of the 41 delayed shipments and missed 10. Of the 45 shipments flagged as delayed, 31 were actually delayed. For every 10 actually delayed shipments, the model identified about 7.6; for every 10 flagged shipments, about 6.9 were actually delayed in the test set. Test recall was lower than validation recall, so the model should be monitored on additional labelled shipment data.
Operations Recommendation
Shipment and Hub Priorities
1. Prioritize long-distance shipments. Shipments travelling more than 600 km showed a high delay rate in this dataset. Monitor them closely and intervene when the classifier indicates elevated risk.
2. Review hub performance. Tier 3 had a higher delay rate than the other tiers. Investigate workload, staffing, capacity, routing, and processing efficiency.
3. Monitor Express shipments. Express shipments had the highest observed delay rate. Review handling and possible bottlenecks for this service type.
4. Use predictions for early alerts. Use the regression model to estimate delivery time and the classifier to flag potentially delayed shipments for operational review.
5. Track performance continuously. Monitor delay rate, prediction errors, false alerts, and missed delays across hubs and distance bands.
Hub Segmentation Results
K-Means was applied to four standardized hub-level metrics: mean delivery hours, delay rate, mean distance, and mean number of stops. The hubs were ordered by ascending hub_id before clustering. Three clusters were used because operations requires three actionable playbooks.
Segment	Hubs	Mean delivery hours	Delay rate	Mean distance (km)	Mean stops
Standard	7	47.24	2.57%	188.93	55.37
Watch	8	56.81	15.50%	314.18	70.72
Escalate	9	64.65	46.00%	470.48	84.34


The cluster-separation results were:
K	Across-to-within ratio	Cluster sizes
2	2.894	15-9
3	5.393	9-8-7
4	6.305	8-7-5-4
5	6.887	8-7-4-3-2


The ratio increases for K=4 and K=5, but those choices produce clusters with fewer than six hubs. K=3 provides a practical balance between separation and group sizes and matches the business requirement for three playbooks.
Tier-versus-Segment Mismatches
The expected mapping is Tier 1 → Standard, Tier 2 → Watch, and Tier 3 → Escalate. Four hubs differ from that mapping:
Hub	City	Tier	Segment	Interpretation
HUB07	Kolkata	1	Watch	Higher risk profile than expected for Tier 1; investigate.
HUB13	Nagpur	2	Escalate	Higher risk profile than expected for Tier 2; prioritize review.
HUB15	Kochi	2	Escalate	Higher risk profile than expected for Tier 2; prioritize review.
HUB22	Dehradun	3	Watch	Lower-risk segment than expected for Tier 3; review what may explain the difference.


These mismatches are investigation priorities, not proof of operational failure or success.
New-Shipment Action Results
The prediction script scored all 48 new shipments at the selected threshold of 0.5.
Recommended action	Shipments
No action	40
Escalate to hub manager	6
Proactive customer call	2
Total flagged	8


Eight of 48 shipments (16.7%) were flagged. Flagged shipments at Escalate hubs were assigned Escalate to hub manager. Other flagged shipments were assigned Proactive customer call. Unflagged shipments received No action.

The script was run twice and produced the same displayed summary and first five predictions. The model reload check also recorded a maximum absolute probability difference of 0.0 across 180 test rows.

These recommendations are based on generated data and should be validated against real shipment records before operational use.
Production Plan and Limitations
Production Plan
If this pipeline were deployed in a real delivery operation, the following steps would be required:
1. Data integration: Connect the pipeline to reliable shipment, hub, dispatch, and delivery-status data sources.
2. Data validation: Validate incoming records, handle missing values, and check for unexpected values before generating predictions.
3. Model packaging: Use package_model.py to save the classification pipeline, threshold, feature order, and training-data imputation values. Keep the Python package versions compatible with the saved model.
4. Prediction and alerts: Run predict_delay.py on new shipments and route flagged shipments to the appropriate operational action.
5. Monitoring: Track model performance using actual outcomes, including regression MAE and classification precision, recall, F1-score, flagged rate, and missed delays.
6. Model updates: Evaluate performance monthly when enough labelled outcomes are available. Use quarterly retraining as an initial schedule, or retrain sooner if meaningful drift or performance deterioration is observed.
Proposed Monitoring Triggers
The following are proposed starting thresholds, not validated business limits. They should be reviewed with the operations team.
1. Flagged shipment rate: Investigate if more than 25% of shipments are flagged in a daily batch. The current generated batch has 8 flagged shipments out of 48 (16.7%).
2. Observed delay rate: Investigate if the rolling seven-day delay rate exceeds 30%.
3. Missed delays: Review the model and threshold if more than 25% of actually delayed shipments are missed in a sufficiently large labelled evaluation window.
4. Data quality: Investigate unexpected service types, missing required inputs after cleaning, or unsuccessful hub joins.
5. Model reproducibility: Investigate if the maximum probability difference between the saved and reloaded model exceeds 1e-9.
Drift and Sale-Event Monitoring
Monitor prediction volumes, delay rates, feature distributions, precision, recall, and regression errors weekly. Compare shipments from the 15–28 September sale-event window with comparable sale-event periods, rather than only with ordinary non-sale periods. Track changes in sale_event_flag, service type, distance, parcel weight, hub mix, and other input distributions.
Limitations
- The dataset is synthetic and may not represent real-world shipment behaviour.
- The classification model did not identify every delayed shipment in the test set. At the selected threshold of 0.5, it detected 31 of 41 delayed shipments and missed 10.
- False alerts can lead to unnecessary operational follow-ups, while missed delays can prevent teams from intervening early.
- The random train/validation/test split may not fully represent performance on future shipments, and shipments from the same hubs may appear in more than one split.
- Predicted probabilities have not been established as calibrated real-world delay probabilities.
- Median imputation is practical but does not explain why a value is missing.
- Model performance may change with different hubs, seasons, weather, shipment volumes, or operational processes.
- Correlations in the dataset do not establish causation.
- The project creates local output files and recommendations; it is not connected to a live logistics platform or automated alerting service.
- Real-world deployment requires validation on historical operational data, monitoring, data-quality checks, and appropriate security controls.
The models should be treated as decision-support tools, not guarantees of delivery time or delay outcomes.