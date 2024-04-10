# Predicting Severity of Pedestrian Collisions

## Objective
Develop a predictive modeling tool to assess the most dangerous locations in the City of Toronto for traffic-related injuries and fatalities involving pedestrians.

## Approach
1. **Data Preparation**: Merged two datasets to create a unified pandas dataframe focused on pedestrian-involved collisions.
2. **Exploratory Data Analysis**: Analyzed the effect of various factors (age, speed, driver condition, impact location, maneuver type, pedestrian action, collision type, vehicle class, etc.) on the severity of pedestrian-involved collision injuries.
3. **Feature and Target Variable Selection**:
  - Target Variable: 'involved_injury_class' categorized into two classes (0 for non-serious injury, 1 for serious injury)
  - Features: 'month', 'involved_age', 'pedestrian_action', 'pedestrian_condition', 'collision_type', 'vehicle_class', 'actual_speed', 'longitude', 'latitude', and others
4. **Data Preprocessing**:
  - Encoded categorical features using OneHotEncoder
  - Handled missing values with KNNImputer
  - Split data into train (80%) and test (20%) sets
5. **Feature Selection**: Investigated feature importance using SelectKBest with 'f_classif' and 'mutual_info_classif' score functions.
6. **Model Training and Evaluation**: Trained and evaluated several machine learning algorithms (Gradient Boosting, Decision Tree, SVM, KNN) using accuracy, recall, F-score, precision, and ROC AUC as evaluation metrics.
7. **Feature Importance**: Utilized Permutation Importance method to identify the most influential features for each model.

## Key Findings
- Influential features included vehicle speed, pedestrian age, collision location, and vehicle class.
- Due to imbalanced data, the models achieved a fair ROC AUC score of around 0.68.

## Future Improvements
- Perform feature selection to improve model performance.
- Apply data balancing techniques to handle the imbalanced dataset.