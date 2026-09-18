# Predicting Earthquake Building Damage Grades

## The Problem

After the 7.8 magnitude earthquake that hit Nepal on April 25, 2015, surveys were conducted to assess damage to buildings. Given the huge number of buildings and structural variations, manual post-earthquake assessment is extremely slow. Your job is to predict the damage grade (1 = low, 2 = medium, 3 = near-total destruction) of a building based on its structural and ownership features.

## Why This Is Uncommon and Worth Doing

Most people practice on Titanic, house prices, or churn. This is a multi-class ordinal classification problem on one of the largest post-disaster datasets ever collected, containing valuable information on earthquake impacts, household conditions, and socio-economic-demographic statistics. It throws real-world messiness at you — heavy categorical features like foundation type, roof material, land surface condition, and legal ownership status — mixed with geo-location IDs and normalized numerical features.

- 39 features in the training dataset with ~260K rows
- Data was collected through surveys by Kathmandu Living Labs and the Central Bureau of Statistics under Nepal's National Planning Commission
- Evaluation metric is micro-averaged F1 score

## What Makes It a Solid Intermediate Challenge

The dataset has a class imbalance (grade 2 dominates), so accuracy will mislead you — you need to optimize F1. The 38 features include a mix of binary, categorical, and continuous types, so you'll practice encoding strategies (one-hot vs. target encoding, especially for high-cardinality geo columns like `geo_level_1_id` through `geo_level_3_id`). Preprocessing involves label encoding of categorical columns, handling missing values, and normalization using MinMaxScaler. And since the target is ordinal (1 < 2 < 3), you can experiment with whether treating it as standard multi-class vs. ordinal regression changes your results.

## Suggested Learning Path

1. Start with EDA — look at damage distribution across building ages, foundation types, and geographic regions.
2. Build a logistic regression baseline, move to random forest, and finally try XGBoost or LightGBM.
3. Use stratified k-fold cross-validation.
4. For feature engineering, try interaction features (e.g., age × floor count), binning building age, and different encoding strategies for the geo columns.
5. Finally, use SHAP to explain which building characteristics are the strongest predictors of severe damage.
