import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer

# Load the dataset
heart_df = pd.read_csv("heart.csv")

# Create age categories for stratified sampling
heart_df["age_cat"] = pd.cut(
    heart_df["Age"],
    bins=[20,25,30,35,40,45,50,55,60,65,70,75,np.inf],
    labels=[1,2,3,4,5,6,7,8,9,10,11,12]
)

# Perform stratified split based on age categories
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for test_index, train_index in split.split(heart_df, heart_df["age_cat"]):
    stect_test_set = heart_df.loc[test_index].drop("age_cat", axis=1)
    stect_train_set = heart_df.loc[train_index].drop("age_cat", axis=1)

# Use the training set for further processing
heart_df = stect_train_set.copy()

# Separate the label from the features
heart_df_lables = heart_df["HeartDisease"].copy()
heart_df = heart_df.drop("HeartDisease", axis=1)

# Separate numerical and categorical columns
heart_num = heart_df.select_dtypes(include=np.number)
heart_cat = heart_df.select_dtypes(exclude=np.number)

# Pipeline for numerical features (scaling)
nums_pipeline = Pipeline([
    ("scaler", StandardScaler()),
])

# Pipeline for categorical features (one-hot encoding)
cat_pipeline = Pipeline([
    ("Onehotencoder", OneHotEncoder(handle_unknown="ignore")),
])

# Combine numerical and categorical pipelines
full_pieline = ColumnTransformer([
    ("nums", nums_pipeline, heart_num.columns.tolist()),
    ("cat", cat_pipeline, heart_cat.columns.tolist()),
])

# Preprocess the data
heart_preped = full_pieline.fit_transform(heart_df)

# Train a RandomForestClassifier
modle = RandomForestClassifier(random_state=42)
modle.fit(heart_preped, heart_df_lables)
rfg_pred = modle.predict(heart_preped)

# Evaluate RandomForestClassifier using cross-validation (F1 score)
modle_rmse = cross_val_score(modle, heart_preped, heart_df_lables, scoring="f1", cv=10)
print(pd.Series(modle_rmse).describe())

# Train a LogisticRegressionCV
log = LogisticRegressionCV(cv=10)
log.fit(heart_preped, heart_df_lables)
rfg_pred = log.predict(heart_preped)

# Evaluate LogisticRegressionCV using cross-validation (F1 score)
log_rmse = cross_val_score(log, heart_preped, heart_df_lables, scoring="f1", cv=10)
print(pd.Series(log_rmse).describe())

# Train a DecisionTreeRegressor (note: regressor used on classification labels)
dec_tree = DecisionTreeRegressor()
dec_tree.fit(heart_preped, heart_df_lables)
rfg_pred = dec_tree.predict(heart_preped)

# Evaluate DecisionTreeRegressor using cross-validation (F1 score)
log_rmse = cross_val_score(dec_tree, heart_preped, heart_df_lables, scoring="f1", cv=10)
print(pd.Series(log_rmse).describe())