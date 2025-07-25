import pandas as pd
import numpy as np
import os 
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score,cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer       

# Define file paths for model and pipeline
MODEL_PKL = "modle.pkl"
PIPELINE_PKL =  "pipeline.pkl"

# Function to build the preprocessing pipeline
def pipene_bulder(num_att,cat_att):
    nums_pipeline  = Pipeline([
        ("scaler",StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ("Onehotencoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    full_pieline = ColumnTransformer([
        ("nums",nums_pipeline,num_att),
        ("cat",cat_pipeline,cat_att),
    ])
    return full_pieline

# Check if model and pipeline already exist
if not os.path.exists(MODEL_PKL) or not os.path.exists(PIPELINE_PKL):
    # Load the dataset
    heart_df = pd.read_csv("heart.csv")
    # Create age categories for stratified sampling
    heart_df["age_cat"] = pd.cut(
        heart_df["Age"],
        bins=[20,25,30,35,40,45,50,55,60,65,70,75,np.inf],
        labels=[1,2,3,4,5,6,7,8,9,10,11,12]
    )
    # Perform stratified split based on age categories
    split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    for test_index,train_index in split.split(heart_df,heart_df["age_cat"]):
        stect_test_set = heart_df.loc[test_index].drop("age_cat",axis=1)
        stect_train_set = heart_df.loc[train_index].drop("age_cat",axis=1)
    # Use the training set for further processing
    heart_df = stect_train_set.copy()
    # Separate the label from the features
    heart_df_lables = heart_df["HeartDisease"].copy()
    heart_df = heart_df.drop("HeartDisease",axis=1)
    # Separate numerical and categorical columns
    heart_num = heart_df.select_dtypes(include = np.number).columns.tolist()
    heart_cat = heart_df.select_dtypes(exclude = np.number).columns.tolist()
    # Build and fit the pipeline
    pipeline = pipene_bulder(heart_num,heart_cat)
    heart_preped = pipeline.fit_transform(heart_df)
    # Train the RandomForestClassifier
    modle  = RandomForestClassifier(random_state=42)
    modle.fit(heart_preped,heart_df_lables)
    # Save the trained model and pipeline
    joblib.dump(modle,MODEL_PKL)
    joblib.dump(pipeline,PIPELINE_PKL)
else:
    # Load the model and pipeline if they exist
    modle = joblib.load(MODEL_PKL)
    pipeline = joblib.load(PIPELINE_PKL)
    # Load new input data
    input_data = pd.read_csv("input.csv")
    # Transform input data using the pipeline
    tansfo_data = pipeline.transform(input_data)
    # Make predictions
    pred = modle.predict(tansfo_data)
    # Add predictions to the input data and save
    input_data['HeartDisease'] = pred
    input_data.to_csv("Output.csv",index=False)