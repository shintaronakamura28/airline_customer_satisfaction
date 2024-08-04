import pandas as pd 
from imblearn.over_sampling import SMOTE
import seaborn as sns
import numpy as np
from imblearn.pipeline import Pipeline as ImbPipeline

def make_sampler_pipeline(sampler):
    return ImbPipeline([
        ('sampler', sampler)
        ])
    
# Function to preprocess and rebalance the data
def preprocess_and_rebalance_data(preprocessor, X_train, X_test, y_train):
    
    # Transform the training data into the fitted transformer
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Create a sampling pipeline and use SMOTE
    sampler = make_sampler_pipeline(SMOTE(random_state=42))
    
    # Balance the training data 
    X_train_balanced, y_train_balanced = sampler.fit_resample(X_train_transformed, y_train)
    
    return X_train_balanced, X_test_transformed, y_train_balanced