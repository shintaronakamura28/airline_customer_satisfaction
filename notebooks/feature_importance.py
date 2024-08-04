import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

#fuinction to train the model
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

#fucntion to get feature importance
def get_feature_importance(model, fearture_names):
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_names, 
         'importance': importance,
         }).sort_values(by='importance', ascending=False)
    
    return feature_importance

#plot the features with a color palette
def plot_feature_importance(feature_importance, title):
    plt.figure(figsize=(10,5))

    #use a color palette
    sns.barplot(
        x='importance', 
        y='feature', 
        data=feature_importance,
        palette='viridis',
        hue='feature'
        )
    
    plt.title(title)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show 