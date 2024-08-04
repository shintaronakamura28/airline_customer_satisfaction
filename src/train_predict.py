from sklearn.pipeline import Pipeline

def train_and_predict_model(X_train, y_train, X_test, preprocessor, model):
    
    # Combine the preprocessing pipeline and model into a single pipeline
    model_pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model),
    ])
    
    # Fit the pipeline on the training data
    model_pipe.fit(X_train, y_train)
    
    # Save the predictions for both train and test sets
    train_preds = model_pipe.predict(X_train)
    test_preds = model_pipe.predict(X_test)
    
    return train_preds, test_preds