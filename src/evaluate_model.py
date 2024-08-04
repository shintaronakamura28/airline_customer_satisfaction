import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib

# Function to predict and compare the models
def evaluate_models(model, X_train, y_train, X_test, y_test):
    
    # Create prediction on the training set
    train_preds = np.rint(model.predict(X_train))
    test_preds = np.rint(model.predict(X_test))
    
    # Classification report on train set
    train_report = classification_report(y_train, train_preds)
    test_report = classification_report(y_test, test_preds)
    
    # Confusion matrix for the training set
    cm_train = confusion_matrix(y_train, train_preds)
    cm_test = confusion_matrix(y_test, test_preds)
    
    # Plot train summary and confusion matrix side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].text(0.01, 0.05, str(train_report), {'fontsize': 12}, fontproperties="monospace")
    axes[0].axis('off')
    disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train)
    disp_train.plot(ax=axes[1], cmap='Blues')
    axes[1].set_title("Confusion Matrix - Training Set")
    
    # Plot train summary and confusion matrix side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].text(0.01, 0.05, str(test_report), {'fontsize': 12}, fontproperties="monospace")
    axes[0].axis('off')
    disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test)
    disp_test.plot(ax=axes[1], cmap='Blues')
    axes[1].set_title("Confusion Matrix - Testing Set")
    
    plt.show()
    
    return train_report, test_report

if __name__ == '__main__':
    pass

def save_model(model, model_path):
    joblib.dump(model, model_path)
    print(f'Model Saved to {model_path}')