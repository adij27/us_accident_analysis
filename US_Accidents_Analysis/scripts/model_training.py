# Model Training
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

def split_data(df, target, test_size=0.2, random_state=42):
    """Splits the dataset into training and testing sets."""
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_random_forest(X_train, y_train):
    """Trains a Random Forest classifier."""
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf

def evaluate_model(model, X_test, y_test):
    """Evaluates the model using classification metrics."""
    y_pred = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    if len(y_test.unique()) == 2:  # Binary classification
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        print("ROC AUC Score:", auc)

# Example usage
if __name__ == "__main__":
    df = pd.read_csv("data/Processed_US_Accidents.csv")
    X_train, X_test, y_train, y_test = split_data(df, target="Severity")

    rf_model = train_random_forest(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test)
