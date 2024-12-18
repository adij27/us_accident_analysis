# Preprocessing
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(file_path):
    """Loads the dataset from a CSV file."""
    return pd.read_csv(file_path)

def handle_missing_values(df):
    """Fills missing values or removes rows based on the dataset's needs."""
    df.fillna({'Temperature(F)': df['Temperature(F)'].mean(), 
               'Visibility(mi)': df['Visibility(mi)'].mean()}, inplace=True)
    df.dropna(inplace=True)  # Drops rows with critical missing data
    return df

def detect_and_treat_outliers(df, columns):
    """Handles outliers in numerical columns using capping."""
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    return df

def encode_features(df, categorical_cols):
    """Encodes categorical features."""
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def normalize_features(df, numerical_cols):
    """Normalizes numerical features using StandardScaler."""
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

# Example usage
if __name__ == "__main__":
    file_path = "data/US_Accidents_Dataset.csv"
    df = load_data(file_path)

    # Preprocessing pipeline
    df = handle_missing_values(df)
    df = detect_and_treat_outliers(df, ['Temperature(F)', 'Visibility(mi)', 'Wind_Speed(mph)'])
    df = encode_features(df, ['Weather_Condition'])
    df = normalize_features(df, ['Temperature(F)', 'Visibility(mi)', 'Wind_Speed(mph)'])

    # Save processed data
    df.to_csv("data/Processed_US_Accidents.csv", index=False)
    print("Preprocessing complete. Processed data saved.")
