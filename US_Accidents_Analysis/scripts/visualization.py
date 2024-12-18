# Visualization
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_heatmap(df, save_path):
    """Plots a heatmap of feature correlations."""
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.savefig(save_path)
    plt.show()

def plot_accidents_by_hour(df, save_path):
    """Plots accidents distribution by hour."""
    df['Hour'] = pd.to_datetime(df['Start_Time']).dt.hour
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Hour', data=df, palette='viridis')
    plt.title("Accidents by Hour")
    plt.xlabel("Hour of the Day")
    plt.ylabel("Number of Accidents")
    plt.savefig(save_path)
    plt.show()

def plot_accidents_by_state(df, save_path):
    """Plots accidents distribution by state."""
    state_counts = df['State'].value_counts()
    plt.figure(figsize=(15, 8))
    state_counts.plot(kind='bar', color='skyblue')
    plt.title("Accidents by State")
    plt.xlabel("State")
    plt.ylabel("Number of Accidents")
    plt.savefig(save_path)
    plt.show()

# Example usage
if __name__ == "__main__":
    df = pd.read_csv("data/Processed_US_Accidents.csv")

    plot_correlation_heatmap(df, "visualizations/correlation_heatmap.png")
    plot_accidents_by_hour(df, "visualizations/accidents_by_hour.png")
    plot_accidents_by_state(df, "visualizations/accidents_by_state.png")

    print("Visualizations generated and saved.")
