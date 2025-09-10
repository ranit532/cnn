
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Define paths
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
LABELS_CSV_PATH = os.path.join(RAW_DATA_DIR, "labels.csv")

def main():
    """
    Splits the raw data into training and testing sets.
    """
    print("Processing data...")

    # Create the output directory if it doesn't exist
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)

    # Load the labels
    df = pd.read_csv(LABELS_CSV_PATH)

    # Split the data
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42, 
        stratify=df["label"]
    )

    # Save the splits
    train_csv_path = os.path.join(PROCESSED_DATA_DIR, "train.csv")
    test_csv_path = os.path.join(PROCESSED_DATA_DIR, "test.csv")
    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)

    print(f"Data successfully split into training and testing sets.")
    print(f"Training set size: {len(train_df)}")
    print(f"Testing set size: {len(test_df)}")
    print(f"Training set saved to: {train_csv_path}")
    print(f"Testing set saved to: {test_csv_path}")

if __name__ == "__main__":
    main()
