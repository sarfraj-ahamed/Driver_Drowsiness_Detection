from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np
import os

def train_and_save_model(X, y, model_path):
    num_samples = X.shape[0]
    X = X.reshape(num_samples, -1)  # Flatten images

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    try:
        data_file = "processed_data.npy"
        labels_file = "labels.npy"

        if not os.path.exists(data_file) or not os.path.exists(labels_file):
            print("Error: Processed dataset not found! Run data_preprocessing.py first.")
        else:
            X = np.load(data_file)
            y = np.load(labels_file)

            model_path = "models/drowsiness_model.pkl"
            os.makedirs("models", exist_ok=True)

            train_and_save_model(X, y, model_path)
    except Exception as e:
        print(f"Error during training: {e}")
