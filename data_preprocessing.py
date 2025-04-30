import os
import cv2
import numpy as np

def load_and_preprocess_data(dataset_path, img_size=32):
    categories = ['Drowsy', 'Non-Drowsy']
    data, labels = [], []
    
    for category in categories:
        path = os.path.join(dataset_path, category)
        if not os.path.exists(path):
            print(f"Warning: Folder '{path}' not found! Creating an empty folder.")
            os.makedirs(path, exist_ok=True)
            continue

        label = categories.index(category)
        
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (img_size, img_size))
            img = img / 255.0  # Normalize
            data.append(img)
            labels.append(label)

    return np.array(data).reshape(-1, img_size, img_size, 1), np.array(labels)

if __name__ == "__main__":
    dataset_path = "dataset/"
    data_file = "processed_data.npy"
    labels_file = "labels.npy"

    if not os.path.exists(dataset_path):
        print("Error: Dataset folder not found! Creating an empty dataset folder.")
        os.makedirs(dataset_path, exist_ok=True)
    
    X, y = load_and_preprocess_data(dataset_path)
    
    np.save(data_file, X)
    np.save(labels_file, y)
    
    print(f"Dataset saved: {data_file}, {labels_file}")
    print(f"Dataset loaded with {len(X)} samples.")
