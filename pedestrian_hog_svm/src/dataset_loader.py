import os
import cv2
import numpy as np
from hog_feature import extract_hog

def load_dataset(pos_dir, neg_dir):
    X = []
    y = []

    # Positive images → label 1
    for file in os.listdir(pos_dir):
        path = os.path.join(pos_dir, file)
        img = cv2.imread(path)
        if img is None:
            continue
        features = extract_hog(img)
        X.append(features)
        y.append(1)

    # Negative images → label 0
    for file in os.listdir(neg_dir):
        path = os.path.join(neg_dir, file)
        img = cv2.imread(path)
        if img is None:
            continue
        features = extract_hog(img)
        X.append(features)
        y.append(0)

    return np.array(X), np.array(y)
