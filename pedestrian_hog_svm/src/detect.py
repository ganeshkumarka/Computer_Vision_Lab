import cv2
import joblib
from skimage.feature import hog

MODEL_PATH = "../models/svm_model.pkl"

svm, _, _ = joblib.load(MODEL_PATH)

def detect_pedestrians(image_path):
    img = cv2.imread(image_path)
    clone = img.copy()

    winW, winH = 64, 128
    step = 16

    for y in range(0, img.shape[0] - winH, step):
        for x in range(0, img.shape[1] - winW, step):
            window = img[y:y+winH, x:x+winW]
            gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)

            features = hog(gray,
                           orientations=9,
                           pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2),
                           block_norm='L2-Hys',
                           visualize=False)

            pred = svm.predict([features])

            if pred == 1:
                cv2.rectangle(clone, (x, y), (x+winW, y+winH), (0,255,0), 2)

    cv2.imshow("Pedestrian Detection", clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    detect_pedestrians("../test_image.jpg")
