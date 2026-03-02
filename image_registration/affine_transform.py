import cv2
import numpy as np

def affine_transform(image_path):

    img = cv2.imread(image_path)
    rows, cols = img.shape[:2]

    pts1 = np.float32([[50,50],[200,50],[50,200]])

    pts2 = np.float32([[10,100],[200,50],[100,250]])

    M = cv2.getAffineTransform(pts1, pts2)

    transformed = cv2.warpAffine(img, M, (cols, rows))

    cv2.imshow("Original", img)
    cv2.imshow("Affine Transform", transformed)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    affine_transform("../images/img1.jpg")