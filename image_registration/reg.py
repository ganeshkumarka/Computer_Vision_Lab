import cv2
import numpy as np

def register_images(img1_path, img2_path):

    img1 = cv2.imread(img1_path, 0)
    img2 = cv2.imread(img2_path, 0)

    orb = cv2.ORB_create(5000)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC)

    height, width = img1.shape
    aligned = cv2.warpPerspective(img2, H, (width, height))

    cv2.imshow("Reference Image", img1)
    cv2.imshow("Aligned Image", aligned)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    register_images("../images/img1.jpg",
                    "../images/img2.jpg")