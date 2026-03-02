import cv2
import numpy as np

def recognize_object(object_path, scene_path):

    obj = cv2.imread(object_path, 0)
    scene = cv2.imread(scene_path, 0)

    orb = cv2.ORB_create(2000)

    kp1, des1 = orb.detectAndCompute(obj, None)
    kp2, des2 = orb.detectAndCompute(scene, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = bf.knnMatch(des1, des2, k=2)

    good = []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) > 10:

        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good]
        ).reshape(-1,1,2)

        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good]
        ).reshape(-1,1,2)

        H, _ = cv2.findHomography(
            src_pts, dst_pts, cv2.RANSAC, 5.0
        )

        h, w = obj.shape
        pts = np.float32(
            [[0,0],[0,h],[w,h],[w,0]]
        ).reshape(-1,1,2)

        dst = cv2.perspectiveTransform(pts, H)

        scene_color = cv2.cvtColor(scene,
                                   cv2.COLOR_GRAY2BGR)

        cv2.polylines(scene_color,
                      [np.int32(dst)],
                      True,(0,255,0),3)

        cv2.imshow("Object Detected", scene_color)
        cv2.waitKey(0)

    else:
        print("Object not found")

if __name__ == "__main__":
    recognize_object("../images/object.jpg",
                     "../images/scene.jpg")