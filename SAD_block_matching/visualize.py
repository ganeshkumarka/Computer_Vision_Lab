import cv2
from sad_matching import block_matching

def visualize_motion(frame1_path, frame2_path):

    img1 = cv2.imread(frame1_path, 0)
    img2 = cv2.imread(frame2_path, 0)

    motion_vectors = block_matching(img1, img2)

    output = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

    for x, y, dx, dy in motion_vectors:

        start_point = (x+8, y+8)
        end_point = (x+8+dx, y+8+dy)

        cv2.arrowedLine(output,
                        start_point,
                        end_point,
                        (0,255,0),
                        1,
                        tipLength=0.3)

    cv2.imshow("Motion Estimation (SAD)", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    visualize_motion("../images/frame1.jpg",
                     "../images/frame2.jpg")