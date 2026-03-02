import cv2
import numpy as np

def compute_sad(block1, block2):
    return np.sum(np.abs(block1 - block2))


def block_matching(img1, img2,
                   block_size=16,
                   search_range=8):

    h, w = img1.shape

    motion_vectors = []

    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):

            block1 = img1[y:y+block_size,
                          x:x+block_size]

            min_sad = float("inf")
            best_dx = 0
            best_dy = 0
            for dy in range(-search_range,
                             search_range+1):

                for dx in range(-search_range,
                                 search_range+1):

                    new_x = x + dx
                    new_y = y + dy

                    if (new_x < 0 or
                        new_y < 0 or
                        new_x+block_size >= w or
                        new_y+block_size >= h):
                        continue

                    block2 = img2[new_y:new_y+block_size,
                                  new_x:new_x+block_size]

                    sad = compute_sad(block1, block2)

                    if sad < min_sad:
                        min_sad = sad
                        best_dx = dx
                        best_dy = dy

            motion_vectors.append(
                (x, y, best_dx, best_dy)
            )

    return motion_vectors