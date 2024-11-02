          

import glob
import cv2
import os
import numpy as np
import random

class PanaromaStitcher():
    def __init__(self):
        pass

    def make_panaroma_for_images_in(self, path):
        imf = path
        all_images = sorted(glob.glob(imf + os.sep + '*'))
        print('Found {} Images for stitching'.format(len(all_images)))

        # Load images and ensure they are valid
        images = [cv2.imread(img) for img in all_images]
        images = [img for img in images if img is not None]  # Filter out any None values
        num_images = len(images)

        if num_images < 2:
            print("Need at least two images to stitch.")
            return None, []

        # Initialize the panorama canvas
        height, width = images[0].shape[:2]
        canvas_width, canvas_height = width * num_images, height * 2
        panorama_canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        # Place the first image in the middle of the canvas
        center_x = canvas_width // 2 - width // 2
        center_y = canvas_height // 2 - height // 2
        panorama_canvas[center_y:center_y + height, center_x:center_x + width] = images[0]

        # Homography matrices list for debugging
        homography_matrix_list = []
        cumulative_H = np.eye(3)  

        # Use the first image as the reference and stitch the rest sequentially
        for i in range(1, num_images):
            # Keypoint detection and matching between consecutive images
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(images[i - 1], None)
            kp2, des2 = sift.detectAndCompute(images[i], None)
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            # Prepare points for homography
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Calculate homography and accumulate transformations
            if len(matches) >= 4:  # Ensure enough matches for homography
                H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                cumulative_H = cumulative_H @ H
                homography_matrix_list.append(cumulative_H)

                # Center the transformation to reduce warping distortion
                shift_matrix = np.array([[1, 0, center_x], [0, 1, center_y], [0, 0, 1]])
                centered_H = shift_matrix @ cumulative_H

                # Warp the next image using the cumulative homography
                warped_image = cv2.warpPerspective(images[i], centered_H, (canvas_width, canvas_height))

                # Fix mask creation: create mask of the warped image
                mask = (warped_image.sum(axis=2) > 0).astype(np.uint8) * 255  # Using sum to create a valid mask
                panorama_canvas = cv2.bitwise_and(panorama_canvas, panorama_canvas, mask=~mask)
                panorama_canvas = cv2.addWeighted(panorama_canvas, 1, warped_image, 1, 0)

        # Crop out the black borders
        gray = cv2.cvtColor(panorama_canvas, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        final_panorama = panorama_canvas[y:y + h, x:x + w]

        return final_panorama, homography_matrix_list
    


   