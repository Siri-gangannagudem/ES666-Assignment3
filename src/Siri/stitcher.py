          

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
        cumulative_H = np.eye(3)  # Start with the identity matrix for the first image

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
                # H, _ = self.ransac_homography(src_pts, dst_pts)
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
                # panorama_canvas = cv2.add(panorama_canvas, warped_image)
                panorama_canvas = cv2.addWeighted(panorama_canvas, 1, warped_image, 1, 0)

        # Crop out the black borders
        gray = cv2.cvtColor(panorama_canvas, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        final_panorama = panorama_canvas[y:y + h, x:x + w]

        return final_panorama, homography_matrix_list
    


    # def compute_homography(self, src_pts, dst_pts):
    #     # Form the matrix A for solving Ah = 0
    #     A = []
    #     for i in range(len(src_pts)):
    #         x, y = src_pts[i][0]
    #         u, v = dst_pts[i][0]
    #         A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
    #         A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    #     A = np.array(A)
        
    #     # Solve for h using SVD
    #     U, S, Vt = np.linalg.svd(A)
    #     H = Vt[-1].reshape(3, 3)
    #     H /= H[2, 2]  # Normalize to make H[2,2] = 1
        
    #     return H

    # def ransac_homography(self, src_pts, dst_pts, num_iters=500, threshold=5.0):
    #     max_inliers = 0
    #     best_H = None

    #     for _ in range(num_iters):
    #         # Randomly select 4 points
    #         idxs = random.sample(range(len(src_pts)), 4)
    #         src_sample = src_pts[idxs]
    #         dst_sample = dst_pts[idxs]

    #         # Compute homography matrix from the selected points
    #         H = self.compute_homography(src_sample, dst_sample)

    #         # Compute inliers
    #         inliers = []
    #         for i in range(len(src_pts)):
    #             # Transform src point using H
    #             src_point = np.append(src_pts[i][0], 1)
    #             estimated_dst = np.dot(H, src_point)
    #             estimated_dst /= estimated_dst[2]  # Normalize
                
    #             # Calculate the Euclidean distance between the estimated and actual dst point
    #             actual_dst = dst_pts[i][0]
    #             distance = np.linalg.norm(estimated_dst[:2] - actual_dst)

    #             # Count as inlier if the distance is below the threshold
    #             if distance < threshold:
    #                 inliers.append(i)
            
    #         # Check if the current homography has more inliers
    #         if len(inliers) > max_inliers:
    #             max_inliers = len(inliers)
    #             best_H = H

    #     return best_H, max_inliers
