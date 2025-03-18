import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from datetime import datetime

class ImageAlignementCheck:
    """
    Class to check and visualize image alignment using SIFT keypoints.
    """
    def __init__(self):
        """
        Initializes the ImageAlignementCheck object.
        """
        pass

    def load_tif_image(self, path):
        """
        Loads a TIF image from the given path.

        Args:
            path (str): The path to the image file.

        Returns:
            numpy.ndarray: The loaded image.
        """
        return cv2.imread(path, cv2.IMREAD_UNCHANGED)
    
    def convert_image_type(self, image):
        """
        Converts the image type to 'uint8' if it's not already.

        Args:
            image (numpy.ndarray): The image to convert.

        Returns:
            numpy.ndarray: The converted 'uint8' image.
        """
        if image.dtype != 'uint8':
            image = cv2.convertScaleAbs(image, alpha=(255.0 / image.max()))
        return image

    def convert_to_gray(self, image):
        """
        Converts a color image to grayscale if it is in color.

        Args:
            image (numpy.ndarray): The image to convert.

        Returns:
            numpy.ndarray: The grayscale image.
        """
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            return image
    
    def preprocess_image(self, image):
        """
        Preprocesses the image: converts to 'uint8' type and grayscale.

        Args:
            image (numpy.ndarray): The image to preprocess.

        Returns:
            numpy.ndarray: The preprocessed image.
        """
        image = self.convert_image_type(image)
        image = self.convert_to_gray(image)
        return image
    
    def detect_keypoints(self, image):
        """
        Detects keypoints and descriptors from the image using the SIFT 
        algorithm.

        Args:
            image (numpy.ndarray): The image to detect keypoints in.

        Returns:
            tuple: A list of keypoints and descriptors.
        """
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)
        return keypoints, descriptors
    
    def match_keypoints(self, descriptors1, descriptors2):
        """
        Finds the matches between descriptors from two images using the 
        BFMatcher.

        Args:
            descriptors1 (numpy.ndarray): Descriptors from the first image.
            descriptors2 (numpy.ndarray): Descriptors from the second image.

        Returns:
            list: A list of matches found between the descriptors.
        """
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)
        return matches
    
    def ratio_test(self, matches, ratio=0.75):
        """
        Applies the ratio test to filter out good matches using Lowe's 
        criterion.

        Args:
            matches (list): List of matches found between two sets of 
            descriptors.

        Returns:
            list: A list of good matches after the ratio test.
        """
        good_matches = []
        for match in matches:
            if len(match) < 2:
                # skip this match if it doesn't have 2 neighbors
                continue
            m, n = match
            if m.distance < ratio * n.distance:
                good_matches.append(m)
        return good_matches
    
    def image_alignement_measure(self, image1, image2, ratio=0.75):
        """
        Measures the number of good matches between two images to evaluate 
        their alignment.

        Args:
            image1 (numpy.ndarray): The first image.
            image2 (numpy.ndarray): The second image.

        Returns:
            int: The number of good matches.
        """
        image1 = self.preprocess_image(image1)
        image2 = self.preprocess_image(image2)
        keypoints1, descriptors1 = self.detect_keypoints(image1)
        keypoints2, descriptors2 = self.detect_keypoints(image2)
        matches = self.match_keypoints(descriptors1, descriptors2)
        good_matches = self.ratio_test(matches, ratio=ratio)
        return len(good_matches)
    
    def image_alignement_check(
            self, image1, image2, threshold=10, atol=0.1, ratio=0.75
        ):
        """
        Checks if two images are correctly aligned using SIFT keypoints and 
        homography.

        If a sufficient number of good matches is found, it calculates the 
        homography and checks if it is close to the identity matrix, indicating
        the images are aligned.

        Args:
            image1 (numpy.ndarray): The first image.
            image2 (numpy.ndarray): The second image.
            threshold (int): The minimum number of good matches required to
            check alignment. Default is 10.
            atol (float): The absolute tolerance for checking if the homography
            is close to the identity matrix. Default is 0.1.
        """
        image1 = self.preprocess_image(image1)
        image2 = self.preprocess_image(image2)
        keypoints1, descriptors1 = self.detect_keypoints(image1)
        keypoints2, descriptors2 = self.detect_keypoints(image2)
        matches = self.match_keypoints(descriptors1, descriptors2)
        good_matches = self.ratio_test(matches, ratio=ratio)

        if len(good_matches) > threshold:
            src_pts = np.float32(
                [keypoints1[m.queryIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [keypoints2[m.trainIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)
    
            # calculate Homography, with RANSAC algorithm, using 5.0 as 
            # threshold. The homography is calculated from the source points
            # to the destination points. The threshold is the maximum
            # distance to consider a point as an inlier, to include it in the
            # homography calculation.
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
            # check if the homography is close to the identity matrix
            if np.allclose(H, np.eye(3), atol=atol):
                #print("Les images sont correctement alignées.")
                return True, H
            else:
                #print("Les images ne sont pas alignées.")
                return False, H
        else:
            #print("Pas assez de correspondances pour vérifier l'alignement.")
            return None
        
    def image_alignement_visualisation(self, image1, image2, ratio=0.75):
        """
        Visualizes the keypoint matches between two images.

        Args:
            image1 (numpy.ndarray): The first image.
            image2 (numpy.ndarray): The second image.
        """
        image1 = self.preprocess_image(image1)
        image2 = self.preprocess_image(image2)
        keypoints1, descriptors1 = self.detect_keypoints(image1)
        keypoints2, descriptors2 = self.detect_keypoints(image2)
        matches = self.match_keypoints(descriptors1, descriptors2)
        good_matches = self.ratio_test(matches, ratio=ratio)

        if len(good_matches) == 0:
            print("Aucune correspondance trouvée pour la visualisation.")
            return
        
        img_matches = cv2.drawMatches(
            image1, keypoints1, image2, keypoints2, good_matches, None
        )
        plt.imshow(img_matches)
        plt.show()

    def calculate_pixel_shift(self, image1, image2, ratio=0.75):
        """
        Calculates the shift in pixels between two images using the good
        matches.
        
        Args:
            image1 (numpy.ndarray): The first image.
            image2 (numpy.ndarray): The second image.

        Returns:
            float: the mean shift in pixels between the corresponding points.
            float: the maximum shift in pixels between the corresponding points
        """
        image1 = self.preprocess_image(image1)
        image2 = self.preprocess_image(image2)
        keypoints1, descriptors1 = self.detect_keypoints(image1)
        keypoints2, descriptors2 = self.detect_keypoints(image2)
        matches = self.match_keypoints(descriptors1, descriptors2)
        good_matches = self.ratio_test(matches, ratio=ratio)

        if len(good_matches) == 0:
            print("Aucune correspondance trouvée.")
            return None, None

        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

        # Calcul des distances entre les points correspondants
        distances = np.linalg.norm(src_pts - dst_pts, axis=1)

        # Moyenne et maximum des distances
        mean_shift = np.mean(distances)
        max_shift = np.max(distances)

        return mean_shift, max_shift


    def analyze_images_in_directory(
        self,
        folder_path,
        threshold=10,
        atol=0.1,
        ratio=0.75
    ):
        """
        Analyzes images in a folder, but only those dated the 1st of each month
        (based on a 'YYYY-MM-DD.tif' naming convention). Checks alignment of each
        image with the previous one in chronological order, and calculates pixel
        shifts.

        Args:
            folder_path (str): Path of the folder containing the images.
            threshold (int): Minimum number of good matches to confirm alignment.
            atol (float): Absolute tolerance for checking if homography is close
                to the identity matrix.
            ratio (float): Ratio used in the Lowe's ratio test for keypoint matching.

        Returns:
            list: Results of alignment checks ("True", "False", or "Error").
            list: Mean pixel shifts between consecutive images.
            list: Max pixel shifts between consecutive images.
        """

        def is_first_of_month(filename):
            """
            Checks if the file name (format 'YYYY-MM-DD.tif') corresponds to the
            1st day of the month.
            """
            base = filename.rsplit('.', 1)[0]
            try:
                dt = datetime.strptime(base, '%Y-%m-%d')
                return dt.day == 1
            except ValueError:
                return False

        # List .tif files whose date is the first of the month
        image_files = [
            f for f in os.listdir(folder_path)
            if f.endswith('.tif') and is_first_of_month(f)
        ]
        image_files.sort()

        results = []
        mean_shifts = []
        max_shifts = []

        if len(image_files) < 2:
            return results, mean_shifts, max_shifts

        previous_image = cv2.imread(
            os.path.join(folder_path, image_files[0]), cv2.IMREAD_UNCHANGED
        )

        for i in range(1, len(image_files)):
            current_path = os.path.join(folder_path, image_files[i])
            current_image = cv2.imread(current_path, cv2.IMREAD_UNCHANGED)

            # Check alignment
            is_aligned = self.image_alignement_check(
                previous_image,
                current_image,
                threshold=threshold,
                atol=atol,
                ratio=ratio
            )
            if is_aligned:
                results.append('True')
            else:
                results.append('False')

            # Calculate pixel shifts
            mean_shift, max_shift = self.calculate_pixel_shift(
                previous_image,
                current_image,
                ratio=ratio
            )
            if mean_shift is None:
                mean_shifts.append('NA')
                max_shifts.append('NA')
            else:
                mean_shifts.append(mean_shift)
                max_shifts.append(max_shift)

            previous_image = current_image

        return results, mean_shifts, max_shifts


def main():
    image_alignement_check = ImageAlignementCheck()
    results_all = []
    mean_shifts_all = []
    max_shifts_all = []

    folders_file = './folders.txt'
    with open(folders_file, 'r') as file:
        folders_list =[line.strip() for line in file.readlines()]
        
    # folders_list_subset = folders_list[:2]

    for folder_path in folders_list: # or folders_list_subset
        results, mean_shifts, max_shifts = image_alignement_check.analyze_images_in_directory(
            folder_path, ratio=0.65
        )
        results_all.append(results)
        mean_shifts_all.append(mean_shifts)
        max_shifts_all.append(max_shifts)

    # Save alignment results
    df = pd.DataFrame(results_all).transpose()

    df.columns = folders_list # or folders_list_subset

    df.to_csv('results_alignement.csv', index=False)

    # Save mean shift values
    df_mean_shifts = pd.DataFrame(mean_shifts_all).transpose()
    df_mean_shifts.columns = folders_list
    df_mean_shifts.to_csv('mean_shifts.csv', index=False)

    # Save max shift values
    df_max_shifts = pd.DataFrame(max_shifts_all).transpose()
    df_max_shifts.columns = folders_list
    df_max_shifts.to_csv('max_shifts.csv', index=False)


if __name__ == '__main__':
    main()