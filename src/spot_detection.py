import cv2
import numpy as np


def detect_spots(segmented_leaf: np.ndarray, leaf_mask: np.ndarray) -> tuple[np.ndarray, list]:
    """
    Detect disease spots/lesions on the segmented leaf.
    Returns:
        spot_mask (np.ndarray): Binary mask where spot pixels are 255.
        contours (list): List of detected spot contours.
    """
    # Convert segmented leaf to HSV or LAB
    hsv = cv2.cvtColor(segmented_leaf, cv2.COLOR_BGR2HSV)

    # Brown / yellow / necrosis color ranges (spots)
    lower_spot = np.array([10, 50, 50], dtype=np.uint8)
    upper_spot = np.array([30, 255, 200], dtype=np.uint8)
    spot_mask = cv2.inRange(hsv, lower_spot, upper_spot)

    # Ensure spots are strictly inside the leaf region
    spot_mask = cv2.bitwise_and(spot_mask, leaf_mask)

    # Clean spot mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    spot_mask = cv2.morphologyEx(spot_mask, cv2.MORPH_OPEN, kernel)

    # Find contours of detected spots
    contours, _ = cv2.findContours(spot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return spot_mask, contours
