import cv2
import numpy as np


def calculate_severity(leaf_mask: np.ndarray, spot_mask: np.ndarray) -> dict:
    """
    Calculate severity of leaf spot infection based on area ratios.
    Returns:
        dict: Analysis metrics including leaf area, spot area, percentage, and severity grade.
    """
    leaf_area = int(cv2.countNonZero(leaf_mask))
    spot_area = int(cv2.countNonZero(spot_mask))

    if leaf_area == 0:
        severity_percentage = 0.0
    else:
        severity_percentage = (spot_area / leaf_area) * 100.0

    # Severity level grading
    if severity_percentage == 0:
        grade = "Healthy (Normal)"
    elif severity_percentage < 5.0:
        grade = "Low / Early Stage"
    elif severity_percentage < 20.0:
        grade = "Moderate Stage"
    else:
        grade = "Severe Stage"

    return {
        "leaf_area_px": leaf_area,
        "spot_area_px": spot_area,
        "severity_percentage": round(severity_percentage, 2),
        "grade": grade,
    }
