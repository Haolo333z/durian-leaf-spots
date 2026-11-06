import os
import cv2
import numpy as np


def load_image(image_path: str) -> np.ndarray:
    """
    Doc anh tu duong dan tren dia mot cach an toan, ho tro ca duong dan tieng Viet tren Windows.
    
    Args:
        image_path (str): Duong dan toi file anh.
        
    Returns:
        np.ndarray: Anh o he mau BGR (uint8).
        
    Raises:
        FileNotFoundError: Neu file khong ton tai.
        ValueError: Neu file bi loi hoac khong phai dinh dang anh hop le.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Khong tim thay file anh tai duong dan: {image_path}")

    # Su dung np.fromfile ket hop cv2.imdecode de tranh loi ky tu unicode/tieng Viet tren Windows
    try:
        file_bytes = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    except Exception as exc:
        raise ValueError(f"Loi khi doc file anh '{image_path}': {exc}") from exc

    if image is None or image.size == 0:
        raise ValueError(f"Khong the giai ma du lieu anh tu: {image_path}")

    return image


def resize_image(image: np.ndarray, max_side: int = 1200, target_size: tuple = None) -> np.ndarray:
    """
    Doi kich thuoc anh nhung luon bao toan ty le goc (khong lam meo anh).
    
    Args:
        image (np.ndarray): Anh dau vao BGR.
        max_side (int): Chieu dai toi da cho canh dai nhat (mac dinh 1200px).
        target_size (tuple, optional): (width, height) neu can fix cung kich thuoc.
        
    Returns:
        np.ndarray: Anh sau khi resize.
    """
    if image is None or image.size == 0:
        raise ValueError("Anh dau vao de resize khong hop le hoac rong")

    h, w = image.shape[:2]

    if target_size is not None:
        tw, th = target_size
        return cv2.resize(image, (int(tw), int(th)), interpolation=cv2.INTER_AREA)

    if max(h, w) <= max_side:
        return image.copy()

    scale = max_side / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def preprocess_image(image_bgr: np.ndarray, max_side: int = 1200) -> np.ndarray:
    """
    Tien xu ly anh:
    1. Resize giu nguyen ty le (aspect ratio) voi chieu dai canh toi da max_side.
    2. Giam nhiễu bang bo loc Gaussian nhe (3x3), khong lam nhoe hoac mat cac dom benh nho.
    3. Can bang do tuong phan thich ung CLAHE tren kenh L cua he mau LAB.
    
    Args:
        image_bgr (np.ndarray): Anh BGR dau vao.
        max_side (int): Canh lon nhat de resize.
        
    Returns:
        np.ndarray: Anh da duoc tien xu ly (BGR uint8).
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Anh dau vao tien xu ly khong hop le")

    # 1. Resize giu ty le
    resized = resize_image(image_bgr, max_side=max_side)

    # 2. Giam nhieu nhe: dung GaussianBlur (3, 3) de giu nguyen chi tiet dom nho
    denoised = cv2.GaussianBlur(resized, (3, 3), 0)

    # 3. Cai thien tuong phan thich ung cuc bo (CLAHE) trong he mau LAB
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    preprocessed_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return preprocessed_bgr
