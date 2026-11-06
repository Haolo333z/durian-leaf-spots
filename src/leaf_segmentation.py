"""
Module tach la sau rieng (Leaf Segmentation) phu trach boi Le Huu Hao.
Phuong phap:
- Tu dong phat hien loai nen (nen sang studio/giay trang vs nen vuon/dat/cay xanh phuc tap).
- Su dung GrabCut voi vung khoi tao trung tam (center prior) hoac ROI do nguoi dung chi dinh.
- Lap day lo trong (lo hoai tu do dom benh, van la, phan quang) bang cv2.drawContours FILLED.
- Ap dung he thong kiem tra hop ly (sanity checks) ve dien tich va cham vien de loai bo mask sai.
"""

import cv2
import numpy as np


class LeafSegmentationError(ValueError):
    """Ngoai le nem ra khi qua trinh tach la khong the xac dinh mask hop le."""
    pass


def is_plain_light_background(image_bgr: np.ndarray, border_thickness: int = 15) -> bool:
    """
    Kiem tra xem anh co nen sang deu (nen giay trang, nen phong thi nghiem) hay khong.
    """
    h, w = image_bgr.shape[:2]
    bt = min(border_thickness, max(2, h // 20), max(2, w // 20))
    borders = np.concatenate([
        image_bgr[:bt, :].reshape(-1, 3),
        image_bgr[-bt:, :].reshape(-1, 3),
        image_bgr[:, :bt].reshape(-1, 3),
        image_bgr[:, -bt:].reshape(-1, 3),
    ])
    hsv_borders = cv2.cvtColor(borders.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    median_v = np.median(hsv_borders[:, 2])
    median_s = np.median(hsv_borders[:, 1])
    return (median_v > 175) and (median_s < 45)


def validate_leaf_mask(leaf_mask: np.ndarray, min_ratio: float = 0.05, max_ratio: float = 0.85, max_border_touch: float = 0.15) -> None:
    """
    Kiem tra tinh hop ly cua mask la sau rieng:
    1. Dien tich khong duoc rong hoac qua nho (< 5% dien tich anh).
    2. Dien tich khong duoc tran ngap toan bo anh (> 85% dien tich anh, tranh nham lan voi nen).
    3. Mask khong duoc cham vien qua nhieu (> 15% tong chieu dai vien hoac cham tu 3 canh tro len).
    """
    h, w = leaf_mask.shape[:2]
    total_area = h * w
    leaf_area = int(cv2.countNonZero(leaf_mask))
    area_ratio = leaf_area / float(total_area)

    if area_ratio < min_ratio:
        raise LeafSegmentationError(
            f"Vung la tach duoc qua nho ({area_ratio * 100:.2f}% < {min_ratio * 100:.1f}% dien tich anh). "
            f"Anh co the khong chua doi tuong la ro rang hoac bi bo mat."
        )

    if area_ratio > max_ratio:
        raise LeafSegmentationError(
            f"Vung la tach duoc qua lon ({area_ratio * 100:.2f}% > {max_ratio * 100:.1f}% dien tich anh). "
            f"Thuat toan nghi ngo da chon nham toan bo nen lam la."
        )

    # Kiem tra tiep xuc vien 4 phia
    top_touch = np.mean(leaf_mask[0, :] == 255)
    bottom_touch = np.mean(leaf_mask[-1, :] == 255)
    left_touch = np.mean(leaf_mask[:, 0] == 255)
    right_touch = np.mean(leaf_mask[:, -1] == 255)

    total_border_px = 2 * h + 2 * w
    touch_px = (
        np.sum(leaf_mask[0, :] == 255) +
        np.sum(leaf_mask[-1, :] == 255) +
        np.sum(leaf_mask[:, 0] == 255) +
        np.sum(leaf_mask[:, -1] == 255)
    )
    border_touch_ratio = touch_px / float(total_border_px)

    sides_touched = sum([
        top_touch > 0.08,
        bottom_touch > 0.08,
        left_touch > 0.08,
        right_touch > 0.08
    ])

    if border_touch_ratio > max_border_touch or sides_touched >= 3:
        raise LeafSegmentationError(
            f"Mask la tiep xuc vieng bat thuong ({border_touch_ratio * 100:.1f}% vieng, {sides_touched}/4 canh). "
            f"Dau hieu ro ret cua viec chon vung nen vuon/cay xung quanh thay vi la chinh."
        )


def segment_leaf(image_bgr: np.ndarray, roi: tuple = None, return_segmented: bool = False):
    """
    Tach la sau rieng khoi hau canh mot cach manh me va dang tin cay.
    
    Pham vi ho tro tu dong:
    - Anh chup chua 1 la sau rieng la doi tuong chu dao o khu vuc trung tam.
    - Ho tro ca nen sang deu (phong thi nghiem) va nen vuon, dat, bong do nhe.
    
    Truong hop can thao tac bo sung (ROI):
    - Anh co nhieu la chong cheo phuc tap hoac nen vuon qua ram rap. Nguoi dung truyen tham so roi=(x, y, w, h).
    
    Args:
        image_bgr (np.ndarray): Anh BGR uint8 dau vao.
        roi (tuple, optional): (x, y, w, h) vung chua la neu muon chi dinh thu cong.
        return_segmented (bool): Neu True, tra ve tuple (leaf_mask, segmented_leaf).
        
    Returns:
        np.ndarray hoac tuple:
            leaf_mask: anh nhi phan uint8 cung kich thuoc (la=255, nen=0).
            segmented_leaf: anh BGR chi giu vung la, nen bi xoa den (neu return_segmented=True).
            
    Raises:
        ValueError: Khi anh dau vao rong hoac khong hop le.
        LeafSegmentationError: Khi khong the tach duoc mask la hop ly.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Anh dau vao tach la khong hop le hoac rong")

    h, w = image_bgr.shape[:2]

    # 1. Kiem tra xem nen co phai la nen phong sang deu hay khong
    if roi is None and is_plain_light_background(image_bgr):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, raw_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        # Nen vuon / dat / cay co / bong do: Dung GrabCut tren anh thu nho de tang toc va do on dinh
        max_proxy_dim = 600
        scale = min(1.0, max_proxy_dim / float(max(h, w)))
        proxy_w = max(10, int(round(w * scale)))
        proxy_h = max(10, int(round(h * scale)))
        proxy_img = cv2.resize(image_bgr, (proxy_w, proxy_h), interpolation=cv2.INTER_AREA)

        if roi is not None:
            rx, ry, rw, rh = roi
            rect = (
                max(0, int(rx * scale)),
                max(0, int(ry * scale)),
                min(proxy_w, int(rw * scale)),
                min(proxy_h, int(rh * scale))
            )
        else:
            # Mac dinh can bien 4% de loai bo vien ngoai
            mx = max(3, int(proxy_w * 0.04))
            my = max(3, int(proxy_h * 0.04))
            rect = (mx, my, proxy_w - 2 * mx, proxy_h - 2 * my)

        gc_mask = np.zeros((proxy_h, proxy_w), dtype=np.uint8)
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)

        try:
            cv2.grabCut(proxy_img, gc_mask, rect, bgd_model, fgd_model, iterCount=4, mode=cv2.GC_INIT_WITH_RECT)
        except cv2.error as e:
            raise LeafSegmentationError(f"Loi thuat toan GrabCut: {e}")

        proxy_bin = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        raw_mask = cv2.resize(proxy_bin, (w, h), interpolation=cv2.INTER_NEAREST)

    # 2. Xu ly hinh thai hoc de lam muot bien va noi cac manh nho
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # 3. Tim duong bao lon nhat va lap day tat ca cac lo trong (cac dom benh toi, van la)
    contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise LeafSegmentationError("Khong tim thay bat ky duong bao doi tuong nao trong anh")

    largest_contour = max(contours, key=cv2.contourArea)
    leaf_mask = np.zeros((h, w), dtype=np.uint8)
    # FILLED dam bao toan bo dien tich ben trong vien la deu duoc giu, khong bi tao lo do dom benh toi mau
    cv2.drawContours(leaf_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # 4. Kiem tra hop ly (Sanity check)
    validate_leaf_mask(leaf_mask)

    if return_segmented:
        segmented_leaf = cv2.bitwise_and(image_bgr, image_bgr, mask=leaf_mask)
        return leaf_mask, segmented_leaf

    return leaf_mask
