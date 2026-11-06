import os
import sys
import argparse
import cv2

from src.preprocessing import load_image, preprocess_image
from src.leaf_segmentation import segment_leaf, LeafSegmentationError
from src.spot_detection import detect_spots
from src.severity_analysis import calculate_severity


def process_image(image_path: str, output_dir: str = "results", roi: tuple = None):
    """
    Quy trinh tich hop phan tich dom la sau rieng:
    1. Doc anh an toan (load_image).
    2. Tien xu ly giu ty le va cai thien tuong phan (preprocess_image).
    3. Tach la voi kiem tra hop ly (segment_leaf). Dung neu tach la that bai.
    4. Phat hien dom benh ben trong vung la (detect_spots).
    5. Tinh toan ty le va cap do benh (calculate_severity).
    6. Luu anh analyzed, leaf_mask, spot_mask cung he toa do.
    """
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.basename(image_path)
    base_name, ext = os.path.splitext(filename)

    # 1. Doc anh
    try:
        raw_image = load_image(image_path)
    except Exception as e:
        print(f"[LOI DOC ANH]: {e}", file=sys.stderr)
        return None

    # 2. Tien xu ly (resize giu ty le, giam nhieu nhe, can bang sang LAB/CLAHE)
    preprocessed = preprocess_image(raw_image)

    # 3. Tach la voi kiem tra hop ly
    try:
        leaf_mask = segment_leaf(preprocessed, roi=roi)
    except LeafSegmentationError as e:
        print(f"--- THAT BAI: {filename} ---", file=sys.stderr)
        print(f"[LOI TACH LA]: Khong the tach la hop le tu anh '{image_path}'.", file=sys.stderr)
        print(f"Chi tiet ly do: {e}", file=sys.stderr)
        print("Dung quy trinh phan tich de tranh tinh toan sai lech ty le dom gia.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[LOI HE THONG KHI TACH LA]: {e}", file=sys.stderr)
        return None

    # Vung la da tach
    segmented_leaf = cv2.bitwise_and(preprocessed, preprocessed, mask=leaf_mask)

    # 4. Phat hien dom theo giao dien ham cua Nhut
    spot_mask, raw_contours = detect_spots(segmented_leaf, leaf_mask)
    # Dam bao dom tuyet doi nam trong vung la hop le
    spot_mask = cv2.bitwise_and(spot_mask, leaf_mask)

    # Tao lai duong vien tu mask dom da loc de dam bao dong bo toa do
    contours, _ = cv2.findContours(spot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5. Tinh muc do theo giao dien ham cua Phuc
    results = calculate_severity(leaf_mask, spot_mask)

    # 6. Ve duong bao dom tren anh preprocessed (cung kich thuoc va toa do)
    annotated = preprocessed.copy()
    cv2.drawContours(annotated, contours, -1, (0, 0, 255), 2)

    # 7. Luu cac anh ket qua
    out_ext = ext if ext.lower() in [".png", ".jpg", ".jpeg"] else ".png"
    annotated_path = os.path.join(output_dir, f"{base_name}_analyzed{out_ext}")
    leaf_mask_path = os.path.join(output_dir, f"{base_name}_leaf_mask.png")
    spot_mask_path = os.path.join(output_dir, f"{base_name}_spot_mask.png")

    cv2.imwrite(annotated_path, annotated)
    cv2.imwrite(leaf_mask_path, leaf_mask)
    cv2.imwrite(spot_mask_path, spot_mask)

    print(f"--- Analysis Results for: {filename} ---")
    print(f"Leaf Area (px)       : {results['leaf_area_px']}")
    print(f"Spot Area (px)       : {results['spot_area_px']}")
    print(f"Severity Percentage  : {results['severity_percentage']}%")
    print(f"Severity Grade       : {results['grade']}")
    print(f"Saved annotated image to: {annotated_path}")
    print(f"Saved leaf mask to: {leaf_mask_path}")
    print(f"Saved spot mask to: {spot_mask_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Durian Leaf Spot Detection and Severity Analysis"
    )
    parser.add_argument(
        "image",
        nargs="?",
        help="Duong dan anh dau vao (vi du: data/normal/1.png)",
    )
    parser.add_argument(
        "--image",
        dest="image_option",
        help="Duong dan anh dau vao (cach goi bang flag)",
    )
    parser.add_argument(
        "--output",
        default="results",
        help="Thu muc luu ket qua (mac dinh: results)",
    )
    parser.add_argument(
        "--roi",
        nargs=4,
        type=int,
        metavar=("X", "Y", "W", "H"),
        help="Vung chu nhat chua la (x y w h) dung cho truong hop nen phuc tap",
    )
    args = parser.parse_args()

    image_path = args.image_option or args.image
    if not image_path:
        parser.error("Hay cung cap duong dan anh can phan tich")

    roi = tuple(args.roi) if args.roi else None
    res = process_image(image_path, args.output, roi=roi)
    if res is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
