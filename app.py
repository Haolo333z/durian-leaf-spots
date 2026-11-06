import streamlit as st
import numpy as np
import cv2
from PIL import Image

from src.preprocessing import resize_image, preprocess_image
from src.leaf_segmentation import segment_leaf, LeafSegmentationError
from src.spot_detection import detect_spots
from src.severity_analysis import calculate_severity


st.set_page_config(page_title="Durian Leaf Spot Detection", layout="wide")

st.title("🍃 Durian Leaf Spot Detection & Severity Analysis")
st.write("Ung dung nhan dien dom benh tren la sau rieng va phan tich muc do nghiem trong.")

uploaded_file = st.file_uploader("Chon mot anh la sau rieng de phan tich...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Doc file anh thanh mang numpy OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Khong the doc du lieu anh. Vui long kiem tra lai file.")
    else:
        # Tien xu ly
        preprocessed = preprocess_image(image)

        # Tach la voi bat loi kiem tra hop ly
        try:
            leaf_mask = segment_leaf(preprocessed)
            leaf_segmented_success = True
        except LeafSegmentationError as e:
            st.error(f"⚠️ **Loi tach la**: {e}")
            st.info("💡 **Goi y**: Chup anh voi mot la chu dao o trung tam, tranh nen co cay qua ram rap hoac dung anh co do tuong phan ro rang.")
            leaf_segmented_success = False
        except Exception as e:
            st.error(f"Loi khong xac dinh khi tach la: {e}")
            leaf_segmented_success = False

        if leaf_segmented_success:
            # Vung la da tach
            segmented_leaf = cv2.bitwise_and(preprocessed, preprocessed, mask=leaf_mask)

            # Phat hien dom (Nhut)
            spot_mask, raw_contours = detect_spots(segmented_leaf, leaf_mask)
            spot_mask = cv2.bitwise_and(spot_mask, leaf_mask)

            # Tim contour dom da loc
            contours, _ = cv2.findContours(spot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Tinh toan muc do nghiem trong (Phuc)
            results = calculate_severity(leaf_mask, spot_mask)

            # Anh ve duong bao dom
            annotated = preprocessed.copy()
            cv2.drawContours(annotated, contours, -1, (0, 0, 255), 2)

            # Hien thi cac anh ket qua
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.subheader("1. Anh tien xu ly")
                st.image(cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB), use_container_width=True)

            with col2:
                st.subheader("2. Mask tach la")
                st.image(leaf_mask, use_container_width=True)

            with col3:
                st.subheader("3. Mask dom benh")
                st.image(spot_mask, use_container_width=True)

            with col4:
                st.subheader("4. Ket qua phan tich")
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

            st.divider()
            st.subheader("📊 Bang chi so danh gia muc do benh")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Dien tich la (pixels)", f"{results['leaf_area_px']:,}")
            m2.metric("Dien tich dom (pixels)", f"{results['spot_area_px']:,}")
            m3.metric("Ty le dom benh (%)", f"{results['severity_percentage']}%")
            m4.metric("Cap do benh", results["grade"])
