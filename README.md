# Durian Leaf Spot Detection and Severity Analysis
## Hệ Thống Phát Hiện Đốm Lá Sầu Riêng và Phân Tích Mức Độ Bệnh

Hệ thống thị giác máy tính phát hiện đốm bất thường trên lá sầu riêng và định lượng diện tích tổn thương (Severity Analysis) sử dụng Python và OpenCV.

---

## 👥 Phân Công Nhiệm Vụ Trong Nhóm
- **Lê Hữu Hào:** Chuẩn bị dữ liệu, Tiền xử lý (`src/preprocessing.py`), Phân đoạn tách phiến lá (`src/leaf_segmentation.py`), Tích hợp CLI (`main.py`), Kiểm thử kỹ thuật và Biên soạn Chương 1–2.
- **Trần Minh Nhựt:** Thuật toán phát hiện đốm bất thường (`src/spot_detection.py`).
- **Đặng Hoàng Phúc:** Thuật toán phân cấp mức độ nghiêm trọng (`src/severity_analysis.py`) và Giao diện ứng dụng (`app.py`).

---

## 📁 Cấu Trúc Thư Mục Dự Án (Project Structure)

```text
durian-leaf-spots/
├── data/
│   ├── normal/               # Ảnh lá sầu riêng bình thường (1.png - 5.png, ảnh tổng hợp thử nghiệm)
│   └── spotted/              # Ảnh lá sầu riêng có đốm bệnh (hu1.png - hu5.png, ảnh tổng hợp thử nghiệm)
├── docs/
│   ├── BAN_GIAO_HAO.md       # Biên bản bàn giao kỹ thuật của Hào cho Nhựt và Phúc
│   ├── CHUONG_1_2_HAO.md     # Nội dung báo cáo lý thuyết & phương pháp Chương 1 và 2
│   ├── danh_sach_du_lieu.csv # Bảng kê danh mục dữ liệu, nguồn gốc và mức độ khó
│   └── kiem_thu_hao.csv      # Bảng kết quả kiểm thử thực nghiệm trên 10 ảnh mẫu
├── results/                  # Thư mục lưu trữ kết quả phân tích (analyzed, leaf_mask, spot_mask)
├── src/
│   ├── __init__.py           # Khởi tạo package
│   ├── preprocessing.py      # Tiền xử lý (đọc an toàn, resize giữ tỷ lệ, GaussianBlur 3x3, CLAHE)
│   ├── leaf_segmentation.py  # Phân đoạn tách lá (GrabCut + Otsu, lấp đầy lỗ đốm, Sanity Checks)
│   ├── spot_detection.py     # Phát hiện đốm bệnh trên phiến lá (Nhựt)
│   └── severity_analysis.py  # Tính tỷ lệ diện tích đốm/lá và xếp loại cấp độ bệnh (Phúc)
├── app.py                    # Ứng dụng giao diện Web trực quan (Streamlit)
├── main.py                   # Script chạy kiểm tra phân tích bằng dòng lệnh (CLI)
├── .gitignore                # Cấu hình loại trừ file tạm và môi trường ảo
└── README.md                 # Tài liệu hướng dẫn sử dụng dự án
```

---

## ⚙️ Cài Đặt Trên Windows (Installation)

### 1. Yêu cầu môi trường:
- Hệ điều hành: Windows 10/11
- Python 3.10 trở lên (khuyến nghị tạo môi trường ảo `.venv`)

### 2. Kích hoạt môi trường và cài đặt thư viện:
```powershell
# Tạo môi trường ảo (nếu chưa có)
python -m venv .venv

# Kích hoạt môi trường ảo trên PowerShell
.\.venv\Scripts\Activate.ps1

# Cài đặt các thư viện cần thiết
pip install opencv-python numpy streamlit pillow
```

---

## 🚀 Cách Chạy Chương Trình (Usage)

### 1. Phân tích qua dòng lệnh (CLI):
```powershell
# Chạy với tham số vị trí
.\.venv\Scripts\python.exe main.py data\normal\1.png
.\.venv\Scripts\python.exe main.py data\spotted\hu1.png

# Chạy với cờ --image và thư mục lưu tùy chọn
.\.venv\Scripts\python.exe main.py --image data\spotted\hu2.png --output results

# Chạy với vùng ROI có kiểm soát (x y w h) cho ảnh nền phức tạp
.\.venv\Scripts\python.exe main.py data\spotted\hu1.png --roi 100 100 1000 1000
```

### 2. Chạy ứng dụng giao diện web (Streamlit):
```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```
Sau khi khởi chạy, trình duyệt web sẽ mở tại địa chỉ `http://localhost:8501`. Người dùng có thể kéo thả ảnh lá sầu riêng để xem kết quả phân tích trực quan 4 bước.

---

## 🔍 Cách Xem Kết Quả (Outputs)

Với mỗi ảnh đầu vào `<name>.<ext>`, hệ thống tự động xuất và lưu 3 ảnh vào thư mục `results/`:
1. `<name>_analyzed.<ext>`: Ảnh tiền xử lý được vẽ đường viền đỏ (`cv2.drawContours`) bao quanh các vết đốm phát hiện được.
2. `<name>_leaf_mask.png`: Mask nhị phân của phiến lá (`uint8`), trong đó vùng lá là màu trắng (`255`), hậu cảnh là màu đen (`0`).
3. `<name>_spot_mask.png`: Mask nhị phân các vết đốm bệnh (`uint8`), đảm bảo 100% nằm gọn bên trong vùng phiến lá hợp lệ.

Bảng kết quả in ra terminal bao gồm:
- **Leaf Area (px):** Tổng số pixel của phiến lá.
- **Spot Area (px):** Tổng số pixel của các vết đốm bệnh.
- **Severity Percentage:** Tỷ lệ phần trăm diện tích nhiễm bệnh: $\frac{\text{Spot Area}}{\text{Leaf Area}} \times 100\%$.
- **Severity Grade:** Xếp loại mức độ (Healthy, Low, Moderate, Severe).

---

## 📸 Điều Kiện Ảnh Đầu Vào & Giới Hạn Kỹ Thuật

### 1. Điều kiện ảnh được hỗ trợ tự động:
- Ảnh chụp có **một lá sầu riêng là đối tượng chính** nằm ở khu vực trung tâm khung hình.
- Nền chụp: Hỗ trợ nền sáng phòng thí nghiệm (giấy trắng, bàn trơn) hoặc nền vườn tự nhiên (đất, tán cây có độ sâu trường ảnh vừa phải).
- Ánh sáng: Đủ sáng, không bị cháy sáng toàn bộ hoặc tối mịt.

### 2. Các lỗi thường gặp và cách xử lý:
- **Lỗi `LeafSegmentationError: Vùng lá quá lớn (> 85%)`:** Xảy ra khi ảnh chụp góc quá cận sát mép hoặc thuật toán bị bắt nhầm nền xung quanh.  
  *Khắc phục:* Lùi khoảng cách chụp để phiến lá chiếm khoảng 30% - 60% khung hình hoặc dùng tham số `--roi x y w h`.
- **Lỗi `LeafSegmentationError: Mask lá chạm mép bất thường`:** Xảy ra khi tán cây nền ngoài dính chặt vào mép ảnh.  
  *Khắc phục:* Chụp ảnh với hậu cảnh thoáng hơn hoặc đặt lá lên bề mặt đơn sắc.
- **Lỗi nhầm lẫn màu mặt dưới lá:** Mặt dưới lá sầu riêng có màu vàng nâu đồng tự nhiên; cần lưu ý module phát hiện đốm có thể nhầm lẫn vùng này nếu không được điều chỉnh thích ứng.

### 3. Giới hạn hệ thống:
- Hệ thống hiện tại **chưa hỗ trợ tự động tách rời từng lá trong chùm nhiều lá đan xen chồng lấn**.
- Bộ dữ liệu mẫu hiện tại gồm 10 ảnh là **ảnh tổng hợp (AI/Synthetic)** dùng để phát triển và kiểm thử thuật toán; nhóm cần bổ sung tối thiểu 10 ảnh chụp thật tế ngoài vườn để hoàn tất nghiệm thu thực tế.
