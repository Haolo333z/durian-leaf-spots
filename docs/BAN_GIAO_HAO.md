# BIÊN BẢN BÀN GIAO KỸ THUẬT - PHẦN LÊ HỮU HÀO
**Dự án:** Phát hiện đốm bất thường trên lá sầu riêng bằng kỹ thuật xử lý ảnh  
**Người bàn giao:** Lê Hữu Hào (Phụ trách Tiền xử lý, Tách phiến lá, Chuẩn bị dữ liệu, Tích hợp & Chương 1-2)  
**Người nhận bàn giao:** 
1. **Trần Minh Nhựt** (Phụ trách Phát hiện đốm bất thường - `src/spot_detection.py`)
2. **Đặng Hoàng Phúc** (Phụ trách Phân tích mức độ nghiêm trọng & Giao diện - `src/severity_analysis.py`, `app.py`)

---

## 1. DANH SÁCH CÁC TỆP ĐÃ HOÀN THIỆN VÀ BÀN GIAO

| Tên tệp | Vai trò | Người chịu trách nhiệm chính | Trạng thái |
| :--- | :--- | :--- | :--- |
| `src/preprocessing.py` | Đọc ảnh, đổi kích thước giữ tỷ lệ, khử nhiễu Gaussian nhẹ, cân bằng sáng CLAHE | Lê Hữu Hào | **ĐÃ HOÀN THÀNH** |
| `src/leaf_segmentation.py` | Tách phiến lá GrabCut + Otsu, lấp đầy lỗ đốm bệnh, kiểm tra hợp lý (Sanity Checks) | Lê Hữu Hào | **ĐÃ HOÀN THÀNH** |
| `main.py` | Script tích hợp dòng lệnh CLI 4 bước, chuẩn hóa lưu trữ và bắt lỗi | Lê Hữu Hào | **ĐÃ HOÀN THÀNH** |
| `app.py` | Giao diện Streamlit đã tích hợp đồng bộ với module tiền xử lý và tách lá | Phúc (Hào hỗ trợ tích hợp) | **ĐÃ ĐỒNG BỘ** |
| `docs/danh_sach_du_lieu.csv` | Bảng kê 10 ảnh tổng hợp hiện có, chất lượng, mức độ khó | Lê Hữu Hào | **ĐÃ HOÀN THÀNH** |
| `docs/kiem_thu_hao.csv` | Kết quả kiểm thử thực nghiệm trên 10 ảnh mẫu | Lê Hữu Hào | **ĐÃ HOÀN THÀNH** |
| `docs/CHUONG_1_2_HAO.md` | Báo cáo chi tiết lý thuyết, phương pháp và tham số Chương 1 & 2 | Lê Hữu Hào | **ĐÃ HOÀN THÀNH** |
| `README.md` | Tài liệu hướng dẫn cài đặt, cấu trúc và cách chạy | Lê Hữu Hào | **ĐÃ CẬP NHẬT** |
| `.gitignore` | Loại trừ môi trường ảo `.venv` và cache `__pycache__` | Lê Hữu Hào | **ĐÃ CẬP NHẬT** |

---

## 2. CHỮ KÝ HÀM VÀ QUY CÁCH DỮ LIỆU ĐẦU RA BÀN GIAO

### 2.1. Bàn giao cho Nhựt (`src/spot_detection.py`)
Hào cung cấp đầu vào chuẩn cho hàm phát hiện đốm của Nhựt:
```python
spot_mask, contours = detect_spots(segmented_leaf: np.ndarray, leaf_mask: np.ndarray)
```
- `segmented_leaf`: Mảng `np.ndarray` hệ màu BGR, kiểu `uint8`, kích thước `(H, W, 3)`. Vùng ngoài lá đã được gán giá trị 0 (màu đen). Toàn bộ vùng lá và đốm bệnh bên trong được giữ nguyên bản màu sắc và độ nét.
- `leaf_mask`: Mảng `np.ndarray` nhị phân, kiểu `uint8`, kích thước `(H, W)`. Vùng lá có giá trị `255`, vùng nền có giá trị `0`. Tất cả các lỗ thủng do đốm tối bên trong lá đã được lấp đầy hoàn toàn bằng cờ `cv2.FILLED`.

### 2.2. Bàn giao cho Phúc (`src/severity_analysis.py` và `app.py`)
Hào cung cấp mask lá và tích hợp đồng bộ dữ liệu vào pipeline tính toán:
```python
results = calculate_severity(leaf_mask: np.ndarray, spot_mask: np.ndarray) -> dict
```
- `leaf_mask`: Nhị phân `uint8` `(H, W)` với giá trị `255` tại vùng lá.
- `spot_mask`: Nhị phân `uint8` `(H, W)` với giá trị `255` tại các vết đốm bệnh. Hệ thống tích hợp đã gọi `spot_mask = cv2.bitwise_and(spot_mask, leaf_mask)` để bảo đảm tuyệt đối không có đốm rác ngoài vùng lá.
- Kết quả mong đợi từ hàm của Phúc là một từ điển chứa:
  - `leaf_area_px` (int): Diện tích lá tính bằng số pixel.
  - `spot_area_px` (int): Diện tích vết đốm tính bằng số pixel.
  - `severity_percentage` (float): Tỷ lệ % diện tích nhiễm bệnh.
  - `grade` (str): Cấp độ nhiễm bệnh tương ứng.

---

## 3. HƯỚNG DẪN CÁC LỆNH CHẠY KIỂM TRA

### 3.1. Chạy kiểm tra qua dòng lệnh (CLI):
```bash
# Kiểm tra ảnh normal 1.png (trước đây bị lỗi chọn nền 100%, nay đã hết)
.\.venv\Scripts\python.exe main.py data\normal\1.png

# Kiểm tra ảnh spotted hu1.png (trước đây bị lỗi nền trắng lá đen, nay đã hết)
.\.venv\Scripts\python.exe main.py data\spotted\hu1.png

# Kiểm tra lưu vào thư mục tùy chọn
.\.venv\Scripts\python.exe main.py --image data\spotted\hu2.png --output results
```

### 3.2. Khởi chạy giao diện ứng dụng Web (Streamlit):
```bash
.\.venv\Scripts\python.exe -m streamlit run app.py
```

---

## 4. KẾT QUẢ KIỂM THỬ VÀ SO SÁNH TRƯỚC - SAU KHI SỬA

| Tên ảnh | Kết quả phiên bản cũ | Kết quả phiên bản mới (Hào đã sửa) | Đánh giá trực quan |
| :--- | :--- | :--- | :--- |
| `normal/1.png` | `leaf_mask` trắng 100% (483.025 px), tỷ lệ đốm giả 1.2% | `leaf_mask` chuẩn 36.6% (176.800 px), đốm 0 px (0.0%) | **ĐẠT**: Loại bỏ hoàn toàn nền xanh tối bao quanh, không còn nhận nền làm lá |
| `spotted/hu1.png` | `leaf_mask` trắng ở nền (84.2%), đen ở lá, tỷ lệ đốm giả 14.24% | `leaf_mask` chuẩn 26.2% (377.270 px), đốm cháy 289.902 px (76.84%) | **ĐẠT**: Khắc phục dứt điểm lỗi nghịch đảo mask, đốm hoại tử lớn được giữ trọn vẹn |
| 8 ảnh còn lại | Đa số bị bám biên hoặc chọn nhầm cành lá xung quanh | 100% không còn chạm biên bất thường, tỷ lệ diện tích lá nằm trong khoảng 21% - 44% | **ĐẠT**: Toàn bộ vượt qua kiểm tra tính hợp lý (Sanity Checks) |

---

## 5. BÀN GIAO CÁC LỖI VÀ TỒN ĐỌNG CHO NHỰT VÀ PHÚC

### 5.1. Bàn giao cho Nhựt (`src/spot_detection.py`):
1. **Lỗi nhận diện nhầm mặt dưới lá vàng/nâu đồng thành đốm bệnh:**
   - Trên ảnh `data/normal/3.png`, lá sầu riêng khỏe mạnh nhưng có mặt dưới màu vàng đồng đặc trưng. Ngưỡng HSV hiện tại của Nhựt `[10, 50, 50] - [30, 255, 200]` đã gom toàn bộ mặt lá thành đốm bệnh, cho ra tỷ lệ $79.94\%$ đốm (xếp loại Severe Stage sai lệch).
   - Trên ảnh `data/normal/2.png`, gân và cuống lá màu hơi vàng cũng bị nhận nhầm thành đốm bệnh $33.87\%$.
   - **Khuyến nghị cho Nhựt:** Cần nâng cấp thuật toán phát hiện đốm:
     - Sử dụng phân tích gradient độ sáng cục bộ hoặc thuật toán phân đoạn theo không gian màu $CIE\ L^*a^*b^*$ (kênh $a^*$ phản ánh sắc tố xanh lá - đỏ, vết hoại tử sẽ lệch hẳn sang $a^* > 0$).
     - Áp dụng trừ nền màu nền trung bình của phiến lá (local adaptive thresholding) thay vì dùng dải màu HSV toàn cục cố định.

### 5.2. Bàn giao cho Phúc (`src/severity_analysis.py` và `app.py`):
1. **Hoàn thiện thang phân loại mức độ bệnh:**
   - Hiện tại hàm của Phúc chia 4 cấp độ: Healthy (0%), Low (<5%), Moderate (<20%), Severe (>=20%).
   - **Khuyến nghị cho Phúc:** Xem xét đối chiếu với các quy chuẩn bảo vệ thực vật hoặc tài liệu nông nghiệp chuyên ngành để xây dựng thang điểm 5 cấp (ví dụ: Cấp 0: 0%, Cấp 1: 1-5%, Cấp 2: 6-15%, Cấp 3: 16-30%, Cấp 4: >30%) và bổ sung khuyến nghị điều trị tương ứng trên giao diện `app.py`.
2. **Giao diện người dùng:**
   - Hào đã kết nối sẵn pipeline, bổ sung hiển thị 4 bước ảnh và cơ chế bắt lỗi `LeafSegmentationError`. Phúc tiếp tục hoàn thiện bố cục trang trí và các tính năng bổ trợ.

### 5.3. Trách nhiệm chung của nhóm về dữ liệu:
- **Tình trạng:** Bộ 10 ảnh hiện có đều là **ảnh tổng hợp (AI/Synthetic)**.
- **Tồn đọng:** Nhóm cần bổ sung tối thiểu **10 ảnh chụp thật từ vườn sầu riêng** để nghiệm thu độ chính xác thực tế trước khi báo cáo hội đồng.
