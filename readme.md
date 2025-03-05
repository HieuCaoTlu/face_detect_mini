# Cách cài đặt và chạy
- Có cài đặt Python 3.11.5
- Cài đặt các package cần thiết (nên thiết lập venv trước)
- `pip install --no-cache-dir -r requirements.txt`
- Hàm chạy: `python ai.py`

# Hàm Predict

- Đầu vào là địa chỉ của 1 ảnh, trả về nhãn dự đoán
- VD: `predict('image.jpg')`

# Hàm Train

- Đầu vào nhận là danh sách địa chỉ ảnh và tên nhãn, trả về số lớp có thể dự đoán hiện tại
- VD: `train(['image.jpg','image2.jpg'],'Hiếu')`
