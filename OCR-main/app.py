import streamlit as st
import torch
from PIL import Image
from collections import OrderedDict
import io
import argparse
import models.crnn as crnn_model
import utils
import dataset


# 1. Bảng chữ cái đầy đủ (Size 97)
ALPHABET = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '
N_CLASS = len(ALPHABET) + 1
# 2. Cấu hình Model
IMG_H = 32
N_CHANNELS = 1 # Đen trắng
N_HIDDEN = 256

# ==========================================
# CÁC HÀM XỬ LÝ CHÍNH
# ==========================================

@st.cache_resource() # Lưu mô hình vào bộ nhớ đệm, chỉ load 1 lần
def load_ocr_model(model_path):
    """Đánh thức AI và nạp trí nhớ từ file .pth"""
    # Khởi tạo khung mô hình
    model = crnn_model.CRNN(IMG_H, N_CHANNELS, N_CLASS, N_HIDDEN)
    
    # Đưa lên GPU nếu có, không thì dùng CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Xử lý lỗi "module." (Size Mismatch) nếu lúc train dùng nhiều GPU
    try:
        state_dict = torch.load(model_path, map_location=device)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    except Exception as e:
        st.error(f"Lỗi không nạp được file model (.pth): {e}")
        st.stop()
        
    model.to(device)
    model.eval() # Chuyển sang chế độ "làm việc"
    return model, device

def predict_image(model, device, pil_image):
    """Hàm xử lý ảnh và ra lệnh cho AI đọc chữ"""
    converter = utils.strLabelConverter(ALPHABET)
    transformer = dataset.resizeNormalize((100, 32))
    
    # 1. Tiền xử lý ảnh (giống hệt lúc train/demo)
    img_bw = pil_image.convert('L') # Chuyển thành ảnh đen trắng
    img_tensor = transformer(img_bw)
    
    # Đưa ảnh lên cùng thiết bị với model
    img_tensor = img_tensor.to(device)
    
    # Chỉnh lại shape (batch=1, channel=1, h=32, w=100)
    img_tensor = img_tensor.view(1, *img_tensor.size())
    
    # 2. AI bắt đầu dự đoán
    with torch.no_grad(): # Tắt tính toán gradient để tăng tốc
        preds = model(img_tensor)
        
    # 3. Dịch kết quả từ dạng số sang dạng chữ
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    preds_size = torch.IntTensor([preds.size(0)])
    
    # CTC Decode (lấy raw=False để ra chuỗi sạch)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    
    return sim_pred

# ==========================================
# GIAO DIỆN WEB (Streamlit UI)
# ==========================================

# 1. Cấu hình trang (Tiêu đề trên tab trình duyệt)
st.set_page_config(page_title="AI OCR Project ", layout="centered")

# 2. Tiêu đề chính trên trang web
st.title("Nhận Dạng Chữ OCR")
st.markdown("---")

# 3. Phần cấu hình Model
st.sidebar.header("Cấu Hình Hệ Thống")

# Bạn hãy thay đường dẫn này bằng file .pth xịn nhất trong thư mục expr của bạn nhé
default_model = 'expr/netCRNN_24_3000.pth'
model_path = st.sidebar.text_input("Đường dẫn file Model (.pth):", value=default_model)

# Trạng thái thiết bị (CUDA/CPU)
if torch.cuda.is_available():
    st.sidebar.success(f"Thiết bị: GPU ({torch.cuda.get_device_name(0)})")
else:
    st.sidebar.warning("Thiết bị: CPU (Tốc độ sẽ chậm hơn)")

# Nút Load Model
if st.sidebar.button("Đánh Thức AI"):
    with st.spinner("Đang nạp mô hình, vui lòng chờ..."):
        st.session_state['model'], st.session_state['device'] = load_ocr_model(model_path)
    st.sidebar.success("Mô hình đã sẵn sàng!")

# 4. Phần Upload Ảnh và Hiển Thị Kết Quả
st.header("1. Upload Bức Ảnh")
uploaded_file = st.file_uploader("Kéo thả hoặc bấm để chọn ảnh...", type=["png", "jpg", "jpeg"])

# Kiểm tra xem Model đã được load chưa
if 'model' not in st.session_state:
    st.info("Vui lòng bấm nút **'Đánh Thức AI'** ở thanh bên trái trước khi upload ảnh.")
    st.stop()

if uploaded_file is not None:
    # Mở ảnh từ file upload
    input_image = Image.open(uploaded_file)
    
    # Hiển thị ảnh gốc
    
    st.image(input_image, caption='Ảnh đã upload', use_container_width=True)
    st.markdown("---")
    st.header("2. Kết Quả")
    
    # Nút bấm để thực hiện nhận dạng
    if st.button("AI BẮT ĐẦU ĐỌC CHỮ"):
        with st.spinner("AI đang 'nhìn' và 'đọc' chữ, vui lòng chờ..."):
            # Gọi hàm dự đoán
            result = predict_image(st.session_state['model'], st.session_state['device'], input_image)
            
        # Hiển thị kết quả to và rõ ràng
        st.success("AI ĐÃ ĐỌC XONG!")
        st.markdown(f"### Kết quả nhận dạng: ")
        st.code(result, language='text') # Hiện kết quả trong khung code cho đẹp

else:
    st.info("Chờ bạn upload ảnh...")
