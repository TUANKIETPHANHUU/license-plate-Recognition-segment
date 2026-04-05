import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="ALPR System - Phan Hữu Tuấn Kiệt",
    page_icon="🛡️",
    layout="wide"
)

# Giao diện CSS tùy chỉnh để làm đẹp các thành phần
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { 
        background-color: #ffffff; 
        padding: 20px; 
        border-radius: 12px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #ececf1;
    }
    div[data-testid="stExpander"] { border: none !important; box-shadow: none !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. KHỞI TẠO & CACHE MÔ HÌNH (MLOps) ---
@st.cache_resource
def load_alpr_model():
    try:
        # Thay thế bằng class/function load model thực tế của bạn
        # from src.lp_recognition import E2E
        # return E2E()
        return "Model Loaded" 
    except Exception as e:
        return None

model = load_alpr_model()

# --- 3. THANH ĐIỀU HƯỚNG (SIDEBAR) ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=80)
    st.title("🛡️ ALPR Dashboard")
    st.info(f"""
    **Đề tài:** Nhận dạng biển số xe VN bằng YOLOv8 & CNN
    **SV:** Phan Hữu Tuấn Kiệt
    **MSSV:** 22T1020183
    """)
    st.divider()
    page = st.radio(
        "📌 Danh mục báo cáo:",
        ["1. Giới thiệu & EDA", "2. Triển khai mô hình", "3. Đánh giá & Hiệu năng"]
    )

# ---------------------------------------------------------
# TRANG 1: GIỚI THIỆU & EDA
# ---------------------------------------------------------
if page == "1. Giới thiệu & EDA":
    st.title("📄 Báo cáo Đồ án tốt nghiệp")
    
    with st.container():
        st.subheader("💡 Giá trị thực tiễn")
        st.write("""
        Hệ thống ALPR (Automatic License Plate Recognition) giúp tự động hóa việc ghi nhận xe ra vào, 
        giảm thiểu 90% thời gian kiểm soát thủ công, hỗ trợ an ninh và truy xuất dữ liệu thông minh 
        cho các bãi đỗ xe hiện đại.
        """)

    st.divider()
    st.subheader("📊 Khám phá dữ liệu (EDA)")
    
    # Dữ liệu thực tế
    data = {
        'Loại xe': ['Xe ô tô', 'Xe máy', 'Xe quân đội', 'Xe ngoại giao'],
        'Số lượng': [4891, 2726, 536, 79]
    }
    df = pd.DataFrame(data)

    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.markdown("**Bảng dữ liệu thô:**")
        st.dataframe(df, use_container_width=True)
        st.markdown("""
        **📝 Nhận xét:**
        - Dữ liệu bị **lệch (imbalance)** rõ rệt. Nhóm xe dân sự chiếm đa số.
        - Cần áp dụng **Augmentation** (xoay, nhiễu, đổi màu) để cải thiện nhận diện cho xe ngoại giao/quân đội.
        """)

    with col2:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        # Biểu đồ cột
        sns.barplot(x='Loại xe', y='Số lượng', data=df, palette='viridis', ax=ax[0])
        ax[0].set_title("Phân phối nhãn dữ liệu")
        # Biểu đồ tròn
        ax[1].pie(df['Số lượng'], labels=df['Loại xe'], autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
        ax[1].set_title("Tỷ trọng các loại biển số")
        st.pyplot(fig)

# ---------------------------------------------------------
# TRANG 2: TRIỂN KHAI MÔ HÌNH
# ---------------------------------------------------------
elif page == "2. Triển khai mô hình":
    st.title("🚀 Hệ thống nhận diện thực tế")
    
    st.write("Vui lòng tải lên hình ảnh phương tiện để hệ thống phân tích.")
    
    uploaded_file = st.file_uploader("📤 Tải lên ảnh xe (JPG, PNG)...", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        # Đọc ảnh
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        col_input, col_output = st.columns(2)
        
        with col_input:
            st.image(image, caption="Ảnh đầu vào", use_container_width=True)

        with col_output:
            if st.button("🔍 Bắt đầu nhận diện", use_container_width=True, type="primary"):
                with st.spinner("Đang chạy YOLOv8 & CNN..."):
                    # Mô phỏng quá trình xử lý
                    time.sleep(1.5) 
                    
                    # Giả lập kết quả (Thay bằng logic model thực tế của bạn)
                    predicted_plate = "59-H1 435.64"
                    confidence = 0.985
                    
                    # Vẽ khung giả lập (nếu không có model thật tại đây)
                    res_img = img_array.copy()
                    cv2.rectangle(res_img, (50, 50), (250, 150), (0, 255, 0), 3)
                    
                    st.image(res_img, caption="Kết quả Detection", use_container_width=True)
                    st.success(f"📌 **Biển số dự đoán:** {predicted_plate}")
                    st.metric("Độ tin cậy (Confidence)", f"{confidence*100:.2f}%")

# ---------------------------------------------------------
# TRANG 3: ĐÁNH GIÁ & HIỆU NĂNG
# ---------------------------------------------------------
else:
    st.title("📊 Đánh giá & Hiệu năng hệ thống")
    
    # Hiển thị Metric kỹ thuật theo yêu cầu
    st.subheader("🎯 Chỉ số đo lường (Metrics)")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Precision", "99.44%", delta="Tối ưu")
    with c2: st.metric("Recall", "86.86%", delta="Cần cải thiện")
    with c3: st.metric("mIoU", "0.7970")
    with c4: st.metric("Xử lý (FPS)", "7.83")

    st.divider()

    # Hiển thị ảnh đồ thị từ file bạn cung cấp
    st.subheader("📈 Đồ thị huấn luyện thực tế (YOLOv8 & CNN)")
    col_img1, col_img2 = st.columns(2)
    
    with col_img1:
        try:
            st.image("train.png", caption="Training & Validation Accuracy", use_container_width=True)
        except:
            st.warning("Vui lòng thêm file 'accuracy_chart.png' vào thư mục.")

    with col_img2:
        try:
            st.image("loss.png", caption="Training & Validation Loss", use_container_width=True)
        except:
            st.warning("Vui lòng thêm file 'loss_chart.png' vào thư mục.")

    st.divider()
    
    st.subheader("📝 Phân tích sai số (Error Analysis)")
    col_err1, col_err2 = st.columns(2)
    with col_err1:
        st.error("**Trường hợp dự đoán sai:**")
        st.write("""
        - Biển số bị chói đèn pha hoặc bị bùn đất che khuất.
        - Nhầm lẫn các ký tự tương đồng: **8-B**, **0-D**, **5-S**.
        - Biển số xe máy lắp nghiêng quá 45 độ.
        """)
    
    with col_err2:
        st.success("**Hướng cải thiện:**")
        st.write("""
        - Sử dụng **Regex** để hậu xử lý chuỗi ký tự theo format biển số Việt Nam.
        - Thu thập thêm dữ liệu ban đêm và góc nghiêng.
        - Thêm bước **Sharpening** ảnh biển số trước khi đưa vào mô hình CNN.
        """)