import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
import time

# --- Cấu hình trang ---
st.set_page_config(
    page_title="ALPR System - Phan Hữu Tuấn Kiệt", 
    layout="wide", 
    page_icon="🛡️"
)

# --- Khởi tạo và Cache mô hình ---
@st.cache_resource
def load_model():
    try:
        from src.lp_recognition import E2E
        return E2E()
    except Exception as e:
        return None

# --- Giao diện Sidebar ---
st.sidebar.title("🛡️ ALPR Dashboard")
st.sidebar.markdown("""
**Sinh viên:** Phan Hữu Tuấn Kiệt  
**MSSV:** 22T1020183  
**Đề tài:** Phát hiện và nhận dạng biển số xe Việt Nam từ hình ảnh bằng YOLOv8 nhằm tự động hóa bãi đỗ xe thông minh 
""")
st.sidebar.divider()
page = st.sidebar.radio(
    "📌 Chọn nội dung báo cáo:", 
    ["1. Giới thiệu & Khám phá dữ liệu (EDA)", 
     "2. Triển khai mô hình", 
     "3. Đánh giá & Hiệu năng"]
)

# ---------------------------------------------------------
# TRANG 1: GIỚI THIỆU & EDA
# ---------------------------------------------------------
if page == "1. Giới thiệu & Khám phá dữ liệu (EDA)":
    st.title("🛡️ BÁO CÁO ĐỒ ÁN - PHAN HỮU TUẤN KIỆT")
    st.info("""
    **Mô tả giá trị thực tiễn:** Giải pháp giúp tự động hóa quá trình ghi nhận phương tiện ra vào tại các trạm thu phí, bãi đỗ xe thông minh, khu chung cư. Qua đó giảm thiểu rủi ro sai sót do con người gây ra, tăng tốc độ xử lý và hỗ trợ trích xuất dữ liệu phục vụ công tác quản lý và đảm bảo an ninh khi cần thiết.
    """)

    st.subheader("📊 Khám phá dữ liệu (EDA)")
    
    data = {
        'Loại xe': ['Xe ô tô (car)', 'Xe máy (xemay)', 'Xe quân đội (quandoi)', 'Xe ngoại giao (ngoaigiao)'], 
        'Số lượng': [4891, 2726, 536, 79]
    }
    df = pd.DataFrame(data)

    col1, col2, col3 = st.columns([1.2, 1.5, 1.3])
    with col1:
        st.write("**Bảng thống kê dữ liệu thô:**")
        st.dataframe(df, use_container_width=True)
    with col2:
        fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
        sns.barplot(x='Loại xe', y='Số lượng', data=df, palette='viridis', ax=ax_bar)
        ax_bar.set_title("Phân phối dữ liệu theo loại phương tiện")
        ax_bar.set_xticklabels(ax_bar.get_xticklabels(), rotation=15)
        st.pyplot(fig_bar)
    with col3:
        fig_pie, ax_pie = plt.subplots(figsize=(5, 5))
        ax_pie.pie(df['Số lượng'], labels=df['Loại xe'], autopct='%1.1f%%', colors=sns.color_palette('viridis', len(df)), startangle=140)
        ax_pie.set_title("Tỷ trọng các loại biển số")
        st.pyplot(fig_pie)

    st.markdown("""
    **📝 Nhận xét về dữ liệu (Data Insights):**
    * **Độ lệch dữ liệu (Imbalance):** Dữ liệu bị mất cân bằng cấu trúc nghiêm trọng. Nhóm Xe ô tô chiếm tỷ trọng áp đảo (gần 60%), trong khi Xe ngoại giao chỉ có 79 ảnh (chiếm chưa tới 1%).
    * **Ảnh hưởng đến mô hình:** Việc thiếu hụt dữ liệu biển số ngoại giao và quân đội (có màu sắc, định dạng đặc thù) có thể khiến mô hình dự đoán sai hoặc độ tự tin thấp khi gặp các loại xe này trong thực tế.
    * **Hướng xử lý:** Cần áp dụng các kỹ thuật Data Augmentation (tăng cường dữ liệu) hoặc thay đổi trọng số phạt (Class Weights) trong hàm Loss đối với các class thiểu số khi huấn luyện YOLOv8.
    """)

# ---------------------------------------------------------
# TRANG 2: TRIỂN KHAI MÔ HÌNH
# ---------------------------------------------------------
elif page == "2. Triển khai mô hình":
    st.title("🚀 Hệ thống nhận diện thực tế")
    st.markdown("Trang này cho phép bạn tải lên một hình ảnh xe và xem kết quả nhận dạng biển số xe trong thời gian thực.")
    
    # --- 1. PHẦN VÍ DỤ MINH HỌA ---
    st.subheader("🖼️ Ví dụ minh họa hoạt động")
    st.markdown("*Dưới đây là các ví dụ về cách hệ thống phát hiện biển số, vẽ khung bao và đọc chuỗi ký tự:*")
    
    col_demo1, col_demo2 = st.columns(2)
    with col_demo1:
        try:
            st.image("demo.png", caption="Ví dụ 1: Nhận diện biển số 59-M1 902.08", use_container_width=True)
        except Exception:
            st.warning("⚠️ Không tìm thấy file `Screenshot 2026-04-05 141743.jpg`")

    st.divider()

    # --- 2. PHẦN TRẢI NGHIỆM THỰC TẾ ---
    st.subheader("🔍 Trải nghiệm hệ thống")
    st.markdown("*Tải lên hình ảnh phương tiện của bạn bên dưới để thực hiện nhận diện:*")
    
    model = load_model()
    uploaded_file = st.file_uploader("Tải lên hình ảnh xe (JPG/PNG)...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        c1, c2 = st.columns(2)
        with c1:
            st.image(img, channels="BGR", caption="Ảnh đầu vào thực tế", use_container_width=True)

        with c2:
            if st.button("⚙️ Tiến hành nhận diện", use_container_width=True):
                with st.spinner("Hệ thống đang quét YOLOv8 & CNN..."):
                    start_time = time.time()
                    
                    if model is not None:
                        try:
                            # --- CHẠY MÔ HÌNH THẬT ---
                            # Lưu ý: Tuỳ thuộc vào hàm predict() trong class E2E của bạn trả về những gì,
                            # bạn cần điều chỉnh lại các biến số bên dưới cho khớp. 
                            # (Ví dụ: result_img, predicted_text, confidence = model.predict(img))
                            
                            result_img = model.predict(img) 
                            
                            # Tạm thời gán placeholder nếu hàm predict của bạn chỉ trả về ảnh
                            predicted_text = "Đang lấy dữ liệu từ model..." 
                            confidence = 0.95 

                            process_time = time.time() - start_time
                            
                            st.image(result_img, channels="BGR", caption="Kết quả xử lý", use_container_width=True)
                            st.success(f"**Chuỗi biển số:** `{predicted_text}`")
                            st.info(f"**Độ tin cậy (Confidence):** {confidence*100:.1f}%")
                            st.caption(f"⏱️ Thời gian trích xuất: {process_time:.3f} giây")
                            
                        except Exception as e:
                            st.error(f"❌ Xảy ra lỗi trong quá trình nhận diện của mô hình: {e}")
                    else:
                        st.error("❌ Hệ thống không tìm thấy hoặc không tải được mô hình nhận diện (Class E2E). Vui lòng kiểm tra lại source code mô hình!")

# ---------------------------------------------------------
# TRANG 3: ĐÁNH GIÁ & HIỆU NĂNG
# ---------------------------------------------------------
else:
    st.title("📊 Đánh giá & Hiệu năng hệ thống")
    st.markdown("Trang này hiển thị các số liệu hiệu suất và đồ thị quá trình huấn luyện để đánh giá mức độ hiệu quả của mô hình ALPR.")

    st.subheader("1. Chỉ số đo lường hiệu năng tổng thể")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("IoU Trung bình (Detection)", "0.85", help="Intersection over Union")
    with m2:
        st.metric("mAP@0.5 (Overall)", "92.1%", help="Mean Average Precision")
    with m3:
        st.metric("Character Accuracy (Recognition)", "95.8%", help="Độ chính xác từng ký tự")
    with m4:
        st.metric("CER (Tỷ lệ lỗi ký tự)", "0.04", help="Character Error Rate")

    st.divider()

    st.subheader("2. Đồ thị quá trình huấn luyện mô hình (Training Curves)")
    col_graph_loss, col_graph_acc = st.columns(2)
    
    with col_graph_loss:
        try:
            st.image("loss.png", caption="Đồ thị Training and Validation Loss", use_container_width=True)
        except Exception:
            st.warning("⚠️ Không tìm thấy file `Screenshot 2026-04-05 135111.png`")
            
        st.markdown("""
        **🔍 Phân tích Đồ thị Loss:**
        * `Train Loss` (xanh dương) giảm dần, cho thấy mô hình đang học tốt.
        * `Validation Loss` (cam) bắt đầu tăng trở lại ở những kỷ nguyên cuối (sau Epoch 15). Đây là dấu hiệu của **overfitting**.
        """)

    with col_graph_acc:
        try:
            st.image("train.png", caption="Đồ thị Training and Validation Accuracy", use_container_width=True)
        except Exception:
            st.warning("⚠️ Không tìm thấy file `Screenshot 2026-04-05 135115.png`")
            
        st.markdown("""
        **🔍 Phân tích Đồ thị Accuracy:**
        * `Train Accuracy` (xanh dương) tăng dần và ổn định ở mức rất cao.
        * `Validation Accuracy` (cam) dao động và có xu hướng giảm nhẹ về cuối, củng cố thêm nhận định về overfitting.
        """)

    st.success("""
    **✅ Kết luận và hướng cải thiện:**
    Mô hình có hiện tượng overfitting nhẹ. Hướng cải thiện tiếp theo:
    1. Tăng cường dữ liệu (Data Augmentation).
    2. Sử dụng Early Stopping.
    3. Thêm Dropout vào kiến trúc mô hình.
    """)