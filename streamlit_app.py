import streamlit as st
import pandas as pd

# Thiết lập cấu hình trang
st.set_page_config(layout="wide")

# Tiêu đề ứng dụng
st.header("Ứng dụng Dự đoán Bệnh Khô Mắt - Giải thích XAI")
st.markdown("Nhập thông tin bên dưới để dự đoán nguy cơ mắt:")

# --- Thiết lập 2 cột chính cho đầu vào ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Nhập thông tin cơ bản:")
    
    # Giới tính (st.radio)
    st.markdown("**Giới tính**")
    gioi_tinh = st.radio(
        "", 
        options=("Nam", "Nữ"), 
        index=0, # Nam được chọn mặc định
        horizontal=True,
        key="gioi_tinh"
    )

    # Tuổi (st.number_input)
    tuoi = st.number_input(
        "**Tuổi**", 
        min_value=1, 
        max_value=120, 
        value=30, 
        step=1,
        key="tuoi"
    )

    # Thời gian ngủ (st.slider)
    thoi_gian_ngu = st.slider(
        "**Thời gian ngủ (giờ)**", 
        min_value=0, 
        max_value=12, 
        value=7, 
        step=1,
        key="thoi_gian_ngu"
    )

    # Màn hình thời gian (st.slider)
    man_hinh_thoi_gian = st.slider(
        "**Màn hình thời gian (giờ)**", 
        min_value=0, 
        max_value=16, 
        value=8, 
        step=1,
        key="man_hinh_thoi_gian"
    )

    # Chất lượng giấc ngủ (st.slider)
    chat_luong_giac_ngu = st.slider(
        "**Chất lượng giấc ngủ (0-5)**", 
        min_value=0, 
        max_value=5, 
        value=3, 
        step=1,
        key="chat_luong_giac_ngu"
    )

    # Mức độ dị ứng (st.slider)
    muc_do_di_ung = st.slider(
        "**Mức độ dị ứng (0-5)**", 
        min_value=0, 
        max_value=5, 
        value=3, 
        step=1,
        key="muc_do_di_ung"
    )

with col2:
    st.subheader("Thông tin hành vi và triệu chứng:")
    
    # Số bước/ngày (st.number_input) - Giả sử 5000 là giá trị mặc định
    so_buoc_ngay = st.number_input(
        "**Số bước/ngày**", 
        min_value=0, 
        value=5000, 
        step=100,
        key="so_buoc_ngay"
    )

    # Hoạt động thể chất (st.number_input)
    hoat_dong_the_chat = st.number_input(
        "**Hoạt động thể chất (phút)**", 
        min_value=0, 
        value=30, 
        step=5,
        key="hoat_dong_the_chat"
    )
    
    st.write("---") # Đường kẻ phân chia
    
    # Các câu hỏi Checkbox
    hut_thuoc = st.checkbox("Hút thuốc?", key="hut_thuoc")
    thiet_bi_khi_ngu = st.checkbox("Sử dụng thiết bị trước khi ngủ?", key="thiet_bi_khi_ngu")
    roi_loan_giac_ngu = st.checkbox("Có rối loạn giấc ngủ không?", key="roi_loan_giac_ngu")
    anh_sang_xanh = st.checkbox("Sử dụng ánh sáng xanh của kính lọc?", key="anh_sang_xanh")
    kho_chiu_trong_mat = st.checkbox("Cảm giác khó chịu trong mắt?", key="kho_chiu_trong_mat")
    do_mat = st.checkbox("Đỏ mắt?", key="do_mat")
    ngua_khe_chua = st.checkbox("Ngứa/Khó chịu?", key="ngua_khe_chua")


# --- Chọn Mô hình và Dự đoán (Nằm ở cuối trang) ---
st.markdown("---")
st.subheader("Chọn mô hình để dự đoán:")

# Chia cột cho lựa chọn mô hình và nút dự đoán
col_model_select, col_predict = st.columns([1, 1])

with col_model_select:
    # Lựa chọn mô hình
    model_choice = st.radio(
        "",
        options=("Dự đoán với Rừng ngẫu nhiên", "Dự đoán với XGB"),
        index=0,
        horizontal=True,
        key="model_choice"
    )

with col_predict:
    # Nút Dự đoán
    # Sử dụng st.button bên trong cột
    if st.button("Dự đoán", help="Nhấn để chạy mô hình và xem kết quả"):
        
        # --- Phần này chỉ là mô phỏng kết quả dự đoán ---
        st.success(f"Đã chạy dự đoán với mô hình **{model_choice}**!")
        st.info("Kết quả dự đoán: Nguy cơ Trung Bình (Đây là kết quả mô phỏng, cần tích hợp mô hình thực tế để chạy)")
        
        # In các giá trị đầu vào để kiểm tra
        # st.write({
        #     "Giới tính": gioi_tinh,
        #     "Tuổi": tuoi,
        #     "Thời gian ngủ": thoi_gian_ngu,
        #     "Hút thuốc": hut_thuoc
        # })

# Để chạy ứng dụng, lưu file này là `app.py` và chạy lệnh:
# streamlit run app.py
