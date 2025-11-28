import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # Thư viện để vẽ biểu đồ
import io # Thư viện để xử lý dữ liệu hình ảnh trong bộ nhớ

# NOTE: Để chạy được ứng dụng này với mô hình thực tế, bạn sẽ cần:
# 1. Huấn luyện mô hình XGBoost (hoặc Random Forest) ngoài file này.
# 2. Lưu mô hình (ví dụ: bằng joblib hoặc pickle).
# 3. Tải mô hình đã lưu tại đây (ví dụ: model = joblib.load('best_xgb_model.pkl'))
# Do hạn chế của môi trường, chúng tôi chỉ mô phỏng kết quả dự đoán.

# Thiết lập cấu hình trang
st.set_page_config(layout="wide")

# Tiêu đề ứng dụng
st.header("Ứng dụng Dự đoán Nguy cơ Bệnh Tim - Giải thích XAI")
st.markdown("Nhập thông tin bên dưới để đánh giá nguy cơ mắc bệnh tim của bạn:")

# --- Thiết lập 2 cột chính cho đầu vào ---
col1, col2 = st.columns(2)

# ==============================================================================
# CỘT 1: THÔNG TIN CƠ BẢN VÀ THAM SỐ SINH HỌC
# ==============================================================================
with col1:
    st.subheader("Thông tin cơ bản & Sinh học:")

    # 1. Giới tính (st.radio) -> Gender_Male, Gender_Female
    st.markdown("**Giới tính**")
    gioi_tinh = st.radio(
        "",
        options=("Nữ", "Nam"),
        index=0,
        horizontal=True,
        key="gioi_tinh"
    )

    # 2. Tuổi (st.number_input) -> Age
    tuoi = st.number_input(
        "**Tuổi**",
        min_value=18,
        max_value=100,
        value=50,
        step=1,
        key="tuoi"
    )

    # 3. Huyết áp (st.number_input) -> Blood Pressure
    huyet_ap = st.number_input(
        "**Huyết áp (tâm thu)**",
        min_value=90.0,
        max_value=200.0,
        value=135.0,
        step=1.0,
        key="huyet_ap"
    )

    # 4. Mức Cholesterol (st.number_input) -> Cholesterol Level
    cholesterol = st.number_input(
        "**Mức Cholesterol (mg/dL)**",
        min_value=100.0,
        max_value=400.0,
        value=220.0,
        step=5.0,
        key="cholesterol"
    )
    
    # 5. BMI (st.number_input) -> BMI
    bmi = st.number_input(
        "**Chỉ số BMI**",
        min_value=15.0,
        max_value=50.0,
        value=25.0,
        step=0.1,
        key="bmi"
    )
    
    # 6. Mức độ đường huyết lúc đói (st.number_input) -> Fasting Blood Sugar
    fbs = st.number_input(
        "**Đường huyết lúc đói (mg/dL)**",
        min_value=50.0,
        max_value=300.0,
        value=95.0,
        step=1.0,
        help="Giá trị trên 126 mg/dL thường được coi là cao."
    )

# ==============================================================================
# CỘT 2: THÔNG TIN HÀNH VI VÀ CHỈ SỐ KHÁC
# ==============================================================================
with col2:
    st.subheader("Thông tin hành vi & Chỉ số khác:")

    # 1. Thói quen Tập thể dục (Exercise Habits)
    st.markdown("**Thói quen Tập thể dục**")
    exercise_habits = st.radio(
        "Chọn một mức độ:",
        options=("Thấp", "Trung bình", "Cao"),
        index=1,
        horizontal=True,
        key="exercise_habits"
    )

    # 2. Mức độ Căng thẳng (Stress Level)
    st.markdown("**Mức độ Căng thẳng**")
    stress_level = st.select_slider(
        "Chọn mức độ từ Thấp đến Cao:",
        options=["Thấp", "Trung bình", "Cao"],
        value="Trung bình",
        key="stress_level"
    )

    # 3. Thời gian ngủ (Sleep Hours)
    sleep_hours = st.slider(
        "**Thời gian ngủ (giờ/ngày)**",
        min_value=3.0,
        max_value=12.0,
        value=7.0,
        step=0.5,
        key="sleep_hours"
    )
    
    st.write("---") # Đường kẻ phân chia

    # 4. Các câu hỏi Checkbox (Lịch sử y tế)
    smoking = st.checkbox("Hút thuốc lá?", key="smoking")
    family_disease = st.checkbox("Gia đình có tiền sử bệnh tim?", key="family_disease")
    diabetes = st.checkbox("Mắc bệnh tiểu đường?", key="diabetes")
    high_bp = st.checkbox("Bị Cao huyết áp?", key="high_bp")
    low_hdl = st.checkbox("HDL Cholesterol (tốt) thấp?", key="low_hdl")
    high_ldl = st.checkbox("LDL Cholesterol (xấu) cao?", key="high_ldl")

# ==============================================================================
# HÀM DỰ ĐOÁN VÀ TIỀN XỬ LÝ
# ==============================================================================

def preprocess_input(input_data):
    """Chuyển đổi dữ liệu đầu vào từ Streamlit sang định dạng NumPy 2D
    mà mô hình đã được huấn luyện mong đợi."""

    # Tên các cột theo thứ tự trong mô hình (giả định)
    feature_names = [
        'Age', 'Blood Pressure', 'Cholesterol Level', 'BMI', 'Sleep Hours', 
        'Fasting Blood Sugar', 
        'Gender_Female', 'Gender_Male', 
        'Smoking', 'Family Heart Disease', 'Diabetes', 
        'High Blood Pressure', 'Low HDL Cholesterol', 'High LDL Cholesterol',
        'Exercise Habits_Low', 'Exercise Habits_Medium', 'Exercise Habits_High', 
        'Stress Level_Low', 'Stress Level_Medium', 'Stress Level_High',
    ]

    # Khởi tạo ma trận đặc trưng với các giá trị 0
    X = np.zeros(len(feature_names))
    
    # 1. Điền các cột số trực tiếp
    X[0] = input_data['tuoi']
    X[1] = input_data['huyet_ap']
    X[2] = input_data['cholesterol']
    X[3] = input_data['bmi']
    X[4] = input_data['sleep_hours']
    X[5] = input_data['fbs']

    # 2. Xử lý Giới tính (One-Hot Encoding)
    if input_data['gioi_tinh'] == 'Nữ':
        X[6] = 1 # Gender_Female
    else:
        X[7] = 1 # Gender_Male

    # 3. Xử lý các Checkbox (Boolean -> 0/1)
    X[8] = 1 if input_data['smoking'] else 0
    X[9] = 1 if input_data['family_disease'] else 0
    X[10] = 1 if input_data['diabetes'] else 0
    X[11] = 1 if input_data['high_bp'] else 0
    X[12] = 1 if input_data['low_hdl'] else 0
    X[13] = 1 if input_data['high_ldl'] else 0

    # 4. Xử lý Thói quen Tập thể dục (OHE)
    idx_start_exercise = 14
    if input_data['exercise_habits'] == 'Thấp':
        X[idx_start_exercise] = 1
    elif input_data['exercise_habits'] == 'Trung bình':
        X[idx_start_exercise + 1] = 1
    else:
        X[idx_start_exercise + 2] = 1

    # 5. Xử lý Mức độ Căng thẳng (OHE)
    idx_start_stress = 17
    if input_data['stress_level'] == 'Thấp':
        X[idx_start_stress] = 1
    elif input_data['stress_level'] == 'Trung bình':
        X[idx_start_stress + 1] = 1
    else:
        X[idx_start_stress + 2] = 1
        
    # Trả về mảng 2D sẵn sàng cho mô hình
    return X.reshape(1, -1), feature_names

def generate_mock_shap_plot(shap_values_dict):
    """Generates a mock SHAP summary plot (horizontal bar chart)."""
    
    # Sort features based on absolute SHAP value magnitude
    sorted_features = sorted(shap_values_dict.items(), key=lambda item: abs(item[1]), reverse=True)
    
    # Take the top 7 features
    top_n = 7
    top_features = sorted_features[:top_n]
    
    # Extract names and values for plotting
    names = [f[0] for f in top_features]
    values = [f[1] for f in top_features]
    
    # Determine colors for positive (risk increase) and negative (risk decrease)
    # Giả sử giá trị SHAP dương -> Tăng nguy cơ (Đỏ); Giá trị SHAP âm -> Giảm nguy cơ (Xanh)
    colors = ['red' if v > 0 else 'blue' for v in values]
    
    # Reverse order for plotting (most important at top)
    names.reverse()
    values.reverse()
    colors.reverse()

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot bars
    ax.barh(names, values, color=colors)
    
    # Add labels and title
    ax.set_xlabel("Đóng góp vào Nguy cơ (Giá trị SHAP)")
    ax.set_ylabel("Đặc trưng")
    ax.set_title("7 Yếu tố Quan trọng nhất cho Dự đoán (Mô phỏng SHAP)")
    
    # Add custom legend for color interpretation
    red_patch = plt.Rectangle((0, 0), 1, 1, fc="red", label='Tăng nguy cơ')
    blue_patch = plt.Rectangle((0, 0), 1, 1, fc="blue", label='Giảm nguy cơ')
    ax.legend(handles=[red_patch, blue_patch], loc='lower right', frameon=True)

    # Use BytesIO to save the plot as an image in memory
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig) # Close the figure to free memory
    
    return buf

def mock_predict_and_explain(model_name, features):
    """Mô phỏng kết quả dự đoán, xác suất và giải thích XAI."""
    
    # Tên đặc trưng được sử dụng trong giải thích
    feature_map = {
        0: 'Tuổi', 1: 'Huyết áp', 2: 'Mức Cholesterol', 3: 'BMI', 
        8: 'Hút thuốc', 11: 'Cao huyết áp', 16: 'Tập thể dục Cao', 18: 'Stress Trung bình'
    }
    
    # Trích xuất các giá trị quan trọng cho mô phỏng
    age = features[0]
    bp = features[1]
    cholesterol = features[2]
    smoking = features[8]
    exercise_high = features[16]
    high_bp_flag = features[11]
    
    # Nguy cơ cơ bản dựa trên tuổi và cholesterol
    base_risk = (age * 0.4 + cholesterol * 0.2 + bp * 0.1) / 100
    
    # Thêm yếu tố hành vi
    base_risk += 0.1 * smoking
    base_risk -= 0.05 * exercise_high
    base_risk += 0.15 * high_bp_flag
    
    # Thêm yếu tố ngẫu nhiên và điều chỉnh
    np.random.seed(42)
    risk_score = np.clip(base_risk + np.random.uniform(-0.1, 0.1), 0.05, 0.95)
    
    # Dự đoán (0: Thấp/Không, 1: Cao/Có)
    prediction = 1 if risk_score >= 0.5 else 0
    
    # --- MOCK SHAP VALUES ---
    mock_shap_values = {
        'Tuổi': 0.007 * age - 0.3, # Luôn dương và tăng theo tuổi
        'Mức Cholesterol': 0.0015 * cholesterol - 0.2, # Luôn dương và tăng theo Cholesterol
        'Huyết áp': 0.001 * bp - 0.15,
        'BMI': features[3] * 0.002,
        'Hút thuốc': 0.25 * smoking, 
        'Tiền sử gia đình': 0.15 * features[9],
        'Tập thể dục Cao': -0.2 * exercise_high, # Rất âm nếu tập thể dục cao
        'Stress Level_Medium': 0.08 * features[18],
        'Đường huyết đói': 0.001 * features[5]
    }
    
    # --- MOCK EXPLANATION ---
    if prediction == 1:
        result_text = "Nguy cơ **CAO** mắc bệnh tim."
        color = "red"
        explanation = f"""
        Kết quả này được thúc đẩy bởi các yếu tố sau:
        - **Tuổi {age:.0f}:** Yếu tố đóng góp mạnh mẽ nhất, đặc biệt khi trên 60 tuổi.
        - **Mức Cholesterol {cholesterol:.1f} mg/dL:** Là yếu tố sinh học tăng nguy cơ quan trọng.
        - **Hành vi (Hút thuốc/Tiền sử):** Nếu có, yếu tố này đóng góp đáng kể.
        Mô hình đề xuất cần theo dõi chặt chẽ và tham khảo ý kiến bác sĩ.
        """
    else:
        result_text = "Nguy cơ **THẤP** mắc bệnh tim."
        color = "green"
        explanation = f"""
        Nguy cơ thấp là nhờ sự kết hợp của:
        - **Tập thể dục Cao:** Yếu tố giảm nguy cơ quan trọng nhất.
        - **Không hút thuốc:** Giảm đáng kể đóng góp nguy cơ.
        - **Chỉ số sinh học ổn định:** Mức Cholesterol và Huyết áp nằm trong phạm vi chấp nhận được.
        Hãy duy trì thói quen sinh hoạt lành mạnh này!
        """
    
    # Trả về kết quả và mô phỏng SHAP
    return prediction, risk_score, result_text, color, explanation, mock_shap_values

# ==============================================================================
# PHẦN CHỌN MÔ HÌNH VÀ DỰ ĐOÁN
# ==============================================================================

st.markdown("---")
st.subheader("Chọn mô hình để dự đoán:")

# Chia cột cho lựa chọn mô hình và nút dự đoán
col_model_select, col_predict = st.columns([1, 1])

with col_model_select:
    # Lựa chọn mô hình
    model_choice = st.radio(
        "Chọn mô hình đã được tối ưu:",
        options=("XGBoost Classifier", "Random Forest Classifier"),
        index=0,
        horizontal=True,
        key="model_choice"
    )

with col_predict:
    # Nút Dự đoán
    if st.button("DỰ ĐOÁN NGUY CƠ", help="Nhấn để chạy mô hình và xem kết quả"):
        
        # 1. Thu thập dữ liệu
        input_data = {
            'gioi_tinh': gioi_tinh, 'tuoi': tuoi, 'huyet_ap': huyet_ap,
            'cholesterol': cholesterol, 'bmi': bmi, 'sleep_hours': sleep_hours,
            'fbs': fbs, 'exercise_habits': exercise_habits,
            'stress_level': stress_level, 'smoking': smoking,
            'family_disease': family_disease, 'diabetes': diabetes,
            'high_bp': high_bp, 'low_hdl': low_hdl, 'high_ldl': high_ldl
        }
        
        # 2. Tiền xử lý dữ liệu
        X_processed, feature_names = preprocess_input(input_data)
        
        # 3. Dự đoán (MÔ PHỎNG)
        prediction, risk_score, result_text, color, explanation, mock_shap_values = mock_predict_and_explain(model_choice, X_processed[0])

        st.markdown("### Kết quả Dự đoán")
        st.markdown(f"<div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid {color};'>"
                    f"<p style='font-size: 1.2em; font-weight: bold;'>{result_text}</p>"
                    f"<p>Xác suất Nguy cơ Tim mạch: <span style='color: {color}; font-weight: bold;'>{risk_score*100:.2f}%</span></p>"
                    f"</div>", unsafe_allow_html=True)
        
        # 4. Hiển thị Giải thích XAI (Văn bản)
        st.markdown("### 💡 Giải thích Mô hình (XAI)")
        st.info(explanation)
        
        # 5. Hiển thị Biểu đồ SHAP (Mô phỏng)
        st.markdown("### 📈 Biểu đồ Đóng góp Đặc trưng (SHAP - Mô phỏng)")
        shap_plot_buffer = generate_mock_shap_plot(mock_shap_values)
        st.image(shap_plot_buffer, caption='Biểu đồ SHAP Summary (Mô phỏng)')
        
        st.markdown(f"*(Lưu ý: Kết quả và biểu đồ được tạo ra bằng mô phỏng, không phải từ mô hình học máy thực tế.)*")

# Thêm ghi chú về các giá trị ánh xạ để người dùng dễ hiểu
st.markdown("---")
st.markdown("#### Bảng ánh xạ giá trị cho Mô hình (Dự kiến):")
st.markdown("""
* **Thói quen Tập thể dục/Stress:** Thấp (0), Trung bình (1), Cao (2).
* **Giới tính:** Nam (1, 0), Nữ (0, 1) trong One-Hot Encoding.
* **Checkbox:** True (1), False (0).
""")
