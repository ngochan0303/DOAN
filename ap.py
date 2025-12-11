# import streamlit as st
# import joblib
# import pandas as pd
# import shap
# import matplotlib.pyplot as plt

# st.set_page_config(
#     page_title="My Streamlit App",
#     page_icon="🧠",
#     layout="wide",  # * makes the app use the full screen width
# )

# @st.cache_resource
# def load_model():
#     return joblib.load('npar2/rfc.joblib')
# @st.cache_data
# def load_data():
#     return joblib.load('npar2/X_train.joblib')
# rfc = load_model()

# X_train = load_data()
# explainer = shap.Explainer(model = rfc, masker =X_train,feature_names=X_train.shape)




# st.title('AP Predict: Machine Learning-Based Mortality Prediction in Acute Pancreatitis')
# col1, col2 = st.columns(2)

# vars = {}
# with col1:
#     vars['pt_max'] = st.number_input(label='PT', min_value=8.8, max_value=148.7, step=0.1)
#     vars['bilirubin_total_max'] = st.number_input(label='Bilirubin total', min_value=-23.2, max_value=51.2, step=0.1)
#     vars['bun_max'] = st.number_input(label='BUN', min_value=3, max_value=181, step=1)
#     vars['rdw_max'] = st.number_input(label='RDW', min_value=11.8, max_value=34.9, step=0.1)
#     vars['NPAR'] = st.number_input(label='NPAR', min_value=1.36, max_value=71.5, step=0.1)
#     vars['sapsii'] = st.number_input(label='SAP SII', min_value=6, max_value=94, step=1)

# with col2:
#     vars['sofa'] = st.number_input(label='SOFA', min_value=0, max_value=21, step=1)
#     vars['cci'] = st.number_input(label='CCI', min_value=0, max_value=17, step=1)
#     vars['apsiii'] = st.number_input(label='APSIII', min_value=7, max_value=159, step=1)
#     vars['temperature_mean'] = st.number_input(label='Temperature body', min_value=33.6, max_value=40.1, step=0.1)
#     vars['vasopressin'] = st.selectbox(label='Vasopressin', options=[(0, 'No'), (1, 'Yes')], format_func=lambda v: v[1])[0]
#     vars['has_sepsis'] = st.selectbox(label='Has sepsis', options=[(0, 'No'), (1, 'Yes')], format_func=lambda v: v[1])[0]

# arr = ['Survival', 'Died']

# if st.button('Predict'):
#     df_pred = pd.DataFrame([vars])
#     st.write(df_pred.iloc[:1])

#     pred = rfc.predict(df_pred.iloc[:1])[0]
#     pred_prob = rfc.predict_proba(df_pred.iloc[:1])[0]
#     st.write(df_pred.iloc[:1])

#     st.write(f'Predict: {arr[pred]}, Probability: {pred_prob:.2f}')
#     # st.write( df_pred.iloc[0])
#     shap_values = explainer(df_pred.iloc[:11])
#     fig = shap.plots.force(shap_values[0,:,1],matplotlib =True)
#     st.pyplot(fig)
#     fig,ax = plt.subplots()
#     shap.plots.waterfall(shap_values[0,:,1],show= False)
#     st.pyplot(fig)



# (pt_max                  8.800000
#  bilirubin_total_max   -23.193906
#  bun_max                 3.000000
#  rdw_max                11.800000
#  NPAR                    1.363636
#  sapsii                  6.000000
#  sofa                    0.000000
#  cci                     0.000000
#  apsiii                  7.000000
#  temperature_mean       33.600000
#  vasopressin             0.000000
#  has_sepsis              0.000000
#  dtype: float64,
#  pt_max                 148.700000
#  bilirubin_total_max     51.200000
#  bun_max                181.000000
#  rdw_max                 34.900000
#  NPAR                    71.538462
#  sapsii                  94.000000
#  sofa                    21.000000
#  cci                     17.000000
#  apsiii                 159.000000
#  temperature_mean        40.104118
#  vasopressin              1.000000
#  has_sepsis               1.000000
#  dtype: float64)

# import streamlit as st
# import joblib
# import pandas as pd
# import shap
# import matplotlib.pyplot as plt

# st.set_page_config(
#     page_title="My Streamlit App",
#     page_icon="🧠",
#     layout="wide",
# )

# @st.cache_resource
# def load_model():
#     return joblib.load('npar2/rfc.joblib')

# @st.cache_data
# def load_data():
#     return joblib.load('npar2/X_train.joblib')

# rfc = load_model()
# X_train = load_data()

# # ===========================
# # 🔥 SỬA LỖI EXPLAINER Ở ĐÂY  
# # ===========================
# explainer = shap.Explainer(
#     model=rfc,
#     masker=shap.maskers.Independent(X_train),
#     feature_names=X_train.columns
# )

# st.title('AP Predict: ML Mortality Prediction in Acute Pancreatitis')

# col1, col2 = st.columns(2)

# vars = {}
# with col1:
#     vars['pt_max'] = st.number_input('PT', 8.8, 148.7, step=0.1)
#     vars['bilirubin_total_max'] = st.number_input('Bilirubin total', -23.2, 51.2, step=0.1)
#     vars['bun_max'] = st.number_input('BUN', 3, 181, step=1)
#     vars['rdw_max'] = st.number_input('RDW', 11.8, 34.9, step=0.1)
#     vars['NPAR'] = st.number_input('NPAR', 1.36, 71.5, step=0.1)
#     vars['sapsii'] = st.number_input('SAP SII', 6, 94, step=1)

# with col2:
#     vars['sofa'] = st.number_input('SOFA', 0, 21, step=1)
#     vars['cci'] = st.number_input('CCI', 0, 17, step=1)
#     vars['apsiii'] = st.number_input('APSIII', 7, 159, step=1)
#     vars['temperature_mean'] = st.number_input('Temperature body', 33.6, 40.1, step=0.1)
#     vars['vasopressin'] = st.selectbox('Vasopressin', [(0, 'No'), (1, 'Yes')], format_func=lambda v: v[1])[0]
#     vars['has_sepsis'] = st.selectbox('Has sepsis', [(0, 'No'), (1, 'Yes')], format_func=lambda v: v[1])[0]

# arr = ['Survival', 'Died']

# if st.button('Predict'):
#     df_pred = pd.DataFrame([vars])
#     st.write("🔍 Input:", df_pred)

#     pred = rfc.predict(df_pred)[0]
#     prob = rfc.predict_proba(df_pred)[0]

#     st.write(f"### 🧮 Predict: **{arr[pred]}**, Probability: **{prob[pred]:.2f}**")

#     # ===========================
#     # 🔥 SHAP CHÍNH XÁC
#     # ===========================
#     shap_values = explainer(df_pred)

#     # Force plot
#     st.write("### 🔥 SHAP Force Plot")
#     fig = shap.plots.force(shap_values[0], matplotlib=True)
#     st.pyplot(fig)

#     # Waterfall
#     st.write("### 💧 SHAP Waterfall Plot")
#     fig2, ax = plt.subplots()
#     shap.plots.waterfall(shap_values[0], show=False)
#     st.pyplot(fig2)


import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="AP Predict – Mortality ML Model",
    page_icon="🧠",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("npar2/rfc.joblib")

@st.cache_data
def load_data():
    return joblib.load("npar2/X_train.joblib")

rfc = load_model()
X_train = load_data()

explainer = shap.Explainer(
    model=rfc,
    masker=shap.maskers.Independent(X_train),
    feature_names=X_train.columns
)


st.title("🧠 **AP Predict**")
st.subheader("Machine Learning-Based Mortality Prediction in Acute Pancreatitis")

st.markdown("---")


st.markdown("### 📝 **Patient Clinical Input**")
st.markdown(
    """
    Nhập các chỉ số lâm sàng quan trọng.  
    Những biến này phản ánh **mức độ viêm, chức năng gan–thận, nguy cơ shock và suy đa cơ quan**,  
    giúp mô hình dự đoán chính xác nguy cơ tử vong.
    """
)

col1, col2 = st.columns(2)
vars = {}

with col1:
    vars["pt_max"] = st.number_input("PT (Đông máu)", 8.8, 148.7, step=0.1,
                                     help="PT cao → rối loạn đông máu, dấu hiệu nhiễm trùng nặng hoặc suy gan.")
    vars["bilirubin_total_max"] = st.number_input("Total Bilirubin (Chức năng gan)", -23.2, 51.2, step=0.1,
                                     help="Bilirubin tăng → tắc mật hoặc tổn thương gan, thường gặp trong viêm tụy do sỏi mật.")
    vars["bun_max"] = st.number_input("BUN (Thận / Mất nước)", 3, 181, step=1,
                                     help="BUN cao → mất nước, giảm tưới máu thận hoặc suy thận cấp.")
    vars["rdw_max"] = st.number_input("RDW (Tình trạng viêm)", 11.8, 34.9, step=0.1,
                                     help="RDW cao liên quan viêm hệ thống và tiên lượng tử vong trong ICU.")
    vars["NPAR"] = st.number_input("NPAR (Chỉ số viêm)", 1.36, 71.5, step=0.1,
                                     help="NPAR = neutrophil% / albumin, phản ánh mức độ viêm nặng và tổn thương cơ quan.")
    vars["sapsii"] = st.number_input("SAPS II (Điểm nặng toàn thân)", 6, 94, step=1,
                                     help="Thang điểm ICU dự đoán tử vong dựa trên sinh hiệu và xét nghiệm.")

with col2:
    vars["sofa"] = st.number_input("SOFA (Suy đa cơ quan)", 0, 21, step=1,
                                     help="SOFA cao → suy hô hấp, gan, thận… Dùng để đánh giá sepsis.")
    vars["cci"] = st.number_input("CCI (Bệnh nền)", 0, 17, step=1,
                                     help="CCI cao → bệnh nền nặng, nguy cơ tử vong cao hơn.")
    vars["apsiii"] = st.number_input("APS III (Mức độ nặng ICU)", 7, 159, step=1,
                                     help="Điểm đánh giá toàn trạng ngay khi nhập ICU.")
    vars["temperature_mean"] = st.number_input("Body Temperature", 33.6, 40.1, step=0.1,
                                     help="Sốt → nhiễm trùng. Hạ thân nhiệt → shock tuần hoàn.")
    vars["vasopressin"] = int(st.selectbox("Vasopressin", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No"))
    vars["has_sepsis"] = int(st.selectbox("Sepsis", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No"))


arr = ["Survival", "Died"]

st.markdown(
    """
    ---
    ### 🧬 **Tại sao mô hình cần những chỉ số này?**
    Các biến này đại diện cho:
    - 🔥 Mức độ viêm & nhiễm trùng  
    - 🩸 Đông máu & chức năng gan  
    - 💧 Tình trạng thận & mất nước  
    - ⚠️ Mức độ shock & nhu cầu vận mạch  
    - 🧠 Suy đa cơ quan (SOFA, APS III, SAPS II)  
    - 🏥 Bệnh nền (CCI)

    → Tất cả giúp mô hình đánh giá chính xác nguy cơ tử vong.
    """
)

st.markdown("---")

# ======================
# BUTTON PREDICT
# ======================
if st.button("🚀 **Run Prediction**"):

    df_pred = pd.DataFrame([vars])
    df_pred = df_pred.reindex(columns=rfc.feature_names_in_, fill_value=0)

    pred = rfc.predict(df_pred)[0]
    prob = rfc.predict_proba(df_pred)[0][pred]

    # Display prediction in a card
    st.markdown("### 🧮 **Prediction Result**")

    if pred == 1:
        st.error(f"### ⚠️ Predicted: **Died**  \n**Probability: {prob:.2f}**")
    else:
        st.success(f"### ✅ Predicted: **Survival**  \n**Probability: {prob:.2f}**")

    st.markdown("### 📋 **Input Summary**")
    st.dataframe(df_pred)

    st.markdown("---")

    # ======================
    # SHAP SECTION
    # ======================
    st.markdown("## 🔍 **Explainability with SHAP**")
    shap_values = explainer(df_pred)

    # Tabs for SHAP
    tab1, tab2 = st.tabs(["🔥 Decision Plot", "💧 Waterfall Plot"])

    # ---- DECISION PLOT ----
    with tab1:
        st.write("### 🔥 SHAP Decision Plot")
        shap_vals_pos = shap_values.values[:, :, 1]
        expected_val = explainer.expected_value[1]

        fig_dec = plt.figure()
        shap.decision_plot(
            expected_val,
            shap_vals_pos,
            df_pred.columns.tolist(),
            show=False
        )
        st.pyplot(fig_dec)

    # ---- WATERFALL PLOT ----
    with tab2:
        st.write("### 💧 SHAP Waterfall Plot")

        shap_water = shap.Explanation(
            values=shap_values.values[0, :, 1],
            base_values=explainer.expected_value[1],
            data=df_pred.iloc[0].values,
            feature_names=df_pred.columns
        )

        fig2 = plt.figure()
        shap.plots.waterfall(shap_water, show=False)
        st.pyplot(fig2)

