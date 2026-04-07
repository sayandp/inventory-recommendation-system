import streamlit as st
import pandas as pd
import joblib

from src.models.inventory import recommend_stock


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Inventory AI",
    page_icon="📦",
    layout="wide"
)

# =========================
# CUSTOM PREMIUM CSS
# =========================
st.markdown("""
<style>
.main {
    background-color: #0e1117;
    color: white;
}

h1, h2, h3 {
    color: #00f5d4;
}

.stButton>button {
    background: linear-gradient(90deg, #00f5d4, #00bbf9);
    color: black;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}

.metric-card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# =========================
# LOAD MODEL + MAPPING
# =========================
model = joblib.load("models/model.pkl")
mapping = joblib.load("models/item_mapping.pkl")
reverse_mapping = {v: k for k, v in mapping.items()}


# =========================
# HEADER
# =========================
st.title("📦 Smart Inventory System")
st.markdown("### AI-powered demand forecasting & stock optimization")

st.divider()


# =========================
# SIDEBAR INPUTS
# =========================
st.sidebar.header("🛒 Product Selection")

item_name = st.sidebar.selectbox("Select Product", list(mapping.values()))
item_id = reverse_mapping[item_name]


# =========================
# DATE INPUT
# =========================
st.sidebar.header("📅 Date Info")

selected_date = st.sidebar.date_input("Select Date")

day_of_week = selected_date.weekday()
month = selected_date.month
week_of_year = selected_date.isocalendar()[1]


# =========================
# SALES INPUT
# =========================
st.sidebar.header("📊 Recent Sales")

sales_yesterday = st.sidebar.number_input("Sales Yesterday", value=20)
sales_last_7_days = st.sidebar.number_input("Total Sales Last 7 Days", value=140)
sales_last_14_days = st.sidebar.number_input("Total Sales Last 14 Days", value=280)
sales_last_28_days = st.sidebar.number_input("Total Sales Last 28 Days", value=560)


# =========================
# HOLIDAY FIX
# =========================
holiday_option = st.sidebar.selectbox("Is Holiday?", ["No", "Yes"])
is_holiday = 1 if holiday_option == "Yes" else 0


# =========================
# FEATURE ENGINEERING
# =========================
lag_1 = sales_yesterday
lag_7 = sales_last_7_days / 7
lag_14 = sales_last_14_days / 14
lag_28 = sales_last_28_days / 28

rolling_mean_7 = lag_7
rolling_mean_28 = lag_28


# =========================
# CREATE INPUT DATA
# =========================
input_data = pd.DataFrame({
    "item_id": [item_id],
    "day_of_week": [day_of_week],
    "week_of_year": [week_of_year],
    "month": [month],
    "lag_1": [lag_1],
    "lag_7": [lag_7],
    "lag_14": [lag_14],
    "lag_28": [lag_28],
    "rolling_mean_7": [rolling_mean_7],
    "rolling_mean_28": [rolling_mean_28],
    "is_holiday": [is_holiday]
})


# =========================
# FEATURE ORDER
# =========================
FEATURE_ORDER = [
    "item_id",
    "day_of_week", "week_of_year", "month",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "is_holiday"
]

input_data = input_data[FEATURE_ORDER]


# =========================
# MAIN LAYOUT
# =========================
col1, col2 = st.columns([1, 1])


with col1:
    st.subheader("📥 Input Summary")
    st.markdown(f"**🛒 Product:** {item_name}")
    st.markdown(f"**📅 Date:** {selected_date}")
    st.markdown(f"**🎉 Holiday:** {holiday_option}")
    st.dataframe(input_data)


with col2:
    st.subheader("🚀 Prediction")

    if st.button("Predict Demand & Stock"):

        prediction = model.predict(input_data)[0]

        # 🔥 RANGE OUTPUT
        lower = int(prediction * 0.9)
        upper = int(prediction * 1.1)

        recommended = int(round(recommend_stock([prediction])[0]))

        st.success("Prediction Generated!")

        colA, colB = st.columns(2)

        with colA:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📊 Expected Sales", f"{lower} - {upper}")
            st.markdown('</div>', unsafe_allow_html=True)

        with colB:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📦 Recommended Stock", f"{recommended}")
            st.markdown('</div>', unsafe_allow_html=True)


# =========================
# FOOTER
# =========================
# =========================
# 🔍 MODEL INSIGHTS (CLEAR)
# =========================
st.divider()
st.markdown("### 🔍 What This Model Does")

st.markdown("""
This system predicts **future product demand** based on recent sales patterns.

### 📊 How it works:
- Uses recent sales (yesterday, last 7, 14, 28 days)
- Captures demand trends and seasonality
- Considers calendar effects (day, month, holidays)

### 📦 What you get:
- **Expected Sales Range** → Estimated demand for the next period  
- **Recommended Stock** → Suggested inventory including safety buffer  

### 🎯 Business Value:
- Helps avoid stockouts  
- Reduces overstock  
- Supports data-driven inventory decisions  
""")