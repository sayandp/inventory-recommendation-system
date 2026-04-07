# 📦 Smart Inventory Recommendation System

## 🚀 Overview  
This project is an end-to-end inventory recommendation system that predicts product demand and suggests optimal stock levels. It simulates how retail businesses make data-driven inventory decisions using historical sales patterns.

---

## 🎯 What this project does  
- Predicts future sales for each product  
- Recommends how much stock to maintain  
- Uses recent sales behavior to capture demand trends  
- Helps reduce stockouts and overstock  

---

## 🧠 How it works  
The system follows a structured pipeline:

1. Data preprocessing  
- Cleans raw retail transaction data  
- Removes invalid entries (returns, missing values)  
- Aggregates daily sales per product  

2. Feature engineering  
- Creates time-based features (day, month)  
- Builds lag features (recent sales history)  
- Adds UK public holiday indicators  

Weather data was tested but removed from the final model because it showed low importance and did not improve predictions.

3. Model training  
- Trains a regression model to forecast demand  
- Learns patterns from historical sales trends  

4. Prediction system  
- Outputs expected demand as a range instead of a single value  
- Generates a recommended stock level with a safety buffer  

---

## ⚙️ Pipeline & Workflow  
This project includes a structured machine learning workflow:

- Data preprocessing and feature engineering pipeline  
- Model training and evaluation  
- Model persistence (saving trained model)  
- Reproducible execution using scripts  

This reflects a foundational step toward production-ready systems.

---

## 📊 Key Features  
- Product-level demand forecasting  
- Real-world input design (recent sales instead of technical terms)  
- Business-friendly outputs (ranges instead of decimals)  
- Interactive Streamlit dashboard  

---

## 🌍 Data Context  
- Based on UK retail sales data  
- Incorporates UK public holidays  

Results may vary depending on data quality, missing values, and seasonal patterns.

---

## ▶️ How to run  

1. Install dependencies  
pip install -r requirements.txt  

2. Add dataset  
Download dataset from:  
KAGGLE(online_retail.csv)

Place it in:  
data/raw/online_retail.csv  

3. Train the model  
python main.py  

4. Run the app  
streamlit run app.py  

---

## 📦 Output  
- Expected Sales → shown as a realistic range  
- Recommended Stock → includes safety buffer  

---

## 🔮 Future Improvements  
- Real-time data integration  
- Multi-product demand modeling  
- Improved uncertainty estimation  
- Deployment as a full web application  

---

## 📌 Note  
- Data and trained model files are not included in this repository  
- Results may differ depending on the dataset and preprocessing  
- This project is built for educational and demonstration purposes  

---

## 👤 Author  
sayand p 

Qualification: BSC APPLIED STATITICS WITH DATA SCINCE
               PG DIPLOMA IN DATA SCIENCE AND ANALYTICS

---
