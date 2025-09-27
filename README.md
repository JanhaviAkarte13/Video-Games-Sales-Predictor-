# 🎮 Video Game Sales Predictor

---

## 📌 Project Overview
The **Video Game Sales Predictor** is a  project designed to analyze historical market dynamics and forecast the sales potential of video games.  

By processing key features like **Genre, Platform, and Publisher** from a comprehensive dataset, the system predicts a game's **global sales units**.  

To ensure prediction reliability, the project implements multiple machine learning models including:  
- 🔹 **Linear Regression (Baseline Model)**  
- 🔹 **Random Forest (Ensemble Model)**  

The system is deployed via a **Flask web interface**, where users can input game attributes and instantly compare predictions from both models.  

✅ Demonstrates a complete, production-ready ML pipeline  
✅ Provides data-driven insights into historical trends  
✅ Compares performance of regression models  

---

## ✨ Features
- 💰 Predicts global video game sales based on historical data  
- 💻 User-friendly **Flask web interface**  
- 📊 Compares multiple ML models (Linear Regression, Random Forest)  
- 🔎 Sales visualization of historical trends (e.g., peak years, genres)  
- ⚡ Easy to deploy and run locally  

---

## 🛠 Tech Stack
**Programming Language**: Python  

**Libraries & Tools**:  
- 📚 Pandas, NumPy, Scikit-learn (modeling & analysis)  
- 📊 Matplotlib, Seaborn (visualization)  
- 🌐 Flask (deployment)  

---

## 📂 Dataset
- **Source**: Kaggle → https://www.kaggle.com/datasets/gregorut/videogamesales  
- **Timeframe**: Physical sales data from ~1980 to 2016  
- **Features**:  
  - Platform  
  - Genre  
  - Publisher  
  - Global Sales (M) *(Target Variable)*  

⚠️ *Note: Predictions and visualizations are based only on historical **physical sales data**. Current trends like digital, mobile, and subscription services are not included.*  

---

## ➡️ Project Workflow
➡️ Data Preparation (Feature Engineering)
➡️ Model Training (2 Models)
➡️ Model Evaluation
➡️ Deployment (Flask Web App)


---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/JanhaviAkarte13/Video-Games-Sales-Predictor.git
cd Video-Games-Sales-Predictor

2. Install Dependencies

(Recommended: use a virtual environment)

pip install Flask pandas scikit-learn joblib numpy matplotlib seaborn
