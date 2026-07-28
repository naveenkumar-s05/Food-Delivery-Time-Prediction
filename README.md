# 🚚 Food Delivery Time Prediction

## 📌 Project Overview

Food Delivery Time Prediction is a Machine Learning web application that predicts the estimated delivery time of a food order based on several factors such as distance, weather conditions, traffic level, preparation time, vehicle type, courier experience, and time of day.

The application is built using Python, Scikit-learn, and Streamlit to provide real-time predictions through an interactive user interface.

---

## ✨ Features

- Predicts food delivery time instantly
- Interactive and user-friendly Streamlit interface
- Machine Learning model using Gradient Boosting Regressor
- Input validation for better user experience
- Supports multiple weather conditions
- Supports different traffic levels
- Real-time prediction results

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

---

## 📊 Machine Learning Workflow

1. Data Collection
2. Data Preprocessing
3. Missing Value Handling
4. Label Encoding
5. Feature Scaling
6. Model Training
7. Model Evaluation
8. Model Deployment using Streamlit

---

## 📂 Project Structure

```
Food_Delivery_Time_Prediction/

│── app.py
│── utils.py
│── food_delivery_model.pkl
│── scaler.pkl
│── encoder.pkl
│── requirements.txt
│── README.md
```

---

## 🚀 How to Run the Project

### Clone the Repository

```bash
git clone https://github.com/naveenkumar-s05/Food-Delivery-Time-Prediction.git
```

### Navigate to the Project Folder

```bash
cd Food-Delivery-Time-Prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

---

## 📥 Input Features

- Distance (km)
- Weather
- Traffic Level
- Time of Day
- Vehicle Type
- Preparation Time (minutes)
- Courier Experience (years)

---

## 📤 Output

Estimated Food Delivery Time (minutes)

---

## 👨‍💻 Author

**Naveen Kumar**

Machine Learning & Python Enthusiast