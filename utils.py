import pickle
import pandas as pd


# --------------------------------------------------
# LOAD MODEL, SCALER AND ENCODERS
# --------------------------------------------------

def load_model():

    with open("food_delivery_model.pkl", "rb") as file:
        model = pickle.load(file)

    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    with open("encoder.pkl", "rb") as file:
        encoder = pickle.load(file)

    return model, scaler, encoder


# --------------------------------------------------
# VALIDATE USER INPUT
# --------------------------------------------------

def validate_inputs(distance, preparation, experience):

    if distance <= 0:
        return False, "Please enter a valid Distance."

    if preparation <= 0:
        return False, "Preparation Time must be greater than 0."

    if experience < 0:
        return False, "Courier Experience cannot be negative."

    return True, ""


# --------------------------------------------------
# ENCODE CATEGORICAL FEATURES
# --------------------------------------------------

def encode_inputs(weather,
                  traffic,
                  vehicle,
                  time_of_day,
                  encoder):

    weather_encoded = encoder["Weather"].transform([weather])[0]

    traffic_encoded = encoder["Traffic_Level"].transform([traffic])[0]

    vehicle_encoded = encoder["Vehicle_Type"].transform([vehicle])[0]

    time_encoded = encoder["Time_of_Day"].transform([time_of_day])[0]

    return (
        weather_encoded,
        traffic_encoded,
        vehicle_encoded,
        time_encoded
    )


# --------------------------------------------------
# CREATE INPUT DATAFRAME
# --------------------------------------------------

def create_dataframe(distance,
                     weather,
                     traffic,
                     time_of_day,
                     vehicle,
                     preparation,
                     experience):

    data = pd.DataFrame({

        "Distance_km": [distance],

        "Weather": [weather],

        "Traffic_Level": [traffic],

        "Time_of_Day": [time_of_day],

        "Vehicle_Type": [vehicle],

        "Preparation_Time_min": [preparation],

        "Courier_Experience_yrs": [experience]

    })

    return data


# --------------------------------------------------
# SCALE INPUT DATA
# --------------------------------------------------

def scale_data(data, scaler):

    return scaler.transform(data)


# --------------------------------------------------
# PREDICT DELIVERY TIME
# --------------------------------------------------

def predict_delivery_time(model, scaled_data):

    prediction = model.predict(scaled_data)

    return prediction[0]


# --------------------------------------------------
# CREATE SUMMARY TABLE
# --------------------------------------------------

def create_summary(distance,
                   weather,
                   traffic,
                   vehicle,
                   time_of_day,
                   preparation,
                   experience):

    summary = pd.DataFrame({

        "Feature": [

            "📍 Distance",

            "🌦 Weather",

            "🚦 Traffic Level",

            "🚗 Vehicle Type",

            "🕒 Time of Day",

            "👨‍🍳 Preparation Time",

            "👨‍💼 Courier Experience"

        ],

        "Value": [

            f"{distance} km",

            weather,

            traffic,

            vehicle,

            time_of_day,

            f"{preparation} min",

            f"{experience} years"

        ]

    })

    return summary