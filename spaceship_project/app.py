import streamlit as st
import pandas as pd
import joblib

pipeline = joblib.load("models/pipeline.pkl")

st.title("ASG 04 MD - Constantine Amadio - Spaceship Titanic Model Deployment")
st.header("Passenger Information")

HomePlanet = st.selectbox("HomePlanet", ["Earth", "Europa", "Mars"])
CryoSleep = st.selectbox("CryoSleep", [True, False])
Destination = st.selectbox("Destination", ["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"])
Age = st.number_input("Age", value=30)
VIP = st.selectbox("VIP", [True, False])

RoomService = st.number_input("RoomService", value=0.0)
FoodCourt = st.number_input("FoodCourt", value=0.0)
ShoppingMall = st.number_input("ShoppingMall", value=0.0)
Spa = st.number_input("Spa", value=0.0)
VRDeck = st.number_input("VRDeck", value=0.0)

CabinDeck = st.selectbox("Cabin Deck", ["A","B","C","D","E","F","G","T"])
CabinNum = st.number_input("Cabin Number", value=100)
CabinSide = st.selectbox("Cabin Side", ["P","S"])

if st.button("Predict"):

    input_data = pd.DataFrame([{
        "HomePlanet": HomePlanet,
        "CryoSleep": CryoSleep,
        "Destination": Destination,
        "Age": Age,
        "VIP": VIP,
        "RoomService": RoomService,
        "FoodCourt": FoodCourt,
        "ShoppingMall": ShoppingMall,
        "Spa": Spa,
        "VRDeck": VRDeck,
        "CabinDeck": CabinDeck,
        "CabinNum": CabinNum,
        "CabinSide": CabinSide
    }])

    prediction = pipeline.predict(input_data)

    if prediction[0]:
        st.success("Passenger was Transported")
    else:
        st.error("Passenger was NOT Transported")