import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st

model = pk.load(open('model.pkl', 'rb'))

# Add CSS for styling
st.markdown(
    """
    <style>
        .header-container {
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px; /* Add space between border and header */
        }
        .header-text {
            text-align: center;
            font-size: 60px;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Create a container without a border
st.markdown('<div class="header-container">', unsafe_allow_html=True)

# Add header text inside the container
st.markdown('<h1 class="header-text">Car Price Predictor</h1>', unsafe_allow_html=True)

# Close the container div
st.markdown('</div>', unsafe_allow_html=True)

# Load the local image from the "images" folder
banner_image_path = 'images/carbrand.png'
st.image(banner_image_path, use_column_width=True)

cars_data = pd.read_csv('Cardetails.csv')

def get_brand_name(car_name):
    car_name = car_name.split(' ')[0]
    return car_name.strip()

cars_data['name'] = cars_data['name'].apply(get_brand_name)

name = st.selectbox('Select Car Brand', cars_data['name'].unique())
year = st.slider('Car Manufactured Year', 1994, 2024)
km_driven = st.slider('No of kms Driven', 11, 200000)
fuel = st.selectbox('Fuel type', cars_data['fuel'].unique())
seller_type = st.selectbox('Seller  type', cars_data['seller_type'].unique())
transmission = st.selectbox('Transmission type', cars_data['transmission'].unique())
owner = st.selectbox('Seller  type', cars_data['owner'].unique())
mileage = st.slider('Car Mileage', 10, 40)
engine = st.slider('Engine CC', 700, 5000)
max_power = st.slider('Max Power', 0, 200)
seats = st.slider('No of Seats', 5, 10)

# Apply styling to the button for centering
st.markdown('<div class="center-button">', unsafe_allow_html=True)

if st.button("Predict"):
    input_data_model = pd.DataFrame(
        [[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
        columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine',
                 'max_power', 'seats'])

    input_data_model['owner'].replace(['First Owner', 'Second Owner', 'Third Owner',
                                       'Fourth & Above Owner', 'Test Drive Car'],
                                      [1, 2, 3, 4, 5], inplace=True)
    input_data_model['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4], inplace=True)
    input_data_model['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3], inplace=True)
    input_data_model['transmission'].replace(['Manual', 'Automatic'], [1, 2], inplace=True)
    input_data_model['name'].replace(['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
                                      'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
                                      'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
                                      'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
                                      'Ambassador', 'Ashok', 'Isuzu', 'Opel'],
                                     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                      23, 24, 25, 26, 27, 28, 29, 30, 31]
                                     , inplace=True)

    car_price = model.predict(input_data_model)

    # Add a green rectangular box for the result
    st.markdown('<div class="green-rect">', unsafe_allow_html=True)
    st.markdown('Car Price is going to be = ₹' + str(car_price[0]))
    st.markdown('</div>', unsafe_allow_html=True)

# Close the styling div
st.markdown('</div>', unsafe_allow_html=True)
