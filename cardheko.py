import streamlit as st
import pandas as pd
import pickle
from PIL import Image

# Load the trained model
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the encoder mappings (LabelEncoder objects)
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Define the expected feature names based on your model
feature_columns = [
    'Body type',
    'Number of previous owners',
    'Seats',
    'city',
    'kilometer driven',
    'Car model',
    'Year of car manufacture',
    'Length',
    'Fuel type',
    'Width',
    'Color',
    'Brand Name',
    'Insurance Validity_value',
    'Transmission type',
    'Engine Displacement_value(cc)'
]

# Title of the app
st.title("CAR DHEKO APP")

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to:", ("HOME", "MAIN MENU"))

if page == "HOME":
    image = Image.open(r'F:\NITHYA ONLINE DATA SCIENCE\CAPSTONE-3 CAR DECO\CarDekho-FY22-social.jpg')
    st.image(image, use_column_width=True)
    st.header("Welcome to CAR DHEKO - Used Car Price Prediction!")

elif page == "MAIN MENU":

    # Function to get user input
    def get_user_input():
        cols = st.columns(3)  # Create 3 columns for a grid layout
        
        # Row 1
        body_type = cols[0].selectbox("Body type", label_encoders['Body type'].classes_)
        num_owners = cols[1].number_input("Number of previous owners", min_value=0, max_value=10)
        seats = cols[2].number_input("Seats", min_value=2, max_value=7)

        # Row 2
        city = cols[0].selectbox("City", label_encoders['city'].classes_)
        kilometers_driven = cols[1].number_input("Kilometers driven", min_value=0, max_value=200000)
        car_model = cols[2].selectbox("Car model", label_encoders['Car model'].classes_)

        # Row 3
        year_of_manufacture = cols[0].number_input("Year of car manufacture", min_value=2000, max_value=2024)
        length = cols[1].number_input("Length (mm)", min_value=2000, max_value=6000)
        fuel_type = cols[2].selectbox("Fuel type", label_encoders['Fuel type'].classes_)

        # Row 4
        width = cols[0].number_input("Width (mm)", min_value=1000, max_value=2500)
        color = cols[1].selectbox("Color", label_encoders['Color'].classes_)
        brand_name = cols[2].selectbox("Brand Name", label_encoders['Brand Name'].classes_)

        # Row 5
        insurance_validity = cols[0].selectbox("Insurance Validity_value", label_encoders['Insurance Validity_value'].classes_)
        transmission_type = cols[1].selectbox("Transmission type", label_encoders['Transmission type'].classes_)
        engine_displacement = cols[2].number_input("Engine Displacement_value(cc)", min_value=500, max_value=5000)

        features = pd.DataFrame([[body_type, num_owners, seats, city, kilometers_driven, car_model,
                                  year_of_manufacture, length, fuel_type, width, color, brand_name,
                                  insurance_validity, transmission_type, engine_displacement]], 
                                columns=feature_columns)
        return features

    # Get user input
    user_features = get_user_input()

    # Display user input
    st.write("### User Input Features:")
    st.write(user_features)

    # Create columns for prediction and result display
    col1, col2 = st.columns([5, 11]) 

    with col1:
        # Display a "Predict Price" button
        if st.button('**Predict Price**'):
            with st.spinner("Predicting..."):
                try:
                    # Encode the categorical features
                    for column in ['Body type', 'city', 'Car model', 'Fuel type', 'Color', 'Brand Name', 'Insurance Validity_value', 'Transmission type']:
                        if column in user_features.columns:
                            user_features[column] = label_encoders[column].transform(user_features[column])

                    # Make the prediction
                    prediction = model.predict(user_features)

                    # Display the result in the right-side column
                    with col2:
                        st.success(f'Estimated Car Price: ***₹ {prediction[0]:,.2f}***')

                except ValueError as ve:
                    st.error(f"Value error: {ve}. Please check your input.")
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
