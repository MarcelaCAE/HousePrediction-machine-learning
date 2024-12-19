import streamlit as st
import pandas as pd

# Main Title of the Application
st.title('🎈 HousePrediction - Machine Learning')

st.info('This is a machine learning model to predict house prices.')

# Section: Dataset Overview and Description
with st.expander('📄 Dataset Overview and Description', expanded=True):
    # Display the original dataset first
    st.markdown('#### Original Dataset')
    url = 'https://raw.githubusercontent.com/MarcelaCAE/HousePrediction-machine-learning/refs/heads/master/king_%20country_%20houses_aa.csv'
    df = pd.read_csv(url)

    st.dataframe(df.head())  # Interactive display of the first few rows

    # Additional dataset details
    st.write(f'**Number of Rows and Columns:** {df.shape[0]} rows and {df.shape[1]} columns.')

    # Description of the dataset columns
    st.markdown('#### Columns Description')
    st.markdown("""
    - **id**: Unique numeric identifier for each house.
    - **date**: Date of house sale.
    - **price**: House price (target variable).
    - **bedrooms**: Number of bedrooms in the house.
    - **bathrooms**: Number of bathrooms in the house.
    - **sqft_living**: Living area size in square feet.
    - **sqft_lot**: Lot size in square feet.
    - **floors**: Number of floors (levels) in the house.
    - **waterfront**: Waterfront view (0 = no, 1 = yes).
    - **view**: If the house has been viewed (0 = no, 1 = yes).
    - **condition**: Overall condition of the house (scale 1–5).
    - **grade**: Overall grade of the house (scale 1–11).
    - **sqft_above**: Square footage above ground level.
    - **sqft_basement**: Square footage of the basement.
    - **yr_built**: Year the house was built.
    - **yr_renovated**: Year the house was renovated.
    - **zipcode**: Zipcode of the house location.
    - **lat**: Latitude of the house location.
    - **long**: Longitude of the house location.
    - **sqft_living15**: Living room area in 2015 (post-renovations).
    - **sqft_lot15**: Lot size area in 2015 (post-renovations).

    ##### **Dataset Source:** 
    [King County Houses Dataset on Kaggle](https://www.kaggle.com/datasets/minasameh55/king-country-houses-aa)
    """)
