import streamlit as st
import pandas as pd

# Main Title of the Application
st.title('🎈 HousePrediction - Machine Learning')

st.info('This is a machine learning model to predict house prices.')

# Section: Dataset Overview and Description
with st.expander('📄 Dataset Overview and Description', expanded=True):
    st.subheader('Key Variables')
    st.markdown("""
    The following dataset consists of **21 columns** across **21,613 rows**. The majority of our data types are numeric, including **15 integer** and **5 float** columns.
    
    ##### **Columns Description:**
    
    - **id**: The unique numeric identifier assigned to each house being sold.
    - **date**: The date on which the house was sold.
    - **price**: The price of the house (target variable for prediction).
    - **bedrooms**: The number of bedrooms in the house.
    - **bathrooms**: The number of bathrooms in the house.
    - **sqft_living**: The size of the house in square feet.
    - **sqft_lot**: The size of the lot in square feet.
    - **floors**: The number of floors (levels) in the house.
    - **waterfront**: Indicates whether the house has a waterfront view (0 = no, 1 = yes).
    - **view**: Indicates whether the house has been viewed (0 = no, 1 = yes).
    - **condition**: The overall condition of the house, rated on a scale from 1 to 5.
    - **grade**: The overall grade given to the house, based on the King County grading system, rated from 1 to 11.
    - **sqft_above**: The square footage of the house excluding the basement.
    - **sqft_basement**: The square footage of the basement.
    - **yr_built**: The year the house was built.
    - **yr_renovated**: The year the house was renovated (if applicable).
    - **zipcode**: The zipcode of the house’s location.
    - **lat**: The latitude of the house’s location.
    - **long**: The longitude of the house’s location.
    - **sqft_living15**: The living room area in 2015 (post-renovations).
    - **sqft_lot15**: The lot size area in 2015 (post-renovations).
    
    ##### **Dataset Source:** 
    [King County Houses Dataset on Kaggle](https://www.kaggle.com/datasets/minasameh55/king-country-houses-aa)
    """)

    st.subheader('Original Dataset')
    # Load the dataset
    url = 'https://raw.githubusercontent.com/MarcelaCAE/HousePrediction-machine-learning/refs/heads/master/king_%20country_%20houses_aa.csv'
    df = pd.read_csv(url)
    
    # Display the first few rows of the dataset
    st.write('This is the original dataset used in the model:')
    st.dataframe(df.head())  # Interactive display of the first few rows

    # Additional dataset details
    st.write(f'**Number of Rows and Columns:** {df.shape[0]} rows and {df.shape[1]} columns.')
    st.write(f'**Dataset Columns:** {", ".join(df.columns)}')
