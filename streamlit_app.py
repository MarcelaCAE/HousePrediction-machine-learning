import streamlit as st
import pandas as pd
st.title('🎈 HousePrediction-machine-learning')

st.info('This app builds a machine learning app')

api = KaggleApi()
api.authenticate()
dataset_name = "minasameh55/king-country-houses-aa"
url = f'https://www.kaggle.com/api/v1/datasets/download/{dataset_name}'
response = requests.get(url)
zipped_file = io.BytesIO(response.content)

with zipfile.ZipFile(zipped_file, 'r') as zip_ref: 
    file_name = zip_ref.namelist()[0]  
    with zip_ref.open(file_name) as file:
        data = pd.read_csv(file)
data.head()
