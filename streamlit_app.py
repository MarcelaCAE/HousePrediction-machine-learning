import streamlit as st
import pandas as pd

st.title('🎈 HousePrediction-machine-learning')

st.info('This is a machine learning model')

with st.expander('Dataset'):
  st.write('**Raw data**')
  df = pd.read_csv('https://raw.githubusercontent.com/MarcelaCAE/HousePrediction-machine-learning/refs/heads/master/king_%20country_%20houses_aa.csv')
  df
