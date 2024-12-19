import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor

# Main Title of the Application
st.title('🎈 HousePrediction - Machine Learning')

st.info('This is a machine learning model to predict house prices.')


# Section: Dataset Overview (Everything Inside This Expander)
with st.expander('📄 Data', expanded=True):
    # Display the original dataset first
    st.markdown('#### Raw Data')
    url = 'https://raw.githubusercontent.com/MarcelaCAE/HousePrediction-machine-learning/refs/heads/master/model_best_final.csv'
    df = pd.read_csv(url)
    
    Target = df_machine_learning['price']  # A variável alvo 'price'
    Features = df_machine_learning.drop(columns=["price"])  # As features


# Barra lateral para escolher a feature
with st.sidebar:
    st.header('Input Features')
    selected_feature = st.selectbox('Select a feature to analyze', Features.columns)

# Criar um DataFrame com a feature selecionada, preço e previsão
df_analysis = Features.copy()
df_analysis['price'] = Target
df_analysis['predicted'] = predictions

# Calcular a diferença percentual entre o preço real e o previsto
df_analysis['percentage_diff'] = 100 * abs(df_analysis['price'] - df_analysis['predicted']) / df_analysis['price']

# Filtrar o DataFrame com a feature selecionada
df_selected = df_analysis[['price', 'predicted', 'percentage_diff', selected_feature]]

# Exibir os resultados
st.write(f"Analisando a feature: {selected_feature}")
st.write(df_selected.head())

