import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Main Title of the Application
st.title('🎈 HousePrediction - Machine Learning')

st.info('This is a machine learning model to predict house prices.')

# Expander para Data Understanding
with st.expander('📄 Data Understanding', expanded=True):
    # Carregar o dataset
    url = 'https://raw.githubusercontent.com/MarcelaCAE/HousePrediction-machine-learning/refs/heads/master/king_%20country_%20houses_aa.csv'
    df = pd.read_csv(url)

    # Exibição do dataset
    st.markdown('#### Original Dataset')
    st.dataframe(df.head(10))  # Visualização interativa das primeiras linhas

    # Informações sobre o dataset
    st.write(f'**Número de Linhas e Colunas:** {df.shape[0]} linhas e {df.shape[1]} colunas.')

    # Descrição das colunas
    st.markdown('#### Descrição das Colunas')
    st.markdown("""
    - **id**: Identificador único de cada casa.
    - **date**: Data de venda da casa.
    - **price**: Preço da casa (variável alvo).
    - **bedrooms**: Número de quartos.
    - **bathrooms**: Número de banheiros.
    - **sqft_living**: Área útil em metros quadrados.
    - **sqft_lot**: Tamanho do terreno em metros quadrados.
    - **floors**: Número de andares.
    - **waterfront**: Se a casa tem vista para o mar (0 = não, 1 = sim).
    - **view**: Se a casa foi vista (0 = não, 1 = sim).
    - **condition**: Condição geral da casa (escala de 1 a 5).
    - **grade**: Avaliação geral da casa (escala de 1 a 11).
    - **sqft_above**: Área acima do solo.
    - **sqft_basement**: Área do porão.
    - **yr_built**: Ano de construção.
    - **yr_renovated**: Ano da última reforma.
    - **zipcode**: Código postal.
    - **lat**: Latitude.
    - **long**: Longitude.
    - **sqft_living15**: Área útil da casa nos últimos 15 anos.
    - **sqft_lot15**: Tamanho do terreno nos últimos 15 anos.
    """)

    # Estatísticas descritivas
    st.markdown('#### Estatísticas Descritivas')
    st.write(df.describe())

    # Verificar valores ausentes
    st.markdown('#### Verificar Valores Ausentes')
    st.write(df.isnull().sum())

    # Visualizações
    st.markdown('#### Visualização do Preço das Casas por Número de Quartos')
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='bedrooms', y='price')
    st.pyplot(plt)

    st.markdown('#### Visualização de Correlation Matrix')
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    st.pyplot(plt)

# Expander para Data Modeling
with st.expander('🛠️ Data Modeling', expanded=True):
    # Definir variáveis independentes e dependentes
    X = df.drop(columns=['id', 'date', 'price'])
    y = df['price']

    # Separar os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalizar os dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Treinar o modelo XGBoost
    model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=5)
    model.fit(X_train_scaled, y_train)

    # Prever os valores no conjunto de teste
    y_pred = model.predict(X_test_scaled)

    # Avaliação do modelo
    st.markdown('#### Performance do Modelo')
    st.write(f'**Mean Squared Error (MSE):** {mean_squared_error(y_test, y_pred):.2f}')
    st.write(f'**Mean Absolute Error (MAE):** {mean_absolute_error(y_test, y_pred):.2f}')
    st.write(f'**R² Score:** {r2_score(y_test, y_pred):.2f}')

    # Visualização dos erros de previsão
    st.markdown('#### Erros de Previsão')
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue')
    plt.xlabel('Valores Reais')
    plt.ylabel('Valores Previstos')
    plt.title('Comparação entre Valores Reais e Previstos')
    st.pyplot(plt)

    # Importância das features
    st.markdown('#### Importância das Features')
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(model, importance_type='weight', max_num_features=10, title='Top 10 Features Importantes')
    st.pyplot(plt)

