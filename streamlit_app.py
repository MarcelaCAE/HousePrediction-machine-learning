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


# Section: Dataset Overview (Everything Inside This Expander)
with st.expander("Data Modeling", expanded=True):
    st.write("""
    Aqui, vamos realizar o treinamento de modelos de aprendizado de máquina para prever o preço de casas com base em várias variáveis.
    """)

    # Copiar o dataframe para a modelagem
    df_machine_learning = df.copy()
    
    # Verificar se há valores ausentes
    st.write("Verificando valores ausentes no DataFrame:")
    st.write(df_machine_learning.isnull().sum())
    
    # Tratar valores ausentes - Exemplo: substituir por média (dependendo do tipo de variável, você pode escolher outro método)
    df_machine_learning.fillna(df_machine_learning.mean(), inplace=True)
    
    # Verificar se há valores ausentes após o tratamento
    st.write("Valores ausentes após tratamento:")
    st.write(df_machine_learning.isnull().sum())
    
    # Separando variáveis dependente (target) e independentes (features)
    y = df_machine_learning["price"]
    X = df_machine_learning.drop(columns=["price"]) 

    # Se houver variáveis categóricas, é necessário convertê-las para variáveis numéricas
    X = pd.get_dummies(X, drop_first=True)  # Usando one-hot encoding para variáveis categóricas
    
    # Verificando tipos de dados
    st.write("Tipos de dados das variáveis:")
    st.write(X.dtypes)
    
    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    st.write(f'100% dos dados: {len(df)}.')
    st.write(f'70% para treino: {len(X_train)}.')
    st.write(f'30% para teste: {len(X_test)}.')
    
    # Modelo de regressão linear
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Previsões
    predictions = model.predict(X_test)
    
    # Criar um DataFrame para avaliar a diferença entre valores reais e previstos
    eval_df = pd.DataFrame({'actual': y_test, 'pred': predictions})
    eval_df["difference"] = round(abs(eval_df["actual"] - eval_df["pred"]), 2)
    st.write("Diferenças entre valores reais e previstos (exemplo):")
    st.write(eval_df.head())
    
    # Modelos para avaliar
    results = {}
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'Decision Tree': DecisionTreeRegressor(),
        'KNN': KNeighborsRegressor(),
        'XGBoost': xgb.XGBRegressor()
    }

    # Avaliação dos modelos
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        MSE = mean_squared_error(y_test, predictions)
        RMSE = np.sqrt(MSE)
        r2 = r2_score(y_test, predictions)
        MAE = mean_absolute_error(y_test, predictions)
        
        results[model_name] = {
            'R²': r2,
            'RMSE': RMSE,
            'MSE': MSE,
            'MAE': MAE
        }

    # Criar DataFrame de resultados e exibir no Streamlit
    results_df_ml = pd.DataFrame(results).T
    results_df_ml = results_df_ml.round(2)
    
    st.write("Resultados de Avaliação dos Modelos:")
    st.write(results_df_ml)
