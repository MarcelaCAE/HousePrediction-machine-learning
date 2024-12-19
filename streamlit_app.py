import streamlit as st
import pandas as pd
import xgboost as xgb

# Main Title of the Application
st.title('🎈 HousePrediction - Machine Learning')

st.info('This is a machine learning model to predict house prices.')

# Section: Dataset Overview (Everything Inside This Expander)
with st.expander('📄 Data', expanded=True):
    # Display the original dataset first
    st.markdown('#### Raw Data')
    url = 'https://raw.githubusercontent.com/MarcelaCAE/HousePrediction-machine-learning/refs/heads/master/model_best_final.csv'
    df = pd.read_csv(url)
    df.head(20)
    
    # Definir a variável target 'price' e as features
    Target = df['price']  # A variável alvo 'price'
    Features = df.drop(columns=["price", "Predicted"])  # As features (sem a coluna 'price' e 'Predicted')

# Barra lateral para escolher a feature
with st.sidebar:
    st.header('Input Features')
    selected_feature = st.selectbox('Select a feature to analyze', Features.columns)

# Carregar previsões (isso deve ser feito previamente com o seu modelo, mas vamos gerar previsões aqui)
# Assumindo que as previsões já estão no seu CSV
df['Predicted'] = df['Predicted']  # Garantir que a coluna de previsões esteja presente

# Criar um DataFrame com preço real e previsão
df_analysis = Features.copy()
df_analysis['price'] = Target
df_analysis['Predicted'] = df['Predicted']  # Substitua com a coluna de previsões do seu CSV

# Calcular a diferença percentual entre o valor previsto e o valor real
df_analysis['percentage_diff'] = 100 * (df_analysis['Predicted'] - df_analysis['price']) / df_analysis['price']

# Adicionar a feature selecionada ao DataFrame
df_analysis['selected_feature'] = df_analysis[selected_feature]

# Excluir a feature selecionada do DataFrame
df_selected = df_analysis[['price', 'Predicted', 'percentage_diff', 'selected_feature']]

# Exibir os resultados
st.write(f"Analisando a diferença para a feature: {selected_feature}")
st.write(df_selected.head())

with st.expander(f"📈 Visualização: Price vs Predicted para {selected_feature}", expanded=False):
    plt.figure(figsize=(10,6))
    plt.plot(df_analysis['selected_feature'], df_analysis['price'], label='Price', color='blue', marker='o', linestyle='-', alpha=0.7)
    plt.plot(df_analysis['selected_feature'], df_analysis['Predicted'], label='Predicted', color='red', marker='x', linestyle='--', alpha=0.7)

    # Adicionar título e rótulos aos eixos
    plt.title(f'Price vs Predicted - Feature: {selected_feature}')
    plt.xlabel(selected_feature)
    plt.ylabel('Price')
    plt.legend()

    # Exibir o gráfico no Streamlit
    st.pyplot(plt)

