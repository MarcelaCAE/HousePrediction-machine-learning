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
    
    # Definir a variável target 'price' e as features
    Target = df['price']  # A variável alvo 'price'
    Features = df.drop(columns=["price", "Predicted"])  # As features (sem a coluna 'price' e 'Predicted')

# Barra lateral para escolher a feature
with st.sidebar:
    st.header('Input Features')
    selected_feature = st.selectbox('Select a feature to analyze', Features.columns)

# Carregar previsões (isso deve ser feito previamente com o seu modelo, mas vamos gerar previsões aqui)
# Assumindo que as previsões já estão no seu CSV
# A coluna de previsões é chamada 'Predicted' no seu CSV, então a linha correta seria:
df['Predicted'] = df['Predicted']  # Apenas para garantir que a coluna esteja presente

# Criar um DataFrame com a feature selecionada, preço real e previsão
df_analysis = Features.copy()
df_analysis['price'] = Target
df_analysis['Predicted'] = df['Predicted']  # Substitua com a coluna de previsões do seu CSV

# Calcular a diferença percentual entre o preço real e o previsto
df_analysis['percentage_diff'] = 100 * abs(df_analysis['price'] - df_analysis['Predicted']) / df_analysis['price']

# Adicionar a feature selecionada ao DataFrame
df_analysis['selected_feature'] = df_analysis[selected_feature]

# Filtrar o DataFrame com a feature selecionada
df_selected = df_analysis[['price', 'Predicted', 'percentage_diff', 'selected_feature']]

# Exibir os resultados
st.write(f"Analisando a feature: {selected_feature}")
st.write(df_selected.head())


