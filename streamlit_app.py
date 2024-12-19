import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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
import streamlit as st
import pandas as pd

# Exemplo: Carregar dados fictícios (substitua pelos seus dados reais)
# Features: DataFrame com variáveis independentes
# Target: Série ou coluna com os valores reais
# df: DataFrame com as previsões (incluindo a coluna 'Predicted')
# Certifique-se de que a coluna 'date_month' já exista no Features.

# Adicionar barra lateral para selecionar a feature
with st.sidebar:
    st.header('Input Features')
    selected_feature = st.selectbox('Select a feature to analyze', Features.columns)

# Adicionar previsões (garanta que a coluna 'Predicted' exista em df)
df['Predicted'] = df['Predicted']

# Criar o DataFrame de análise
df_analysis = Features.copy()
df_analysis['price'] = Target  # Valores reais
df_analysis['Predicted'] = df['Predicted']  # Valores previstos

# Calcular a diferença percentual
df_analysis['percentage_diff'] = 100 * (df_analysis['Predicted'] - df_analysis['price']) / df_analysis['price']

# Adicionar a feature selecionada ao DataFrame
df_analysis['selected_feature'] = df_analysis[selected_feature]

# Exibir os resultados na interface do Streamlit
st.write(f"Analisando a diferença para a feature: {selected_feature}")

# Exibir o DataFrame de forma interativa
st.dataframe(df_analysis[['date_month', selected_feature, 'price', 'Predicted', 'percentage_diff']])


