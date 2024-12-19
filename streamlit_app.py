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
    
    # Definir a variável target 'price' e as features
    Target = df['price']  # A variável alvo 'price'
    Features = df.drop(columns=["price", "Predicted"])  # As features (sem a coluna 'price' e 'Predicted')

# Barra lateral para escolher a feature
with st.sidebar:
    st.header('Input Features')
    selected_feature = st.selectbox('Select a feature to analyze', Features.columns)

# Carregar previsões (isso deve ser feito previamente com o seu modelo, mas vamos gerar previsões aqui)
df['Predicted'] = df['Predicted']  # Garantir que a coluna de previsões esteja presente

# Criar um DataFrame com preço real e previsão
df_analysis = Features.copy()
df_analysis['price'] = Target
df_analysis['Predicted'] = df['Predicted']  # Substitua com a coluna de previsões do seu CSV

# Calcular a diferença percentual entre o valor previsto e o valor real
df_analysis['percentage_diff'] = 100 * (df_analysis['Predicted'] - df_analysis['price']) / df_analysis['price']

# Adicionar a feature selecionada ao DataFrame
df_analysis['selected_feature'] = df_analysis[selected_feature]

# Exibir os resultados
st.write(f"Analisando a diferença para a feature: {selected_feature}")

# Criar o gráfico de barras para o preço médio por valor da feature selecionada
with st.expander(f"📊 Média do Preço por {selected_feature}", expanded=False):
    # Agrupar os dados pela feature selecionada e calcular a média do preço
    avg_price = df_analysis.groupby('selected_feature')['price'].mean().reset_index()
    
    # Criar o gráfico de barras
    plt.figure(figsize=(10,6))
    plt.bar(avg_price['selected_feature'], avg_price['price'], color='skyblue')

    # Adicionar título e rótulos aos eixos
    plt.title(f'Média do Preço por {selected_feature}')
    plt.xlabel(selected_feature)
    plt.ylabel('Média do Preço')
    
    # Exibir o gráfico no Streamlit
    st.pyplot(plt)
