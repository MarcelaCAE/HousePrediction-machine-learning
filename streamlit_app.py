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

# Criar um DataFrame com preço real e previsão
df_analysis = Features.copy()
df_analysis['price'] = Target
df_analysis['Predicted'] = df['Predicted']  # Substitua com a coluna de previsões do seu CSV

# Adicionar a coluna date_month ao DataFrame
df_analysis['date_month'] = df['date_month']  # Utilizando a coluna 'date_month' existente

# Adicionar a feature selecionada ao DataFrame
df_analysis['selected_feature'] = df_analysis[selected_feature]

# Agrupar os dados por mês e calcular a média do preço real, do preço previsto e da feature selecionada
monthly_avg = df_analysis.groupby('date_month')[['price', 'Predicted', 'selected_feature']].mean().reset_index()

# Exibir os resultados
st.write(f"Analisando a média dos preços reais, previstos e a feature selecionada por mês")

# Criar o gráfico de linha para as médias mensais de preço e preço previsto
with st.expander(f"📊 Média Mensal do Preço Real e Preço Previsto", expanded=False):
    plt.figure(figsize=(12,6))
    
    # Plotando as linhas para preço real e previsto por mês
    plt.plot(monthly_avg['date_month'], monthly_avg['price'], label='Average Price', color='skyblue', marker='o')
    plt.plot(monthly_avg['date_month'], monthly_avg['Predicted'], label='Average Predicted Price', color='salmon', marker='o')

    # Adicionar título e rótulos aos eixos
    plt.title(f'Média do Preço Real e Preço Previsto por Mês')
    plt.xlabel('Mês')
    plt.ylabel('Preço Médio')
    plt.xticks(rotation=45)
    plt.legend()

    # Exibir o gráfico no Streamlit
    st.pyplot(plt)
