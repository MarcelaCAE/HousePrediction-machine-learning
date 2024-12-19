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

# Adicionar a coluna date_month ao DataFrame
df_analysis['date_month'] = df['date_month']  # Utilizando a coluna 'date_month' existente

# Exibir os resultados
st.write(f"Analisando a média dos preços reais e previstos por mês")

# Agrupar os dados por mês e calcular a média do preço real e do preço previsto
monthly_avg = df_analysis.groupby('date_month')[['price', 'Predicted']].mean().reset_index()

# Criar o gráfico de barras para as médias mensais de preço e preço previsto
with st.expander(f"📊 Média Mensal do Preço Real e Preço Previsto", expanded=False):
    plt.figure(figsize=(12,6))
    
    # Plotando a média de preço real e previsto por mês
    plt.bar(monthly_avg['date_month'], monthly_avg['price'], label='Average Price', color='skyblue', alpha=0.7)
    plt.bar(monthly_avg['date_month'], monthly_avg['Predicted'], label='Average Predicted Price', color='salmon', alpha=0.7, width=0.4)

    # Adicionar título e rótulos aos eixos
    plt.title(f'Média do Preço Real e Preço Previsto por Mês')
    plt.xlabel('Mês')
    plt.ylabel('Preço Médio')
    plt.xticks(rotation=45)
    plt.legend()

    # Exibir o gráfico no Streamlit
    st.pyplot(plt)
