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

# Adicionar a coluna 'Predicted' e 'date_month' ao DataFrame
df['date_month'] = df['date_month']  # Certificando-se que a coluna 'date_month' está no DataFrame

# Adicionar a feature selecionada ao DataFrame
df['selected_feature'] = df[selected_feature]

# Agrupar os dados por 'date_month' e calcular a média de 'price', 'Predicted' e da feature selecionada
monthly_avg = df.groupby('date_month')[['price', 'Predicted', 'selected_feature']].mean().reset_index()

# Exibir os resultados
st.write(f"Analisando a média dos preços reais, previstos e a feature selecionada por mês")

# Criar o gráfico de linha para as médias mensais de preço, preço previsto e a feature selecionada
with st.expander(f"📊 Média Mensal do Preço Real, Preço Previsto e {selected_feature}", expanded=True):
    plt.figure(figsize=(12,6))
    
    # Plotando as linhas para preço real, preço previsto e a feature selecionada por mês
    plt.plot(monthly_avg['date_month'], monthly_avg['price'], label='Average Price', color='skyblue', marker='o')
    plt.plot(monthly_avg['date_month'], monthly_avg['Predicted'], label='Average Predicted Price', color='salmon', marker='o')
    plt.plot(monthly_avg['date_month'], monthly_avg['selected_feature'], label=f'Average {selected_feature}', color='green', marker='o')

    # Adicionar título e rótulos aos eixos
    plt.title(f'Média do Preço Real, Preço Previsto e {selected_feature} por Mês')
    plt.xlabel('Mês')
    plt.ylabel('Valor Médio')
    plt.xticks(rotation=45)
    plt.legend()

    # Exibir o gráfico no Streamlit
    st.pyplot(plt)


    # Exibir o gráfico no Streamlit
    st.pyplot(plt)
