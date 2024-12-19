import streamlit as st
import pandas as pd

# Configuração do título principal da aplicação
st.title('🎈 HousePrediction - Machine Learning')

st.info('Este é um modelo de aprendizado de máquina para prever preços de casas.')

# Primeira seção: Exibindo o dataset original
with st.expander('📊 Visualizar Dataset Original', expanded=True):  
    st.subheader('Dataset Original')
    # Carregar o dataset
    url = 'https://raw.githubusercontent.com/MarcelaCAE/HousePrediction-machine-learning/refs/heads/master/king_%20country_%20houses_aa.csv'
    df = pd.read_csv(url)
    
    # Exibir os primeiros registros do dataset
    st.write('Este é o dataset original usado no modelo:')
    st.dataframe(df.head())  # Mostra os primeiros registros de forma interativa

    # Informações adicionais do dataset
    st.write(f'**Número de Linhas e Colunas:** {df.shape[0]} linhas e {df.shape[1]} colunas.')
    st.write(f'**Colunas do Dataset:** {", ".join(df.columns)}')
