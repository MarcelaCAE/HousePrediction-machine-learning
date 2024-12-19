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

with st.expander('📄 Features', expanded=True):
    Features.head(30)

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

# Adicionar o nome da feature selecionada ao DataFrame
df_analysis['selected_feature'] = df_analysis[selected_feature]

grouped = df_analysis.groupby(['selected_feature', 'date_month']).mean()

# Calculating percentage changes and growth percentages
grouped['price_pct_change'] = grouped.groupby(level=0)['price'].pct_change() * 100
grouped['Predicted_pct_change'] = grouped.groupby(level=0)['Predicted'].pct_change() * 100
grouped['growth_percentage'] = (
    (grouped['Predicted'] - grouped['price']) / grouped['price']
) * 100

# Resetting the index for better readability
grouped_reset = grouped.reset_index()

# Displaying the data in Streamlit
st.title("Analysis by Selected Feature")
st.write("### Grouped Data")
st.dataframe(grouped_reset)

# Transposing the data (optional)
if st.checkbox("Transpose DataFrame"):
    st.write(grouped_reset.T)
