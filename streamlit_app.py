import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Main Title of the Application
st.title('🎈 Machine Learning')

st.info('This is a machine learning model used to predict house prices.')

# Section: Dataset Overview (Everything Inside This Expander)
with st.expander('📄 Data', expanded=True):
    # Display the original dataset first
    st.markdown('#### Raw Data')
    url = 'https://raw.githubusercontent.com/MarcelaCAE/HousePrediction-machine-learning/refs/heads/master/model_best_final.csv'
    df = pd.read_csv(url)
    st.dataframe(df.head(20))
    
    # Definir a variável target 'price' e as features
    Target = df['price']  # A variável alvo 'price'
    Features = df.drop(columns=["price", "Predicted"])  # As features (sem a coluna 'price' e 'Predicted')

with st.expander('📄 Features', expanded=True):
    st.dataframe(Features.head(30))

# Assuming `Features`, `Target`, and `df` are already loaded
# Create the DataFrame for analysis
df_analysis = Features.copy()
df_analysis['price'] = Target  # Actual values
df_analysis['Predicted'] = df['Predicted']  # Predicted values


import streamlit as st
import pandas as pd

# Assuming `Features`, `Target`, and `df` are already loaded and `df_analysis` is created

with st.expander('📄 Model Insigths', expanded=True):
    # Agrupar diretamente por 'date_month' e calcular a média de 'price' e 'Predicted'
    grouped = df_analysis.groupby('date_month')[['price', 'Predicted']].mean()

    # Calcular a variação percentual para 'price' e 'Predicted'
    grouped['price_pct_change'] = grouped['price'].pct_change() * 100
    grouped['Predicted_pct_change'] = grouped['Predicted'].pct_change() * 100

    # Resetar o índice para facilitar a visualização
    grouped_reset = grouped.reset_index()

    # Exibir os dados no Streamlit
    st.title("Trend Price Analysis")
    st.write("### Grouped per month")
    st.dataframe(grouped_reset[['date_month', 'price', 'Predicted', 'price_pct_change', 'Predicted_pct_change']])

# Transpor os dados (opcional)
if st.checkbox("Transpose DataFrame"):
    st.write(grouped_reset[['date_month', 'price', 'Predicted', 'price_pct_change', 'Predicted_pct_change']].T)


with st.expander('📄 Data Visualization', expanded=True):

# Gráfico de Variação Percentual (price_pct_change e Predicted_pct_change)
    st.write("### Graph Variation Actual vs Predicted price")
    fig2, ax2 = plt.subplots(figsize=(10, 6))  

# Plotando as variações percentuais de preço real e previsto
    ax2.plot(grouped_reset['date_month'], grouped_reset['price_pct_change'], label='Variation % actual price ', color='blue', marker='o')
    ax2.plot(grouped_reset['date_month'], grouped_reset['Predicted_pct_change'], label='Variation % prediction price', color='orange', marker='o')

# Adicionando título e rótulos
    ax2.set_title('Variation Percentage actual price vs predicted over time ', fontsize=14)
    ax2.set_xlabel('Month', fontsize=12)
    ax2.set_ylabel('Variation Percentage', fontsize=12)

# Adicionando a legenda
    ax2.legend()

# Exibir o gráfico no Streamlit
    st.pyplot(fig2)
