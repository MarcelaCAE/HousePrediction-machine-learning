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

# Sidebar for user input
st.sidebar.title("Dynamic Feature Selection")
selected_feature = st.sidebar.selectbox(
    "Select a feature to group by:",
    df_analysis.columns,  # Assuming all columns are valid; adjust as needed
)

# Add the selected feature to the DataFrame
df_analysis['selected_feature'] = df_analysis[selected_feature]

import streamlit as st
import pandas as pd

# Assuming `Features`, `Target`, and `df` are already loaded and `df_analysis` is created

with st.expander('📄 Features', expanded=True):
    # Agrupar diretamente por 'date_month' e calcular a média de 'price' e 'Predicted'
    grouped = df_analysis.groupby('date_month')[['price', 'Predicted']].mean()

    # Calcular a variação percentual para 'price' e 'Predicted'
    grouped['price_pct_change'] = grouped['price'].pct_change() * 100
    grouped['Predicted_pct_change'] = grouped['Predicted'].pct_change() * 100

    # Resetar o índice para facilitar a visualização
    grouped_reset = grouped.reset_index()

    # Exibir os dados no Streamlit
    st.title("Análise por Mês")
    st.write("### Dados Agrupados por Mês")
    st.dataframe(grouped_reset[['date_month', 'price', 'Predicted', 'price_pct_change', 'Predicted_pct_change']])

# Transpor os dados (opcional)
if st.checkbox("Transpor DataFrame"):
    st.write(grouped_reset[['date_month', 'price', 'Predicted', 'price_pct_change', 'Predicted_pct_change']].T)
