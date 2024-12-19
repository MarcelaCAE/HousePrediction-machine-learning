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



# Adicionar barra lateral para selecionar a feature
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

# Adicionar o nome da feature selecionada ao DataFrame
df_analysis['selected_feature'] = df_analysis[selected_feature]

# Extrair o mês da coluna `date_month` para agrupar
df_analysis['month'] = df_analysis['date_month'].dt.month  # Extrai o número do mês

# Realizar o agrupamento por `month` e calcular as médias
df_grouped = (
    df_analysis.groupby('month')
    .agg(
        avg_price=('price', 'mean'),
        avg_predicted=('Predicted', 'mean'),
        avg_selected_feature=('selected_feature', 'mean')
    )
    .reset_index()
)

# Adicionar o nome da feature selecionada ao DataFrame agrupado
df_grouped['feature_name'] = selected_feature

# Exibir os resultados na interface do Streamlit
st.write(f"Análise agregada por mês para a feature: {selected_feature}")
st.dataframe(df_grouped)
