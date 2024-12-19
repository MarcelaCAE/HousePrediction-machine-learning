import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar o dataset
st.title('🎈 HousePrediction-machine-learning')
st.info('This is a machine learning model')

# Carregar o dataset
with st.expander('Dataset'):
    st.write('**Raw Data**')
    df = pd.read_csv('https://raw.githubusercontent.com/MarcelaCAE/HousePrediction-machine-learning/refs/heads/master/king_%20country_%20houses_aa.csv')
    st.write(df.head())  # Exibe as primeiras linhas do dataset

# Função para limpar os dados
def clean_data(data):
    df = data.copy()
    df.columns = [column.lower().replace(" ", "_") for column in df.columns]  # Padroniza os nomes das colunas
    st.write(f'Rows with missing values: {df.isna().any(axis=1).sum()}')
    st.write(f'Duplicate rows: {df[df.duplicated()].shape[0]}')
    st.write("Cleaned DataFrame preview:")
    st.write(df.head())
    return df

# Função para exibir informações sobre o DataFrame
def info_about_dataframe(df):
    st.write("\n### Informações sobre o DataFrame:")
    st.write(f"Shape do DataFrame: {df.shape}")
    st.write(f"Tipos de dados:\n{df.dtypes}")
    st.write("\nInformações detalhadas sobre o DataFrame:")
    st.write(df.info())

# Função para exibir as estatísticas descritivas e IQR
def descriptive_statistics(df):
    st.write("### Descriptive Statistics, IQR, and Outliers...")
    
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    desc_stats = df[numerical_cols].describe().T
    
    iqr_values = {}
    outlier_counts = {}
    
    for col in numerical_cols:
        q1 = df[col].quantile(0.25)  
        q3 = df[col].quantile(0.75)  
        iqr = q3 - q1
        iqr_values[col] = iqr
        
        lower_limit = q1 - 1.5 * iqr
        upper_limit = q3 + 1.5 * iqr
        
        outliers = df[(df[col] < lower_limit) | (df[col] > upper_limit)][col]
        outlier_counts[col] = len(outliers)
    
    desc_stats['IQR'] = desc_stats.index.map(iqr_values)
    desc_stats['Outliers'] = desc_stats.index.map(outlier_counts)
    
    st.write(round(desc_stats, 2))

# Função para explorar distribuições
def exploration(df):
    st.write("### Distributions of Features")
    color = '#18354f'  # Cor para as barras do histograma
    nrows, ncols = 5, 4  # Tamanho da grade para os subgráficos
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 16))
    
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i >= len(numeric_cols):
            ax.set_visible(False)  
            continue
        
        ax.hist(df[numeric_cols[i]], bins=30, color=color, edgecolor='black')
        ax.set_title(numeric_cols[i]) 
    plt.tight_layout()
    st.pyplot(fig)  # Passar fig para evitar erro

# Função para explorar o target (preço)
def explorate_target(df):
    st.write("### Distribution of the Target Variable (Price)")
    color = '#18354f'
    fig, ax = plt.subplots()  # Criar uma figura antes de passar para o st.pyplot
    sns.kdeplot(df["price"], color=color, ax=ax)
    st.pyplot(fig)  # Passar fig para evitar erro

# Função para exibir os outliers usando boxplot
def exploration_outliers(df):
    st.write("### Boxplots to Identify Outliers")
    color = '#18354f'  # Cor para os boxplots
    nrows, ncols = 5, 4  # Tamanho da grade para os subgráficos
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 16))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i >= len(numeric_cols):
            ax.set_visible(False)  # Esconde os subgráficos não utilizados
            continue
        
        # Criar o boxplot para a coluna
        ax.boxplot(df[numeric_cols[i]].dropna(), vert=False, patch_artist=True, 
                boxprops=dict(facecolor=color, color='black'), 
                medianprops=dict(color='yellow'), whiskerprops=dict(color='black'), 
                capprops=dict(color='black'), flierprops=dict(marker='o', color='red', markersize=5))
        ax.set_title(numeric_cols[i], fontsize=10)  # Definir o título como o nome da coluna
        ax.tick_params(axis='x', labelsize=8)  # Ajustar os rótulos do eixo x para melhor visibilidade
    
    # Ajustar layout para melhor espaçamento
    plt.tight_layout()
    st.pyplot(fig)  # Passar fig para evitar erro

# Correlation Matrix
def correlation_matrix(df):
    st.write("### Correlation Matrix")
    
    # Remover ou preencher valores NaN antes de calcular a correlação
    df_clean = df.select_dtypes(include=['float64', 'int64']).dropna()  # Remove colunas não numéricas e NaNs
    corr = df_clean.corr(method='pearson').round(2)
    
    fig, ax = plt.subplots(figsize=(14, 10))  # Criar figura para o gráfico
    sns.set_style("white")
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, cmap=sns.diverging_palette(230, 30, as_cmap=True), 
                vmin=-1, vmax=1, center=0, annot_kws={"fontsize": 8}, ax=ax)
    
    st.pyplot(fig)  # Passar fig para evitar erro

# Secções principais
with st.expander('Data Understanding'):
    st.header('Data Understanding')

    # Exibir os dados limpos
    with st.expander('Data Cleaning'):
        cleaned_df = clean_data(df)

    # Exibir as informações sobre o dataframe
    with st.expander('Info About DataFrame'):
        info_about_dataframe(df)

    # Exibir estatísticas descritivas
    with st.expander('Descriptive Statistics'):
        descriptive_statistics(df)

    # Explorar as distribuições das variáveis
    with st.expander('Exploration'):
        exploration(df)

    # Explorar a distribuição do target (preço)
    with st.expander('Exploration of Target Variable'):
        explorate_target(df)

    # Exibir os outliers
    with st.expander('Exploration of Outliers'):
        exploration_outliers(df)

    # Exibir a matriz de correlação
    with st.expander('Correlation Matrix'):
        correlation_matrix(df)
