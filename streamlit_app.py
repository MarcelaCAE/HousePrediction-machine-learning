import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor

# Main Title of the Application
st.title('🎈 HousePrediction - Machine Learning')

st.info('This is a machine learning model to predict house prices.')




# Section: Dataset Overview (Everything Inside This Expander)
with st.expander('📄 Data Understading', expanded=True):
    # Display the original dataset first
    st.markdown('#### Original Dataset')
    url = 'https://raw.githubusercontent.com/MarcelaCAE/HousePrediction-machine-learning/refs/heads/master/king_%20country_%20houses_aa.csv'
    df = pd.read_csv(url)

    st.dataframe(df.head(10))  # Interactive display of the first few rows

    # Additional dataset details
    st.write(f'**Number of Rows and Columns:** {df.shape[0]} rows and {df.shape[1]} columns.')

    # Description of the dataset columns
    st.markdown('#### Columns Description')
    st.markdown("""
    - **id**: Unique numeric identifier for each house.
    - **date**: Date of house sale.
    - **price**: House price (target variable).
    - **bedrooms**: Number of bedrooms in the house.
    - **bathrooms**: Number of bathrooms in the house.
    - **sqft_living**: Living area size in square feet.
    - **sqft_lot**: Lot size in square feet.
    - **floors**: Number of floors (levels) in the house.
    - **waterfront**: Waterfront view (0 = no, 1 = yes).
    - **view**: If the house has been viewed (0 = no, 1 = yes).
    - **condition**: Overall condition of the house (scale 1–5).
    - **grade**: Overall grade of the house (scale 1–11).
    - **sqft_above**: Square footage above ground level.
    - **sqft_basement**: Square footage of the basement.
    - **yr_built**: Year the house was built.
    - **yr_renovated**: Year the house was renovated.
    - **zipcode**: Zipcode of the house location.
    - **lat**: Latitude of the house location.
    - **long**: Longitude of the house location.
    - **sqft_living15**: Living room area in 2015 (post-renovations).
    - **sqft_lot15**: Lot size area in 2015 (post-renovations).

    ##### **Dataset Source:** 
    [King County Houses Dataset on Kaggle](https://www.kaggle.com/datasets/minasameh55/king-country-houses-aa)
    """)

    # Data Cleaning Section
    st.markdown("### 🧹 Data Cleaning")
    def clean_data(data):
        df = data.copy()
        data.columns = [column.lower().replace(" ", "_") for column in data.columns]  # Standardizing column names
        st.write("Rows with missing values:", df.isna().any(axis=1).sum())
        st.write("Duplicate rows:", df[df.duplicated()].shape[0])
        return df

    # Clean data section
    df_cleaned = clean_data(df)
    st.dataframe(df_cleaned.head(10))  # Display the cleaned data preview

    # Converting the Date to Datetime
    df_cleaned['date'] = pd.to_datetime(df_cleaned['date'])

    # Descriptive Statistics Section
    st.markdown("### 📊 Descriptive Statistics")
    def descriptive_statistics(df):
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
        
        return round(desc_stats, 2)

    # Display descriptive statistics
    stats = descriptive_statistics(df_cleaned)
    st.write(stats)

    # Feature Exploration Section (Visualizations)
    st.markdown("### 📈 Feature Exploration")
    def exploration(df):
        color = '#18354f'  # Color for the histograms
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        nrows, ncols = 5, 4
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 16))
        
        axes = axes.flatten()
        for i, ax in enumerate(axes):
            if i >= len(numeric_cols):
                ax.set_visible(False)  # Hide unused subplots
                continue
            ax.hist(df[numeric_cols[i]], bins=30, color=color, edgecolor='black')
            ax.set_title(numeric_cols[i])
        
        plt.tight_layout()
        st.pyplot(fig)  # Pass the figure explicitly

    # Display feature exploration plots
    exploration(df_cleaned)

    # Target Variable Exploration
    st.markdown("### 🔍 Target Variable Exploration")
    def explore_target(df):
        color = '#18354f'
        fig, ax = plt.subplots(figsize=(8, 6))  # Explicitly create a figure and axis
        sns.kdeplot(df["price"], color=color, ax=ax)
        st.pyplot(fig)  # Pass the figure explicitly

    # Display target variable exploration plot
    explore_target(df_cleaned)

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['date_year'] = df['date'].dt.year
    df['date_month'] = df['date'].dt.month
    df['date_quarter'] = df['date'].dt.quarter
    df['date_day'] = df['date'].dt.day

# Reorganizar a coluna 'price'
    price = df.pop("price")
    df["price"] = price


    # Correlation Matrix - Display inside the expander
    st.markdown("### 🔗 Correlation Matrix")
    st.markdown("**Visualizing the correlation between features and the target...**")
    corr = df_cleaned.corr(method='pearson').round(2)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(14, 10))  # Explicitly create a figure and axis for the heatmap
    sns.heatmap(corr, mask=mask, annot=True, cmap=sns.diverging_palette(230, 30, as_cmap=True), 
                vmin=-1, vmax=1, center=0, annot_kws={"fontsize": 8}, ax=ax)
    st.pyplot(fig)  # Pass the figure explicitly

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['date_year'] = df['date'].dt.year
df['date_month'] = df['date'].dt.month
df['date_quarter'] = df['date'].dt.quarter
df['date_day'] = df['date'].dt.day

price = df.pop("price")
df["price"] = price
df = df.drop(['id', 'date'], axis=1)

# Supondo que você já tenha o dataframe df carregado com as colunas necessárias

with st.expander("Data Modeling", expanded=True):
    st.write("Data Modeling used")
   

    # Copiar o dataframe para a modelagem
    df_machine_learning = df.copy()
    
    # Exibir as primeiras 10 linhas do DataFrame
    st.write(df_machine_learning.head(10))

  

# Definir a variável target e as features
    y = df_machine_learning["price"]
    X = df_machine_learning.drop(columns=["price"])

# Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Exibir o tamanho dos dados de treino e teste
    st.markdown(f'**100% of our data:** {len(df_machine_learning)}')
    st.markdown(f'**70% for training data:** {len(X_train)}')
    st.markdown(f'**30% for test data:** {len(X_test)}')

# Inicializar e treinar o modelo
    model = LinearRegression()
    model.fit(X_train, y_train)

# Realizar as previsões
    predictions = model.predict(X_test)

# Criar um DataFrame de avaliação com os resultados
    eval_df = pd.DataFrame({"actual": y_test, "pred": predictions})
    eval_df = eval_df.round()
    eval_df["difference"] = round(abs(eval_df["actual"] - eval_df["pred"]), 2)

# Exibir as primeiras 10 linhas do DataFrame de avaliação no Streamlit
    st.write("### Evaluation Results (First 10 rows):")
    st.dataframe(eval_df.head(10))  # Mostra as 10 primeiras linhas

# Exibir a diferença entre o valor real e o previsto
    st.write(f"### Difference between Actual and Predicted Prices (rounded):")
    st.dataframe(eval_df[["actual", "pred", "difference"]])

# Analisar se a diferença é grande
    st.markdown("### Analysis of the Prediction Differences")
    max_diff = eval_df["difference"].max()
    min_diff = eval_df["difference"].min()
    st.write(f"The largest difference between actual and predicted price is: {max_diff}")
    st.write(f"The smallest difference between actual and predicted price is: {min_diff}")

# Modelos a serem usados
    models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'Decision Tree': DecisionTreeRegressor(),
    'XGBoost': xgb.XGBRegressor()
}

# Preparando o dicionário de resultados
    results = {}

# Loop para treinar e avaliar os modelos
    for model_name, model in models.items():
       model.fit(X_train, y_train)
       predictions = model.predict(X_test)
    
    MSE = mean_squared_error(y_test, predictions)
    RMSE = np.sqrt(MSE)
    r2 = r2_score(y_test, predictions)
    MAE = mean_absolute_error(y_test, predictions)
    
    results[model_name] = {
        'R²': r2,
        'RMSE': RMSE,
        'MSE': MSE,
        'MAE': MAE
    }

# Convertendo os resultados em um DataFrame
     results_df_ml = pd.DataFrame(results).T
     results_df_ml = results_df_ml.round(2)

# Exibindo os resultados no Streamlit
     st.write("### Model Performance Comparison")
     st.dataframe(results_df_ml)


# ### Modelo XGBoost com todos os dados (representado como modelo1)
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
     model1 = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
     model1.fit(X_train, y_train)
     predictions_xgb = model1.predict(X_test)

# Exibindo métricas para o modelo 1
    model1_R2 = r2_score(y_test, predictions_xgb)
    model1_MSE = mean_squared_error(y_test, predictions_xgb)
    model1_RMSE = np.sqrt(model1_MSE)
    model1_MAE = mean_absolute_error(y_test, predictions_xgb)

    st.write("### XGBoost Model Performance")
    st.write(f"R² Score: {model1_R2}")
    st.write(f"MSE (Mean Squared Error): {model1_MSE}")
    st.write(f"RMSE (Root Mean Squared Error): {model1_RMSE}")
    st.write(f"MAE (Mean Absolute Error): {model1_MAE}")

# ### Plotando Real vs Previsões
    st.markdown("### Actual vs Predicted Prices for XGBoost")

# Plotando os valores reais vs previstos
   plt.figure(figsize=(8, 6))
   plt.scatter(y_test, predictions_xgb, color='blue', alpha=0.5)
   plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
   plt.title('XGBoost: Actual vs Predicted')
   plt.xlabel('Actual Values')
   plt.ylabel('Predicted Values')
   st.pyplot(plt)


