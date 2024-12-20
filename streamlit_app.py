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
    # Grouping by selected feature and date_month to calculate mean of relevant columns (price and Predicted)
    grouped = df_analysis.groupby(['selected_feature', 'date_month'])[['price', 'Predicted']].mean()

    # Calculating percentage changes for price and Predicted
    grouped['price_pct_change'] = grouped['price'].pct_change() * 100
    grouped['Predicted_pct_change'] = grouped['Predicted'].pct_change() * 100

    # Resetting the index for better readability (flattening the multi-level index)
    grouped_reset = grouped.reset_index()

    # Displaying the data in Streamlit
    st.title("Analysis by Selected Feature")
    st.write("### Grouped Data (Only Selected Features)")
    st.dataframe(grouped_reset[['selected_feature', 'date_month', 'price', 'Predicted', 'price_pct_change', 'Predicted_pct_change']])

# Transposing the data (optional)
if st.checkbox("Transpose DataFrame"):  # Add the colon at the end of the 'if' statement
    st.write(grouped_reset[['selected_feature', 'date_month', 'price', 'Predicted', 'price_pct_change', 'Predicted_pct_change']].T)
import streamlit as st
import pandas as pd

# Assuming `Features`, `Target`, and `df` are already loaded and `df_analysis` is created

import streamlit as st
import pandas as pd

# Assuming df_analysis has 'date_month', 'price', 'Predicted', and 'selected_feature' columns

with st.expander('📄 Features', expanded=True):
    # Convert date_month to datetime if it's not already
    df_analysis['date_month'] = pd.to_datetime(df_analysis['date_month'], errors='coerce')

    # Extract month and year from date_month to group by month
    df_analysis['month_year'] = df_analysis['date_month'].dt.to_period('M')

    # Grouping by the extracted 'month_year' and calculating the mean for price and Predicted
    grouped = df_analysis.groupby('month_year')[['price', 'Predicted']].mean()

    # Calculating percentage changes for price and Predicted
    grouped['price_pct_change'] = grouped['price'].pct_change() * 100
    grouped['Predicted_pct_change'] = grouped['Predicted'].pct_change() * 100

    # Resetting the index for better readability
    grouped_reset = grouped.reset_index()

    # Displaying the data in Streamlit
    st.title("Analysis by Month")
    st.write("### Grouped Data by Month")
    st.dataframe(grouped_reset[['month_year', 'price', 'Predicted', 'price_pct_change', 'Predicted_pct_change']])

# Transposing the data (optional)
if st.checkbox("Transpose DataFrame"):  # Add the colon at the end of the 'if' statement
    st.write(grouped_reset[['month_year', 'price', 'Predicted', 'price_pct_change', 'Predicted_pct_change']].T)
