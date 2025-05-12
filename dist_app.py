import streamlit as st
import pandas as pd
import sqlite3

# Connect to the SQLite database
def load_data():
    conn = sqlite3.connect('soiltype.db')
    query = "SELECT DISTINCT `State Name`, `District` FROM soiltype;"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def get_soil_type(state, district):
    conn = sqlite3.connect('soiltype.db')
    query = """
    SELECT `Soil Type`
    FROM soiltype
    WHERE `State Name` = ? AND `District` = ?
    """
    df = pd.read_sql(query, conn, params=(state, district))
    conn.close()
    if not df.empty:
        return df.iloc[0]['Soil Type']
    else:
        return "Soil Type not found"

# Load data from the database
data = load_data()

# Streamlit app
st.title('Soil Type')

# Dropdowns for State Name and District
state = st.selectbox('Select State Name', data['State Name'].unique())
district = st.selectbox('Select District', data[data['State Name'] == state]['District'].unique())

# Display the Soil Type based on selections
if state and district:
    soil_type = get_soil_type(state, district)
    st.write(f"**Soil Type:** {soil_type}")
