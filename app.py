import sqlite3
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from langchain_community.llms import OpenAI
from langchain_community.utilities import SQLDatabase
import json
import requests

#####################################
#            FUNCTIONS              #
#####################################
@st.cache_data()
def load_data(url):
    """
    Load data from URL
    """
    df = pd.read_csv(url)
    return df

def prepare_data(df):
    """
    Lowercase columns
    """
    df.columns = [x.replace(' ', '_').lower() for x in df.columns]
    return df

def execute_sql_query(conn, user_query):
    try:
        result = pd.read_sql_query(user_query, con=conn)
        return result
    except Exception as e:
        return f"Error executing SQL query: {e}"

def extract_relevant_info(df):
    """
    Extract relevant information from DataFrame
    """
    # Example: extracting employee names
    employee_names = df['employee_name'].tolist()
    return employee_names

def make_api_request(endpoint, params=None):
    base_url = "http://localhost:5000/api/"  # Change to your Flask server address
    url = base_url + endpoint
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Error {response.status_code} - {response.text}"}

def plot_individual_performance(df, rep_id):
    # Filter data for the specified representative
    rep_data = df[df['employee_id'] == rep_id]
    
    # Plot individual sales for each day
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    days = ['mon', 'tue', 'wed', 'thur', 'fri', 'sat', 'sun']
    for day in days:
        if day+'_text' in rep_data.columns:
            ax[0].plot(rep_data['dated'], rep_data[day+'_text'], label=day.capitalize())
            ax[0].set_title(f"Individual Sales for Representative {rep_id}")
            ax[0].set_xlabel("Date")
            ax[0].set_ylabel("Sales")
            ax[0].legend()
        if day+'_call' in rep_data.columns:
            ax[1].plot(rep_data['dated'], rep_data[day+'_call'], label=day.capitalize())
            ax[1].set_title(f"Individual Calls for Representative {rep_id}")
            ax[1].set_xlabel("Date")
            ax[1].set_ylabel("Calls")
            ax[1].legend()
    st.pyplot(fig)


def plot_team_performance(df):
    # Aggregate data for overall team performance
    team_data = df.groupby('dated').agg({'revenue_confirmed': 'sum', 'revenue_pending': 'sum'}).reset_index()
    
    # Plot summary metrics for the team
    fig, ax = plt.subplots()
    ax.plot(team_data['dated'], team_data['revenue_confirmed'], label='Total Confirmed Revenue')
    ax.plot(team_data['dated'], team_data['revenue_pending'], label='Total Pending Revenue')
    ax.set_title("Overall Sales Team Performance Summary")
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue")
    ax.legend()
    st.pyplot(fig)
    
def plot_performance_trends(df):
    # Plot sales performance trends for each day of the week against sales
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    days = ['mon', 'tue', 'wed', 'thur', 'fri', 'sat', 'sun']
    for day in days:
        if day+'_text' in df.columns:
            ax[0].bar(df['dated'], df[day+'_text'], label=day.capitalize())
            ax[0].set_title(f"Sales Performance Trends ({day.capitalize()})")
            ax[0].set_xlabel("Date")
            ax[0].set_ylabel("Sales")
            ax[0].legend()
        if day+'_call' in df.columns:
            ax[1].bar(df['dated'], df[day+'_call'], label=day.capitalize())
            ax[1].set_title(f"Call Performance Trends ({day.capitalize()})")
            ax[1].set_xlabel("Date")
            ax[1].set_ylabel("Calls")
            ax[1].legend()
    st.pyplot(fig)

#####################################
#        LOCALS & CONSTANTS         #
#####################################
table_name = 'sales_data'
uri = "file::memory:?cache=shared"

#####################################
#            HOME PAGE              #
#####################################
st.title('Sales Team Performance Analysis')

# Read data
url = "https://raw.githubusercontent.com/ahmedali20001/Backend-Development-Exercise-Sales-Team-Performance-Analysis-Using-LLM/main/sales_performance_data.csv"
df = load_data(url)

# Display the entire dataset
show_entire_dataset = st.checkbox("Show entire dataset", False)
st.subheader('Raw Dataset')
if show_entire_dataset:
    st.write(df)
else:
    st.write(df.head(5))


# API key
openai_api_key = st.text_input(
    "API key", 
    placeholder='Type your API Key',
    type='password',
    disabled=False,
    help='Enter your OpenAI API key.'
)

# User query
user_query = st.text_input(
    "User Query", 
    placeholder="Enter your query",
    help="Enter a question based on the dataset"
)

# Commit data to SQL
data = prepare_data(df)
conn = sqlite3.connect(uri)
data.to_sql(table_name, conn, if_exists='replace', index=False)

# Create DB engine
eng = create_engine(
    url='sqlite:///file:memdb1?mode=memory&cache=shared', 
    poolclass=StaticPool, 
    creator=lambda: conn
)
db = SQLDatabase(engine=eng)

# Create OpenAI connection
if openai_api_key:
    llm = OpenAI(
        openai_api_key=openai_api_key, 
        temperature=0, 
        max_tokens=300
    )

# Run query and display result
if openai_api_key and user_query:
    try:
        # Extract relevant information from the DataFrame
        relevant_info = extract_relevant_info(df)
        
        # Use OpenAI to generate text based on the extracted information
        response = llm.generate(prompts=relevant_info, max_tokens=100)
        st.write(response['choices'][0]['text'])
    except Exception as e:
        st.error(f"Error executing OpenAI query: {e}")

# FUNCTIONALITIES
if st.button("Individual Sales Representative Performance Analysis"):
    rep_id = st.text_input("Enter Employee ID")
    if rep_id:
        plot_individual_performance(df, rep_id)

if st.button("Overall Sales Team Performance Summary"):
    plot_team_performance(df)

if st.button("Sales Performance Trends and Forecasting"):
    plot_performance_trends(df)
