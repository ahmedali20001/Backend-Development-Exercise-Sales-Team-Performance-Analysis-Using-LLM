Sales Team Performance Analysis

This Streamlit app allows you to analyze the performance of a sales team based on provided data. It includes functionalities to analyze individual sales representative performance, overall team performance summary, and sales performance trends.
Setup
1.	Clone the repository to your local machine.
2.	Install the required dependencies by running:

       pip install -r requirements.txt 

3.	Run the Streamlit app using the following command:

       streamlit run app.py 

Usage

Home Page
•	Raw Dataset: Displays the raw dataset loaded from the provided CSV file.
•	API Key: Enter your OpenAI API key.
•	User Query: Enter a question based on the dataset to get insights using OpenAI.

Functionalities
•	Individual Sales Representative Performance Analysis: Enter an Employee ID to analyze the individual performance of a sales representative.
•	Overall Sales Team Performance Summary: Displays summary metrics for the overall team performance.
•	Sales Performance Trends and Forecasting: Displays sales performance trends and forecasting for each day of the week.

Functions
•	load_data(url): Load data from a URL (CSV file).
•	prepare_data(df): Lowercase column names.
•	execute_sql_query(conn, user_query): Execute SQL queries on the database.
•	extract_relevant_info(df): Extract relevant information from the DataFrame.
•	make_api_request(endpoint, params=None): Make requests to an API endpoint.
•	plot_individual_performance(df, rep_id): Plot individual sales and call performance for a specific representative.
•	plot_team_performance(df): Plot summary metrics for the overall team performance.
•	plot_performance_trends(df): Plot sales performance trends and forecasting for each day of the week.

Requirements
•	streamlit
•	pandas
•	matplotlib
•	seaborn
•	sqlalchemy
•	langchain_community
•	requests
