import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import openai
from openai import OpenAI
from sklearn.preprocessing import StandardScaler

# Initialize OpenAI client (replace with your own key)
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError('The OPENAI_API_KEY environment variable is not set. Please set it in your .env file.')
justkey = OpenAI(api_key=api_key)

@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

def get_data_summary(df):
    """Generate a summary of the dataset to send to OpenAI for suggestions."""
    summary = {
        "columns": df.columns.tolist(),
        "numeric_summary": df.describe().to_dict(),
        "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
    }
    return summary

def get_openai_insights(summary):
    """Use OpenAI to generate insights, KPIs, and visual suggestions."""
    prompt = (
        "You are a Business Intelligence Analyst. Given the following data summary, "
        "generate key performance indicators (KPIs) and suggest relevant visualization types:\n"
        f"{summary}\n"
        "Please provide concise KPIs and visualization types."
    )
    
    try:
        response = justkey.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a Business Intelligence Analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error fetching insights from OpenAI: {e}"

def main():
    # Set page configuration
    st.set_page_config(page_title="Augumented Data Analysis", layout="wide")

    # Title
    st.title("📊 Augumented Data Analysis")

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Load the data
        df = load_data(uploaded_file)

        # Convert 'Date' column to datetime if it exists
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])
        
        # Update list of date columns
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

        # Tabs Layout
        tab1, tab2, tab3 = st.tabs(["Overview", "Chat with Data", "Visualizations"])

        # Overview Tab
        with tab1:
            st.subheader("Data Overview")
            st.write("Data Preview:")
            st.write(df.head())

            # Basic Statistics
            st.subheader("Basic Statistics")
            st.write(df.describe())

            # AI Insights Button
            if st.button("Get AI Insights"):
                summary = get_data_summary(df)
                ai_insights = get_openai_insights(summary)
                st.subheader("AI Generated Insights")
                st.write(ai_insights)

         # Chat with Data Tab
        with tab2:
            st.subheader("Chat with Data")
            # Query for the data
            query = st.text_area("Chat with Dataframe")

        # Visualization Tab
        with tab3:
            st.sidebar.header("Visualization Settings")
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

            # Select columns for X and Y axes
            x_column = st.sidebar.selectbox("Select X-axis Column", numeric_cols + categorical_cols + date_cols)
            y_columns = st.sidebar.multiselect("Select Y-axis Columns", [col for col in numeric_cols+categorical_cols if col != x_column])

            # Visualization Type
            viz_type = st.sidebar.selectbox("Select Visualization Type", ["Line Chart", "Scatter Plot", "Bar Chart", "Pie Chart", "Box Plot"])

            # Initialize fig
            fig = None

            # Create Visualization
            st.subheader(f"{viz_type} Visualization")
            if viz_type == "Line Chart" and y_columns:
                if pd.api.types.is_datetime64_any_dtype(df[x_column]):
                    df_grouped = df.groupby(pd.Grouper(key=x_column, freq='M')).sum().reset_index()
                    fig = px.line(df_grouped, x=x_column, y=y_columns, title='Line Chart (Monthly Aggregated)')
                else:
                    fig = px.line(df, x=x_column, y=y_columns)
            elif viz_type == "Scatter Plot" and y_columns:
                fig = px.scatter(df, x=x_column, y=y_columns)
            elif viz_type == "Bar Chart" and y_columns:
                fig = px.bar(df, x=x_column, y=y_columns)
            elif viz_type == "Pie Chart" and categorical_cols:
                category_column = st.sidebar.selectbox("Select Categorical Column for Pie Chart", categorical_cols)
                fig = px.pie(df, names=category_column)
            elif viz_type == "Box Plot" and y_columns:
                fig = go.Figure(data=[go.Box(y=df[col], name=col) for col in y_columns])
                fig.update_layout(title='Box Plot of Selected Columns')
            else:
                st.warning("Please select appropriate columns for the visualization.")

            # Display the plot if created
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()

# Requirements to install the necessary libraries
# Create a .env file and add your OpenAI API key as follows:
# OPENAI_API_KEY=your_openai_api_key_here
# Run the following command to install all required libraries:
#
# pip install streamlit pandas plotly scikit-learn openai>=1.0.0 python-dotenv
