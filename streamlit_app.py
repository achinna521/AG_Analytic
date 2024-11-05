import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px 
from plotly.subplots import make_subplots
from openai import OpenAI
from pandasai import SmartDataframe
from pandasai.llm import OpenAI as PandasAI_LLM
from pandasai.responses.response_parser import ResponseParser
import os
from dotenv import load_dotenv

# Load the environment variables
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError('The OPENAI_API_KEY environment variable is not set. Please set it in your .env file.')
justkey = OpenAI(api_key=api_key)

# Custom Response Parser for Streamlit
class StreamlitResponse(ResponseParser):
    def __init__(self, context=None) -> None:
        super().__init__(context)

    def format_dataframe(self, result):
        st.dataframe(result["value"])
        return

    def format_plot(self, result):
        value = result.get("value")
        
        if isinstance(value, list):
            # Create a 2x2 subplot layout for multiple plots
            fig = make_subplots(rows=2, cols=2, subplot_titles=[f"Plot {i+1}" for i in range(len(value))])

            # Add each plot to the subplot layout
            for idx, plot_value in enumerate(value):
                row = idx // 2 + 1
                col = idx % 2 + 1

                if isinstance(plot_value, go.Figure):
                    for trace in plot_value['data']:
                        fig.add_trace(trace, row=row, col=col)

            # Set layout properties and display the combined figure
            fig.update_layout(height=800, width=1000, title_text="Multiple Plots")
            st.plotly_chart(fig)

        elif isinstance(value, go.Figure):
            # Display a single figure if there is only one plot
            st.plotly_chart(value)

        elif isinstance(value, str) and os.path.isfile(value):
            st.image(value)
        
        else:
            st.write("Unexpected plot format, displaying raw output:")
            st.write(value)

    def format_other(self, result):
        st.write(result.get("value", "No content to display"))

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
    st.set_page_config(page_title="Augmented Analysis", layout="wide")

    # Title in the main section
    st.title("📊 Augmented Data Analysis")

    # File uploader in the sidebar
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

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

            # Data Preview in an expander
            with st.expander("Data Preview", expanded=True):
                st.write(df)

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
            st.subheader("Data Preview")
            # Data Preview in an expander
            with st.expander("Data Preview", expanded=True):
                st.write(df)
            st.subheader("Chat with Data")
            # Query for the data
            query = st.text_area("Chat with Dataframe")

            if st.button("Generate Response"):
                if query:
                    with st.spinner("OpenAI is generating an answer, please wait..."):
                        try:
                            # Initialize the language model
                            llm = PandasAI_LLM(api_key=api_key)
                            # Create a SmartDataframe instance with custom response parser
                            query_engine = SmartDataframe(
                                df,
                                config={
                                    "llm": llm,
                                    "response_parser": StreamlitResponse
                                }
                            )
                            
                            # Check if the query involves multiple plots
                            if "plot 1:" in query.lower() or "plot 2:" in query.lower():
                                plots = []
                                plot_queries = query.lower().split("plot ")
                                for plot_query in plot_queries[1:]:
                                    # Extract plot instructions and ask OpenAI to generate individual plots
                                    plot_query_text = plot_query.strip()
                                    plot_fig = query_engine.chat(plot_query_text)
                                    if isinstance(plot_fig, go.Figure):
                                        plots.append(plot_fig)

                                # Display in a 2x2 layout if there are multiple plots
                                if plots:
                                    StreamlitResponse(None).format_plot({"value": plots})

                            else:
                                # Execute single query
                                query_engine.chat(query)

                        except Exception as e:
                            st.error(f"An error occurred: {e}")

        # Visualization Tab
        with tab3:
            # Visualization settings wrapped in an expander within the sidebar
            with st.sidebar.expander("Visualization Settings", expanded=True):
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

                # Select columns for X and Y axes
                x_column = st.selectbox("Select X-axis Column", numeric_cols + categorical_cols + date_cols)
                y_columns = st.multiselect("Select Y-axis Columns", [col for col in numeric_cols + categorical_cols if col != x_column])

                # Visualization Type
                viz_type = st.selectbox("Select Visualization Type", ["Line Chart", "Scatter Plot", "Bar Chart", "Pie Chart", "Box Plot"])

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
                category_column = st.selectbox("Select Categorical Column for Pie Chart", categorical_cols)
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
