import streamlit as st
import pandas as pd
import altair as alt
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
        if isinstance(result["value"], pd.DataFrame):
            st.dataframe(result["value"])
        else:
            st.write("No valid DataFrame to display.")
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
        value = result.get("value", "No content to display")
        if value is None:
            st.write("The response returned no data. Please check your query or try again with different parameters.")
        elif isinstance(value, str):
            st.write(value)
        elif isinstance(value, dict):
            st.write(value)
        else:
            st.write("Unexpected content type, displaying raw output:")
            st.write(value)

@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

@st.cache_data
def get_ai_insights_cached(df):
    summary = get_data_summary(df)
    return get_openai_insights(summary)

def create_altair_chart(df, x_column, y_columns, chart_type, time_aggregation=None):
    """Creates Altair chart depending on the chart type and labels the values."""
    # If the X column is a datetime and time aggregation is selected, aggregate by the chosen level
    if pd.api.types.is_datetime64_any_dtype(df[x_column]) and time_aggregation:
        if time_aggregation == "Year":
            df[x_column] = df[x_column].dt.to_period("Y").dt.start_time
        elif time_aggregation == "Month":
            df[x_column] = df[x_column].dt.to_period("M").dt.start_time

    # Summarize the data by grouping by the X column and aggregating the Y columns
    df_summary = df.groupby(x_column)[y_columns].sum().reset_index()

    base = alt.Chart(df_summary).encode(
        x=alt.X(x_column, title=x_column),
        tooltip=[x_column] + y_columns
    )

    layers = []
    for y_column in y_columns:
        if chart_type == "Line Chart":
            layer = base.mark_line().encode(
                y=alt.Y(y_column, type='quantitative', title='Value')
            )
        elif chart_type == "Scatter Plot":
            layer = base.mark_circle(size=60).encode(
                y=alt.Y(y_column, type='quantitative', title='Value')
            )
        elif chart_type == "Bar Chart":
            layer = base.mark_bar().encode(
                y=alt.Y(y_column, type='quantitative', title='Value')
            )
        elif chart_type == "Area Chart":
            layer = base.mark_area().encode(
                y=alt.Y(y_column, type='quantitative', title='Value')
            )
        elif chart_type == "Histogram":
            layer = alt.Chart(df).mark_bar().encode(
                x=alt.X(y_column, bin=True, title=y_column),
                y=alt.Y('count()', title='Count')
            )
        elif chart_type == "Box Plot":
            layer = alt.Chart(df).mark_boxplot().encode(
                x=alt.X(x_column, title=x_column),
                y=alt.Y(y_column, type='quantitative', title='Value')
            )
        else:
            st.warning("Chart type not supported.")
            return None
        layers.append(layer)

    chart = alt.layer(*layers).properties(
        width=800,
        height=400
    )

    return chart
def summarize_data(df, x_column, y_columns):
    """Generate a summary of the dataset to display in the Streamlit app."""
    summary = {
        "Columns Selected": x_column + ', ' + ', '.join(y_columns),
        "Number of Rows": len(df),
        "Number of Columns": len(df.columns),
    }
    return summary    
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

def parse_multiple_queries(query):
    """Parse the user query into multiple individual queries for multiple plots."""
    queries = query.split("plot ")
    parsed_queries = [q.strip() for q in queries if q.strip()]
    return parsed_queries

def main():
    # Set page configuration
    st.set_page_config(page_title="Augmented Data Analysis with Altair", layout="wide")

    # Title in the main section
    st.title("📊 Augmented Data Analysis with Altair")

    # File uploader in the sidebar
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Load the data
        df = load_data(uploaded_file)

        # Convert 'Date' column to datetime if it exists
        date_columns = st.sidebar.multiselect("Select Date Columns (if any)", df.columns[df.dtypes == 'object'], help="Select columns that should be treated as dates.")
        for col in date_columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df = df.dropna(subset=[col])
        
        # Tabs Layout
        tab1, tab2, tab3 = st.tabs(["Overview", "Chat with Data", "Visualizations"])

        # Overview Tab
        with tab1:
            st.subheader("Data Overview")

            # Data Preview in an expander
            with st.expander("Data Preview", expanded=True):
                st.write(df)

            # Basic Statistics
            with st.expander("Basic Statistics", expanded=True):
               st.write(df.describe())
                
            # AI Insights Button
            if st.button("Get AI Insights"):
                ai_insights = get_ai_insights_cached(df)
                st.subheader("AI Generated Insights")
                st.write(ai_insights)

        # Chat with Data Tab
        with tab2:
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
                            
                            # Parse the query into multiple sub-queries if needed
                            parsed_queries = parse_multiple_queries(query)
                            
                            for sub_query in parsed_queries:
                                response = query_engine.chat(sub_query)
                                if response is None:
                                    st.write("")
                                elif isinstance(response, pd.DataFrame):
                                    st.dataframe(response)
                                elif isinstance(response, str):
                                    st.write(response)
                                else:
                                    st.write("Unexpected response format. Here is the raw response:")
                                    st.write(response)

                        except Exception as e:
                            st.error(f"An error occurred: {e}")

        # Visualization Tab
        with tab3:
            # Visualization settings wrapped in an expander within the sidebar
            with st.sidebar.expander("Visualization Settings", expanded=True):
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                date_cols = df.select_dtypes(include=['datetime']).columns.tolist()

                # Select columns for X and Y axes
                x_column = st.selectbox("Select X-axis Column", numeric_cols + categorical_cols + date_cols)
                y_columns = st.multiselect("Select Y-axis Columns", [col for col in numeric_cols if col != x_column])

                # If X-axis is a date column, allow the user to choose time aggregation level
                time_aggregation = None
                if x_column in date_cols:
                    time_aggregation = st.selectbox("Select Time Aggregation Level", ["None", "Year", "Month"], index=0)
                    if time_aggregation == "None":
                        time_aggregation = None

                # Visualization Type
                viz_type = st.selectbox("Select Visualization Type", ["Line Chart", "Scatter Plot", "Bar Chart", "Area Chart", "Histogram", "Box Plot"])

            # Create Visualization
            if x_column and y_columns:
                st.subheader(f"{viz_type} Visualization")
                summary = summarize_data(df, x_column, y_columns)
                st.write(summary)
                altair_chart = create_altair_chart(df, x_column, y_columns, viz_type, time_aggregation)
                if altair_chart:
                    st.altair_chart(altair_chart, use_container_width=True)
            else:
                st.warning("Please select appropriate columns for the visualization.")

if __name__ == "__main__":
    main()
