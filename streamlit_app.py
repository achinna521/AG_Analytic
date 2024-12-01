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
        if isinstance(value, alt.Chart):
            st.altair_chart(value, use_container_width=True)
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
        "Please provide concise KPIs \n
        \n and visualization types."
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
    st.set_page_config(page_title="Augmented Data Analysis", layout="wide")

    # Title in the main section
    st.title("📊 Augmented Data Analysis")

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
                                elif isinstance(response, dict) and "type" in response and "value" in response:
                                    if response["type"] == "plot":
                                        StreamlitResponse().format_plot(response)
                                    else:
                                        StreamlitResponse().format_other(response)
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

                # Select columns for X and Y axes
                x_column = st.selectbox("Select X-axis Column", numeric_cols + categorical_cols + date_columns)
                y_columns = st.multiselect("Select Y-axis Columns", [col for col in numeric_cols if col != x_column])

                # Visualization Type
                viz_type = st.selectbox("Select Visualization Type", ["Line Chart", "Scatter Plot", "Bar Chart", "Pie Chart", "Box Plot", "Histogram", "Area Chart"])

                # Time Aggregation Level (only if a date column is chosen)
                time_aggregation = None
                if pd.api.types.is_datetime64_any_dtype(df[x_column]):
                    time_aggregation = st.selectbox("Select Time Aggregation Level", ["None", "Year", "Quarter", "Month", "Week", "Day"])

            # Create Visualization
            st.subheader(f"{viz_type} Visualization")
            fig = None

            if time_aggregation != "None" and pd.api.types.is_datetime64_any_dtype(df[x_column]):
                df_grouped = df.copy()
                if time_aggregation == "Year":
                    df_grouped[x_column] = df_grouped[x_column].dt.to_period('Y').dt.start_time
                elif time_aggregation == "Quarter":
                    df_grouped[x_column] = df_grouped[x_column].dt.to_period('Q').dt.start_time
                elif time_aggregation == "Month":
                    df_grouped[x_column] = df_grouped[x_column].dt.to_period('M').dt.start_time
                elif time_aggregation == "Week":
                    df_grouped[x_column] = df_grouped[x_column].dt.to_period('W').dt.start_time
                elif time_aggregation == "Day":
                    df_grouped[x_column] = df_grouped[x_column].dt.to_period('D').dt.start_time
                df_grouped = df_grouped.groupby(x_column).sum().reset_index()
            else:
                df_grouped = df

            if viz_type == "Line Chart" and y_columns:
                fig = alt.Chart(df_grouped).mark_line().encode(
                    x=x_column,
                    y=alt.Y(y_columns[0], type='quantitative')
                ).properties(width=700, height=400)
            elif viz_type == "Bar Chart" and y_columns:
                fig = alt.Chart(df_grouped).mark_bar().encode(
                    x=x_column,
                    y=alt.Y(y_columns[0], type='quantitative')
                ).properties(width=700, height=400)
            elif viz_type == "Scatter Plot" and y_columns:
                fig = alt.Chart(df).mark_point().encode(
                    x=x_column,
                    y=alt.Y(y_columns[0], type='quantitative')
                ).properties(width=700, height=400)
            elif viz_type == "Pie Chart" and x_column in categorical_cols and y_columns:
                df_pie = df.groupby(x_column)[y_columns[0]].sum().reset_index()
                fig = alt.Chart(df_pie).mark_arc().encode(
                    theta=alt.Theta(field=y_columns[0], type="quantitative"),
                    color=alt.Color(field=x_column, type="nominal")
                ).properties(width=700, height=400)
            elif viz_type == "Histogram" and y_columns:
                fig = alt.Chart(df).mark_bar().encode(
                    x=alt.X(y_columns[0], bin=True),
                    y='count()'
                ).properties(width=700, height=400)
            elif viz_type == "Box Plot" and y_columns:
                fig = alt.Chart(df).mark_boxplot().encode(
                    x=x_column,
                    y=alt.Y(y_columns[0], type='quantitative')
                ).properties(width=700, height=400)
            elif viz_type == "Area Chart" and y_columns:
                fig = alt.Chart(df_grouped).mark_area().encode(
                    x=x_column,
                    y=alt.Y(y_columns[0], type='quantitative')
                ).properties(width=700, height=400)
            
            if fig is not None:
                st.altair_chart(fig, use_container_width=True)
            else:
                st.warning("Please select appropriate columns for the visualization.")

if __name__ == "__main__":
    main()
