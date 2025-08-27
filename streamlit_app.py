# Import packages
import streamlit as st
import pandas as pd
import altair as alt
import datetime
from snowflake.connector import connect
import os

# Password protection setup
def check_password():
    """Returns True if the user entered the correct password."""
    def password_entered():
        """Checks whether the entered password is correct."""
        if st.session_state["password"] == st.secrets.get("APP_PASSWORD", "dxt(Kl='1]87"):
            st.session_state["password_correct"] = True
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show password input
        st.text_input("Enter Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        # Password incorrect, show input again
        st.text_input("Enter Password", type="password", on_change=password_entered, key="password")
        st.error("Incorrect password, please try again.")
        return False
    return True

# Check password before showing app content
if not check_password():
    st.stop()  # Halt execution until correct password is entered

# App title
st.title("🧬 BiobankTidy") 
st.caption("BiobankingAI – MVP in collaboration with ASN")


# Snowflake connection (using secrets.toml)
try:
    conn = connect(
        account=st.secrets["connections"]["snowflake"]["account"],
        user=st.secrets["connections"]["snowflake"]["user"],
        password=st.secrets["connections"]["snowflake"]["password"],
        database=st.secrets["connections"]["snowflake"]["database"],
        schema=st.secrets["connections"]["snowflake"]["schema"],
        warehouse=st.secrets["connections"]["snowflake"]["warehouse"],
        role=st.secrets["connections"]["snowflake"]["role"]
    )
except Exception as e:
    st.error(f"Failed to connect to Snowflake: {str(e)}")
    st.stop()

# Load your Snowflake table
try:
    cursor = conn.cursor()
    query = "SELECT * FROM ASN"
    cursor.execute(query)
    df_pandas = pd.DataFrame.from_records(iter(cursor), columns=[x[0] for x in cursor.description])
    cursor.close()
except Exception as e:
    st.error(f"Error querying Snowflake: {str(e)}")
    conn.close()
    st.stop()

# Close the connection
conn.close()

# Cache the data to improve performance
@st.cache_data
def load_data():
    # Convert date/time columns to datetime if possible
    date_cols = [col for col in df_pandas.columns if 'Date' in col or 'Time' in col]
    for col in date_cols:
        try:
            df_pandas[col] = pd.to_datetime(df_pandas[col], errors='coerce')
        except:
            pass
    # Ensure GENDER and other categorical columns are strings
    for col in ['GENDER', 'Patient Race', 'OD_OS']:
        if col in df_pandas.columns:
            df_pandas[col] = df_pandas[col].astype(str).replace('nan', None)
    return df_pandas

data = load_data()

# Tabs for Data, Visualization, and Summary
data_tab, viz_tab, summary_tab = st.tabs(["Data Table", "Visualizations", "Summary Stats"])

with data_tab:
    st.write("### PilotData-WholeGlobe_Banked+Pending")
    
    # Global search across all columns
    search_query = st.text_input("Global Search (case-insensitive, searches all columns)", "")
    if search_query:
        mask = data.apply(lambda row: row.astype(str).str.contains(search_query, case=False, na=False).any(), axis=1)
        filtered_df = data[mask]
    else:
        filtered_df = data.copy()
    
    # Smart filtering system in sidebar
    with st.sidebar:
        st.write("#### Smart Filters")
        filter_cols = st.multiselect("Select columns to filter", sorted(filtered_df.columns.tolist()))
        
        filter_conditions = {}
        for col in filter_cols:
            if col not in filtered_df.columns:
                st.warning(f"Column {col} not found in dataset.")
                continue
            if pd.api.types.is_numeric_dtype(filtered_df[col]):
                min_val = float(filtered_df[col].min()) if not filtered_df[col].isna().all() else 0.0
                max_val = float(filtered_df[col].max()) if not filtered_df[col].isna().all() else 100.0
                selected_range = st.slider(
                    f"{col} (range)",
                    min_val,
                    max_val,
                    (min_val, max_val),
                    key=f"slider_{col}"
                )
                filter_conditions[col] = filtered_df[col].between(selected_range[0], selected_range[1])
            elif pd.api.types.is_datetime64_any_dtype(filtered_df[col]):
                min_date = filtered_df[col].min().date() if not pd.isnull(filtered_df[col].min()) else datetime.date.today()
                max_date = filtered_df[col].max().date() if not pd.isnull(filtered_df[col].max()) else datetime.date.today()
                selected_dates = st.date_input(
                    f"{col} (date range)",
                    (min_date, max_date),
                    key=f"date_{col}"
                )
                if len(selected_dates) == 2:
                    start, end = selected_dates
                    filter_conditions[col] = (filtered_df[col] >= pd.to_datetime(start)) & (filtered_df[col] <= pd.to_datetime(end))
            else:
                unique_vals = filtered_df[col].dropna().astype(str).unique().tolist()
                if len(unique_vals) <= 20:  # For categorical/binary (e.g., GENDER, Yes/No)
                    selected_vals = st.multiselect(f"{col} (select values)", sorted(unique_vals), key=f"multi_{col}")
                    if selected_vals:
                        filter_conditions[col] = filtered_df[col].astype(str).isin(selected_vals)
                else:  # For text columns (e.g., Medical History)
                    filter_text = st.text_input(f"{col} (contains text)", key=f"text_{col}")
                    if filter_text:
                        filter_conditions[col] = filtered_df[col].astype(str).str.contains(filter_text, case=False, na=False)
    
    # Apply all filters
    if filter_conditions:
        combined_mask = pd.Series(True, index=filtered_df.index)
        for condition in filter_conditions.values():
            combined_mask &= condition
        filtered_df = filtered_df[combined_mask]
    
    # Show number of rows after filtering
    st.write(f"Total rows (after filters): {len(filtered_df)}")
    
    # Display data with Streamlit's dataframe, disabling copy-paste and download
    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=400,
        hide_index=False
    )
    
    # Custom CSS to prevent text selection/copy and hide download button
    st.markdown(
        """
        <style>
        .stDataFrame {
            user-select: none;
            -webkit-user-select: none;
            -moz-user-select: none;
            -ms-user-select: none;
        }
    
        [data-testid="stDownloadButton"],
        .stDataFrame .st-eb,
        .stDataFrame [data-testid="stElementToolbar"] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


with viz_tab:
    st.write("### Data Visualizations (based on filtered data)")
    
    # Get numeric, categorical, and flag columns
    numeric_cols = filtered_df.select_dtypes(include=['number']).columns.tolist()
    all_cols = filtered_df.columns.tolist()
    flag_cols = [col for col in all_cols if col.startswith(('Hx_', 'Sx_', 'Med_'))]
    categorical_cols = ['GENDER', 'Patient Race', 'OD_OS'] + flag_cols
    
    # Histogram for numeric columns
    if numeric_cols:
        hist_col = st.selectbox("Histogram: Select numeric column (e.g., Age)", sorted(numeric_cols))
        if hist_col and not filtered_df[hist_col].isna().all():
            hist_chart = alt.Chart(filtered_df).mark_bar().encode(
                alt.X(hist_col, bin=True),
                y='count()',
                tooltip=['count()']
            ).properties(title=f"Histogram of {hist_col}").interactive()
            st.altair_chart(hist_chart, use_container_width=True)
        else:
            st.warning(f"No valid data for {hist_col} to display histogram.")
    
    # Bar chart for categorical/binary columns
    cat_col = st.selectbox("Bar Chart: Select categorical/flag column", sorted(categorical_cols))
    if cat_col in filtered_df.columns and not filtered_df[cat_col].isna().all():
        bar_chart = alt.Chart(filtered_df).mark_bar().encode(
            x=alt.X(cat_col, type='nominal'),
            y='count()',
            color=cat_col,
            tooltip=['count()', cat_col]
        ).properties(title=f"Distribution of {cat_col}").interactive()
        try:
            st.altair_chart(bar_chart, use_container_width=True)
        except Exception as e:
            st.warning(f"Unable to render bar chart for {cat_col}: {str(e)}")
    else:
        st.warning(f"Column {cat_col} not found or has no valid data for bar chart.")
    
    # Scatter plot for numeric columns
    if len(numeric_cols) >= 2:
        scatter_x = st.selectbox("Scatter: X-axis (numeric)", sorted(numeric_cols))
        scatter_y = st.selectbox("Scatter: Y-axis (numeric)", sorted(numeric_cols))
        if scatter_x in filtered_df.columns and scatter_y in filtered_df.columns and not filtered_df[[scatter_x, scatter_y]].isna().all().all():
            scatter_chart = alt.Chart(filtered_df).mark_circle().encode(
                x=scatter_x,
                y=scatter_y,
                tooltip=[scatter_x, scatter_y]
            ).properties(title=f"{scatter_x} vs {scatter_y}").interactive()
            st.altair_chart(scatter_chart, use_container_width=True)
        else:
            st.warning(f"No valid data for {scatter_x} or {scatter_y} to display scatter plot.")
    
    # Pie chart for specific categorical columns
    pie_col = st.selectbox("Pie Chart: Select column (e.g., GENDER)", sorted(['GENDER', 'OD_OS', 'Patient Race']))
    if pie_col in filtered_df.columns and not filtered_df[pie_col].isna().all():
        pie_data = filtered_df[[pie_col]].dropna()  # Drop nulls for pie chart
        if not pie_data.empty:
            pie_chart = alt.Chart(pie_data).mark_arc().encode(
                theta=alt.Theta("count()", stack=True),
                color=alt.Color(pie_col, type='nominal'),
                tooltip=[pie_col, 'count()']
            ).properties(title=f"Pie Chart of {pie_col}")
            try:
                st.altair_chart(pie_chart, use_container_width=True)
            except Exception as e:
                st.warning(f"Unable to render pie chart for {pie_col}: {str(e)}")
        else:
            st.warning(f"No valid data for {pie_col} to display pie chart.")
    else:
        st.warning(f"Column {pie_col} not found or has no valid data for pie chart.")

with summary_tab:
    st.write("### Summary Statistics (based on filtered data)")
    try:
        st.write(filtered_df.describe(include='all'))
    except Exception as e:
        st.warning(f"Error generating summary statistics: {str(e)}")
