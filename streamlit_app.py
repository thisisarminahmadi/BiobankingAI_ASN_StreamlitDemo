# Import packages
import streamlit as st
import pandas as pd
import altair as alt
import datetime
from snowflake.connector import connect
import os
import re
from typing import List, Dict, Any, Tuple
import json

# Set page config and apply custom CSS
st.set_page_config(
    page_title="BiobankTidy - BiobankingAI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dark theme
st.markdown("""
<style>
    /* Main app styling */
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #262730;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #00ff88 !important;
        font-weight: 600;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #00ff88;
        color: #0e1117;
        border: none;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #00cc6a;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 255, 136, 0.3);
    }
    
    /* Selectboxes */
    .stSelectbox > div > div {
        background-color: #262730;
        border: 1px solid #4a4a4a;
        border-radius: 8px;
    }
    
    /* Text inputs */
    .stTextInput > div > div > input {
        background-color: #262730;
        border: 1px solid #4a4a4a;
        border-radius: 8px;
        color: #fafafa;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background-color: #262730;
        border-radius: 8px;
        border: 1px solid #4a4a4a;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #262730;
        border-radius: 8px 8px 0 0;
        border: 1px solid #4a4a4a;
        color: #fafafa;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #00ff88;
        color: #0e1117;
    }
    
    /* Sidebar elements */
    .sidebar .sidebar-content {
        background-color: #262730;
    }
    
    /* Captions and info boxes */
    .caption {
        color: #888888;
    }
    
    /* Success/Info/Warning boxes */
    .stAlert {
        border-radius: 8px;
        border: 1px solid #4a4a4a;
    }
    
    /* Hide download buttons and prevent text selection */
    [data-testid="stDownloadButton"],
    .stDataFrame .st-eb,
    .stDataFrame [data-testid="stElementToolbar"] {
        display: none !important;
    }
    
    .stDataFrame {
        user-select: none;
        -webkit-user-select: none;
        -moz-user-select: none;
        -ms-user-select: none;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #262730;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #00ff88;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #00cc6a;
    }
    
    /* Search results styling */
    .search-result {
        background-color: #262730;
        border: 1px solid #4a4a4a;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
    }
    
    .search-highlight {
        background-color: #00ff88;
        color: #0e1117;
        padding: 2px 4px;
        border-radius: 4px;
        font-weight: bold;
    }
    
    /* Cortex AI response styling */
    .cortex-response {
        background: linear-gradient(135deg, #00ff88 0%, #00cc6a 100%);
        color: #0e1117;
        padding: 16px;
        border-radius: 12px;
        margin: 12px 0;
        border: 2px solid #00ff88;
    }
</style>
""", unsafe_allow_html=True)

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

# Snowflake Cortex AI-Powered Search System
class SnowflakeCortexSearchEngine:
    def __init__(self, connection):
        self.conn = connection
        self.cursor = connection.cursor()
        self.column_metadata = {}
        self._build_column_metadata()
    
    def _build_column_metadata(self):
        """Build metadata for each column to optimize Cortex search."""
        try:
            # Get column information from Snowflake
            self.cursor.execute("DESCRIBE TABLE ASN")
            columns_info = self.cursor.fetchall()
            
            for col_info in columns_info:
                col_name = col_info[0]
                col_type = col_info[1]
                
                # Categorize columns
                category = "Other"
                if any(word in col_name.lower() for word in ['age', 'gender', 'race', 'ethnicity']):
                    category = "Demographics"
                elif col_name.startswith('Hx_'):
                    category = "Medical History"
                elif col_name.startswith('Sx_'):
                    category = "Symptoms"
                elif col_name.startswith('Med_'):
                    category = "Medications"
                elif any(word in col_name.lower() for word in ['bp', 'pressure', 'temp', 'pulse']):
                    category = "Vital Signs"
                elif any(word in col_name.lower() for word in ['lab', 'test', 'result']):
                    category = "Lab Results"
                elif any(word in col_name.lower() for word in ['date', 'time']):
                    category = "Dates & Times"
                
                self.column_metadata[col_name] = {
                    'type': col_type,
                    'category': category
                }
                
        except Exception as e:
            st.warning(f"Could not fetch column metadata: {str(e)}")
    
    def _generate_cortex_search_query(self, user_query: str) -> str:
        """Generate optimized Snowflake Cortex search query."""
        # Build searchable columns list
        searchable_columns = []
        for col, metadata in self.column_metadata.items():
            if metadata['category'] in ['Demographics', 'Medical History', 'Symptoms', 'Medications']:
                searchable_columns.append(col)
        
        # Limit to first 50 columns to avoid query size limits
        searchable_columns = searchable_columns[:50]
        
        # Build Cortex search query
        columns_str = "', '".join(searchable_columns)
        
        cortex_query = f"""
        SELECT *,
               CORTEX_SEARCH('{user_query}', columns => ARRAY_CONSTRUCT('{columns_str}')) as search_score
        FROM ASN 
        WHERE CORTEX_SEARCH('{user_query}', columns => ARRAY_CONSTRUCT('{columns_str}')) > 0.1
        ORDER BY search_score DESC
        """
        
        return cortex_query
    
    def _generate_hybrid_search_query(self, user_query: str) -> str:
        """Generate hybrid search combining Cortex AI with traditional filters."""
        # Parse query for specific conditions
        query_lower = user_query.lower()
        
        # Build WHERE conditions
        where_conditions = []
        
        # Medical conditions
        if 'diabetes' in query_lower:
            where_conditions.append("(Hx_Diabetes = 'Yes' OR LOWER(Hx_Diabetes) LIKE '%diabetes%')")
        
        if 'hypertension' in query_lower or 'high blood pressure' in query_lower:
            where_conditions.append("(Hx_Hypertension = 'Yes' OR LOWER(Hx_Hypertension) LIKE '%hypertension%')")
        
        if 'heart' in query_lower:
            where_conditions.append("(Hx_Heart_Disease = 'Yes' OR LOWER(Hx_Heart_Disease) LIKE '%heart%')")
        
        if 'vision' in query_lower or 'eye' in query_lower:
            where_conditions.append("(Sx_Vision = 'Yes' OR LOWER(Sx_Vision) LIKE '%vision%' OR LOWER(Sx_Vision) LIKE '%eye%')")
        
        # Demographics
        if 'male' in query_lower:
            where_conditions.append("(GENDER = 'Male' OR GENDER = 'M')")
        
        if 'female' in query_lower:
            where_conditions.append("(GENDER = 'Female' OR GENDER = 'F')")
        
        if any(word in query_lower for word in ['over', 'above', 'older', '>']):
            # Extract age if mentioned
            age_match = re.search(r'(\d+)', user_query)
            if age_match:
                age = age_match.group(1)
                where_conditions.append(f"Age > {age}")
        
        # Build the query
        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
        
        # Get searchable columns
        searchable_columns = []
        for col, metadata in self.column_metadata.items():
            if metadata['category'] in ['Demographics', 'Medical History', 'Symptoms', 'Medications']:
                searchable_columns.append(col)
        
        searchable_columns = searchable_columns[:50]
        columns_str = "', '".join(searchable_columns)
        
        hybrid_query = f"""
        SELECT *,
               CORTEX_SEARCH('{user_query}', columns => ARRAY_CONSTRUCT('{columns_str}')) as search_score
        FROM ASN 
        WHERE {where_clause}
        AND CORTEX_SEARCH('{user_query}', columns => ARRAY_CONSTRUCT('{columns_str}')) > 0.05
        ORDER BY search_score DESC
        """
        
        return hybrid_query
    
    def search(self, query: str, search_type: str = "cortex") -> Tuple[pd.DataFrame, Dict]:
        """Perform Snowflake Cortex AI-powered search."""
        if not query.strip():
            # Return all data if no query
            try:
                self.cursor.execute("SELECT * FROM ASN")
                df = pd.DataFrame.from_records(iter(self.cursor), columns=[x[0] for x in self.cursor.description])
                return df, {"interpretation": "No search query provided", "search_score": None}
            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")
                return pd.DataFrame(), {}
        
        try:
            # Generate appropriate query based on search type
            if search_type == "cortex":
                sql_query = self._generate_cortex_search_query(query)
            elif search_type == "hybrid":
                sql_query = self._generate_hybrid_search_query(query)
            else:
                # Fallback to basic search
                sql_query = f"""
                SELECT * FROM ASN 
                WHERE LOWER(CAST(OBJECT_CONSTRUCT(*) AS STRING)) LIKE '%{query.lower()}%'
                """
            
            # Execute the query
            self.cursor.execute(sql_query)
            df = pd.DataFrame.from_records(iter(self.cursor), columns=[x[0] for x in self.cursor.description])
            
            # Build search metadata
            search_metadata = {
                "interpretation": f"Cortex AI search for: {query}",
                "search_type": search_type,
                "query_used": sql_query,
                "results_count": len(df),
                "search_score_available": "search_score" in df.columns
            }
            
            return df, search_metadata
            
        except Exception as e:
            st.error(f"Snowflake Cortex search failed: {str(e)}")
            # Fallback to basic search
            try:
                self.cursor.execute(f"""
                SELECT * FROM ASN 
                WHERE LOWER(CAST(OBJECT_CONSTRUCT(*) AS STRING)) LIKE '%{query.lower()}%'
                """)
                df = pd.DataFrame.from_records(iter(self.cursor), columns=[x[0] for x in self.cursor.description])
                return df, {"interpretation": f"Fallback search for: {query}", "error": str(e)}
            except Exception as fallback_error:
                st.error(f"Fallback search also failed: {str(fallback_error)}")
                return pd.DataFrame(), {"error": str(e)}
    
    def get_search_suggestions(self, partial_query: str) -> List[str]:
        """Get search suggestions based on partial query."""
        if len(partial_query) < 2:
            return []
        
        suggestions = []
        
        # Search in column names
        for col in self.column_metadata.keys():
            if partial_query.lower() in col.lower():
                suggestions.append(f"Column: {col}")
        
        # Add common medical terms
        medical_suggestions = [
            "diabetes patients", "hypertension", "heart disease", "vision problems",
            "pain", "medication", "symptoms", "medical history",
            "male patients", "female patients", "age over 50", "race"
        ]
        
        for term in medical_suggestions:
            if partial_query.lower() in term:
                suggestions.append(f"Medical query: {term}")
        
        return list(set(suggestions))[:10]
    
    def test_cortex_availability(self) -> bool:
        """Test if Snowflake Cortex is available."""
        try:
            self.cursor.execute("SELECT CORTEX_SEARCH('test', columns => ARRAY_CONSTRUCT('GENDER')) FROM ASN LIMIT 1")
            return True
        except Exception as e:
            if "CORTEX_SEARCH" in str(e).upper():
                return False
            return True  # Other errors might not be related to Cortex availability

# Initialize Snowflake Cortex search engine
def get_cortex_search_engine(connection) -> SnowflakeCortexSearchEngine:
    return SnowflakeCortexSearchEngine(connection)

# Initialize search engine without caching to avoid UnhashableParamError
cortex_search_engine = SnowflakeCortexSearchEngine(conn)

# Test Cortex availability
cortex_available = cortex_search_engine.test_cortex_availability()

# Load initial data
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

# Process the data
def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process the dataframe for better compatibility."""
    # Convert date/time columns to datetime if possible
    date_cols = [col for col in df.columns if 'Date' in col or 'Time' in col]
    for col in date_cols:
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        except:
            pass
    # Ensure GENDER and other categorical columns are strings
    for col in ['GENDER', 'Patient Race', 'OD_OS']:
        if col in df.columns:
            df[col] = df[col].astype(str).replace('nan', None)
    return df

# Process the data directly
data = process_data(df_pandas)

# Column categorization function
def categorize_columns(columns: List[str]) -> Dict[str, List[str]]:
    """Categorize columns based on naming patterns and content."""
    categories = {
        'Demographics': [],
        'Medical History (Hx)': [],
        'Symptoms (Sx)': [],
        'Medications (Med)': [],
        'Vital Signs': [],
        'Lab Results': [],
        'Dates & Times': [],
        'Other': []
    }
    
    for col in columns:
        col_lower = col.lower()
        if any(word in col_lower for word in ['age', 'gender', 'race', 'ethnicity', 'height', 'weight']):
            categories['Demographics'].append(col)
        elif col.startswith('Hx_'):
            categories['Medical History (Hx)'].append(col)
        elif col.startswith('Sx_'):
            categories['Symptoms (Sx)'].append(col)
        elif col.startswith('Med_'):
            categories['Medications (Med)'].append(col)
        elif any(word in col_lower for word in ['bp', 'pressure', 'temp', 'pulse', 'heart', 'respiratory']):
            categories['Vital Signs'].append(col)
        elif any(word in col_lower for word in ['lab', 'test', 'result', 'level', 'count']):
            categories['Lab Results'].append(col)
        elif any(word in col_lower for word in ['date', 'time', 'created', 'updated']):
            categories['Dates & Times'].append(col)
        else:
            categories['Other'].append(col)
    
    # Remove empty categories
    return {k: v for k, v in categories.items() if v}

# Advanced filter builder
def create_filter_condition(df: pd.DataFrame, column: str, operator: str, value: Any) -> pd.Series:
    """Create a filter condition based on column type and operator."""
    if column not in df.columns:
        return pd.Series(True, index=df.index)
    
    col_data = df[column]
    
    if pd.api.types.is_numeric_dtype(col_data):
        if operator == "equals":
            return col_data == value
        elif operator == "not equals":
            return col_data != value
        elif operator == "greater than":
            return col_data > value
        elif operator == "less than":
            return col_data < value
        elif operator == "between":
            if isinstance(value, (list, tuple)) and len(value) == 2:
                return col_data.between(value[0], value[1])
        elif operator == "is null":
            return col_data.isna()
        elif operator == "is not null":
            return col_data.notna()
    
    elif pd.api.types.is_datetime64_any_dtype(col_data):
        if operator == "equals":
            return col_data == pd.to_datetime(value)
        elif operator == "not equals":
            return col_data != pd.to_datetime(value)
        elif operator == "greater than":
            return col_data > pd.to_datetime(value)
        elif operator == "less than":
            return col_data < pd.to_datetime(value)
        elif operator == "between":
            if isinstance(value, (list, tuple)) and len(value) == 2:
                return col_data.between(pd.to_datetime(value[0]), pd.to_datetime(value[1]))
        elif operator == "is null":
            return col_data.isna()
        elif operator == "is not null":
            return col_data.notna()
    
    else:  # String/categorical
        if operator == "equals":
            return col_data.astype(str) == str(value)
        elif operator == "not equals":
            return col_data.astype(str) != str(value)
        elif operator == "contains":
            return col_data.astype(str).str.contains(str(value), case=False, na=False)
        elif operator == "not contains":
            return ~col_data.astype(str).str.contains(str(value), case=False, na=False)
        elif operator == "starts with":
            return col_data.astype(str).str.startswith(str(value), na=False)
        elif operator == "ends with":
            return col_data.astype(str).str.endswith(str(value), na=False)
        elif operator == "in list":
            if isinstance(value, list):
                return col_data.astype(str).isin([str(v) for v in value])
        elif operator == "is null":
            return col_data.isna()
        elif operator == "is not null":
            return col_data.notna()
    
    return pd.Series(True, index=df.index)

def apply_advanced_filters(df: pd.DataFrame, filter_groups: List[Dict]) -> pd.DataFrame:
    """Apply advanced filters with AND/OR logic."""
    if not filter_groups:
        return df
    
    final_mask = pd.Series(False, index=df.index)
    
    for group in filter_groups:
        group_conditions = []
        for condition in group['conditions']:
            cond_mask = create_filter_condition(
                df, 
                condition['column'], 
                condition['operator'], 
                condition['value']
            )
            group_conditions.append(cond_mask)
        
        if group_conditions:
            # Apply AND logic within the group
            group_mask = pd.Series(True, index=df.index)
            for cond in group_conditions:
                group_mask &= cond
            
            # Apply OR logic between groups
            final_mask |= group_mask
    
    return df[final_mask]

# Initialize session state for filters
if 'filter_groups' not in st.session_state:
    st.session_state.filter_groups = []
if 'selected_columns' not in st.session_state:
    st.session_state.selected_columns = []

# Tabs for Data, Visualization, and Summary
data_tab, viz_tab, summary_tab = st.tabs(["Data Table", "Visualizations", "Summary Stats"])

with data_tab:
    st.write("### PilotData-WholeGlobe_Banked+Pending")
    
    # Snowflake Cortex AI-Powered Search
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "❄️ Snowflake Cortex AI Search", 
            placeholder="Try: 'diabetes patients over 50' or 'vision problems in females' or 'hypertension medication'",
            help="Native Snowflake Cortex AI for semantic search and natural language understanding."
        )
    
    with col2:
        search_type = st.selectbox(
            "Search Type",
            ["cortex", "hybrid", "basic"],
            help="Cortex: Pure AI search. Hybrid: AI + traditional filters. Basic: Fallback search."
        )
    
    # Show Cortex availability status
    if not cortex_available:
        st.warning("⚠️ Snowflake Cortex AI not available. Using basic search fallback.")
        st.info("To enable Cortex AI, contact your Snowflake administrator to enable Cortex features on your account.")
    
    # Show search suggestions
    if search_query and len(search_query) >= 2:
        suggestions = cortex_search_engine.get_search_suggestions(search_query)
        if suggestions:
            with st.expander("💡 Search Suggestions", expanded=False):
                for suggestion in suggestions:
                    st.write(f"• {suggestion}")
    
    # Perform Snowflake Cortex AI search
    if search_query:
        with st.spinner("❄️ Searching with Snowflake Cortex AI..."):
            filtered_df, search_metadata = cortex_search_engine.search(search_query, search_type)
        
        # Show search results summary
        if search_metadata.get('error'):
            st.error(f"❌ Search error: {search_metadata['error']}")
        else:
            st.success(f"✅ Found {len(filtered_df)} results for '{search_query}' using {search_type} search")
            
            # Show search explanation
            with st.expander("🔍 How Cortex AI interpreted your search", expanded=False):
                st.write("**Search Analysis:**")
                if search_metadata.get('interpretation'):
                    st.write(f"• {search_metadata['interpretation']}")
                
                if search_metadata.get('search_type'):
                    st.write(f"• Search Type: {search_metadata['search_type'].title()}")
                
                if search_metadata.get('search_score_available'):
                    st.write(f"• AI Relevance Scoring: {'Available' if search_metadata['search_score_available'] else 'Not available'}")
                
                if search_metadata.get('query_used'):
                    with st.expander("🔍 SQL Query Used", expanded=False):
                        st.code(search_metadata['query_used'], language='sql')
    else:
        filtered_df = data.copy()
    
    # Advanced filtering system in sidebar
    with st.sidebar:
        st.write("#### 🔍 Advanced Filters")
        
        # Column search and selection
        st.write("**1. Find Columns**")
        column_search = st.text_input("Search columns...", placeholder="Type to search columns")
        
        # Categorize columns
        categorized_cols = categorize_columns(filtered_df.columns.tolist())
        
        # Show categorized columns
        for category, columns in categorized_cols.items():
            if column_search:
                # Filter columns by search term
                matching_cols = [col for col in columns if column_search.lower() in col.lower()]
                if matching_cols:
                    st.write(f"**{category}**")
                    for col in matching_cols:
                        if st.button(f"➕ {col}", key=f"add_{col}"):
                            if col not in st.session_state.selected_columns:
                                st.session_state.selected_columns.append(col)
                                st.rerun()
            else:
                # Show first 5 columns in each category
                if columns:
                    st.write(f"**{category}** ({len(columns)} columns)")
                    for col in columns[:5]:
                        if st.button(f"➕ {col}", key=f"add_{col}"):
                            if col not in st.session_state.selected_columns:
                                st.session_state.selected_columns.append(col)
                                st.rerun()
                    if len(columns) > 5:
                        st.caption(f"... and {len(columns) - 5} more")
        
        st.divider()
        
        # Selected columns for filtering
        st.write("**2. Selected Columns**")
        if st.session_state.selected_columns:
            for i, col in enumerate(st.session_state.selected_columns):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"• {col}")
                with col2:
                    if st.button("❌", key=f"remove_{col}"):
                        st.session_state.selected_columns.remove(col)
                        st.rerun()
        else:
            st.caption("No columns selected. Search and add columns above.")
        
        st.divider()
        
        # Filter builder
        st.write("**3. Build Filters**")
        
        if st.session_state.selected_columns:
            # Add new filter group
            if st.button("➕ Add Filter Group"):
                st.session_state.filter_groups.append({
                    'conditions': [],
                    'logic': 'AND'
                })
                st.rerun()
            
            # Display existing filter groups
            for group_idx, group in enumerate(st.session_state.filter_groups):
                st.write(f"**Group {group_idx + 1}**")
                
                # Add condition to group
                col_name = st.selectbox(
                    "Column", 
                    st.session_state.selected_columns, 
                    key=f"col_{group_idx}"
                )
                
                if col_name in filtered_df.columns:
                    col_data = filtered_df[col_name]
                    
                    # Determine available operators based on column type
                    if pd.api.types.is_numeric_dtype(col_data):
                        operators = ["equals", "not equals", "greater than", "less than", "between", "is null", "is not null"]
                    elif pd.api.types.is_datetime64_any_dtype(col_data):
                        operators = ["equals", "not equals", "greater than", "less than", "between", "is null", "is not null"]
                    else:
                        operators = ["equals", "not equals", "contains", "not contains", "starts with", "ends with", "in list", "is null", "is not null"]
                    
                    operator = st.selectbox("Operator", operators, key=f"op_{group_idx}")
                    
                    # Value input based on operator and column type
                    if operator == "between":
                        if pd.api.types.is_numeric_dtype(col_data):
                            min_val = float(col_data.min()) if not col_data.isna().all() else 0.0
                            max_val = float(col_data.max()) if not col_data.isna().all() else 100.0
                            value = st.slider("Range", min_val, max_val, (min_val, max_val), key=f"val_{group_idx}")
                        elif pd.api.types.is_datetime64_any_dtype(col_data):
                            min_date = col_data.min().date() if not pd.isnull(col_data.min()) else datetime.date.today()
                            max_date = col_data.max().date() if not pd.isnull(col_data.max()) else datetime.date.today()
                            value = st.date_input("Date Range", (min_date, max_date), key=f"val_{group_idx}")
                    elif operator == "in list":
                        unique_vals = col_data.dropna().astype(str).unique().tolist()
                        value = st.multiselect("Select values", sorted(unique_vals), key=f"val_{group_idx}")
                    elif operator in ["is null", "is not null"]:
                        value = None
                    else:
                        if pd.api.types.is_numeric_dtype(col_data):
                            value = st.number_input("Value", key=f"val_{group_idx}")
                        elif pd.api.types.is_datetime64_any_dtype(col_data):
                            value = st.date_input("Date", key=f"val_{group_idx}")
                        else:
                            value = st.text_input("Value", key=f"val_{group_idx}")
                    
                    # Add condition button
                    if st.button("Add Condition", key=f"add_cond_{group_idx}"):
                        group['conditions'].append({
                            'column': col_name,
                            'operator': operator,
                            'value': value
                        })
                        st.rerun()
                    
                    # Show existing conditions in this group
                    if group['conditions']:
                        st.write("**Conditions:**")
                        for cond_idx, condition in enumerate(group['conditions']):
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                st.write(f"• {condition['column']} {condition['operator']} {condition['value']}")
                            with col2:
                                if st.button("❌", key=f"remove_cond_{group_idx}_{cond_idx}"):
                                    group['conditions'].pop(cond_idx)
                                    st.rerun()
                
                # Remove group button
                if st.button("Remove Group", key=f"remove_group_{group_idx}"):
                    st.session_state.filter_groups.pop(group_idx)
                    st.rerun()
                
                st.divider()
        
        # Clear all filters
        if st.session_state.filter_groups:
            if st.button("🗑️ Clear All Filters"):
                st.session_state.filter_groups = []
                st.rerun()
    
    # Apply advanced filters
    filtered_df = apply_advanced_filters(filtered_df, st.session_state.filter_groups)
    
    # Show number of rows after filtering
    st.write(f"**Total rows (after filters): {len(filtered_df)}**")
    
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
