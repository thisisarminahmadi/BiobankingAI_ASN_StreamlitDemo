# Import packages
import streamlit as st
import pandas as pd
import altair as alt
import datetime
from snowflake.connector import connect
import os

# --------- imports (add these) ----------
from dataclasses import dataclass
from typing import Any, List, Tuple
# ----------------------------------------

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

# ---------- connection (cache) ----------
@st.cache_resource(show_spinner=False)
def get_conn():
    return connect(
        account=st.secrets["connections"]["snowflake"]["account"],
        user=st.secrets["connections"]["snowflake"]["user"],
        password=st.secrets["connections"]["snowflake"]["password"],
        database=st.secrets["connections"]["snowflake"]["database"],
        schema=st.secrets["connections"]["snowflake"]["schema"],
        warehouse=st.secrets["connections"]["snowflake"]["warehouse"],
        role=st.secrets["connections"]["snowflake"]["role"],
    )

conn = get_conn()
DB = st.secrets["connections"]["snowflake"]["database"]
SCHEMA = st.secrets["connections"]["snowflake"]["schema"]
TABLE = "ASN"  # adjust if needed
# ----------------------------------------

# --------- metadata helpers -------------
def quote_ident(identifier: str) -> str:
    # for names with spaces / mixed case (e.g., Patient Race)
    return '"' + identifier.replace('"', '""') + '"'

@st.cache_data(show_spinner=False)
def get_column_meta():
    sql = """
    SELECT COLUMN_NAME, DATA_TYPE
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_CATALOG=%s AND TABLE_SCHEMA=%s AND TABLE_NAME=%s
    ORDER BY ORDINAL_POSITION
    """
    cur = conn.cursor()
    cur.execute(sql, [DB, SCHEMA, TABLE])
    rows = cur.fetchall()
    cur.close()

    meta = pd.DataFrame(rows, columns=['column','dtype'])
    def kind(dt):
        u = str(dt).upper()
        if any(x in u for x in ['INT','DECIMAL','NUMBER','FLOAT','DOUBLE']):
            return 'number'
        if any(x in u for x in ['DATE','TIME','TIMESTAMP']):
            return 'datetime'
        return 'text'
    meta['kind'] = meta['dtype'].map(kind)
    return meta

META = get_column_meta()

@st.cache_data(show_spinner=False)
def col_minmax(col: str):
    cur = conn.cursor()
    cur.execute(f"SELECT MIN({quote_ident(col)}), MAX({quote_ident(col)}) FROM {quote_ident(TABLE)}")
    lo, hi = cur.fetchone()
    cur.close()
    return lo, hi

@st.cache_data(show_spinner=False)
def col_distinct(col: str, limit: int = 2000):
    cur = conn.cursor()
    cur.execute(f"SELECT DISTINCT {quote_ident(col)} FROM {quote_ident(TABLE)} LIMIT {limit}")
    vals = [r[0] for r in cur.fetchall() if r[0] is not None]
    cur.close()
    # convert to str for robust matching in 'IN'
    return sorted(map(str, vals))
# ----------------------------------------

# ------------- query builder ------------
@dataclass
class Clause:
    column: str
    op: str              # '=', '!=', 'contains', 'in', 'between', 'isnull', 'notnull', '>', '>=', '<', '<='
    value: Any = None    # str | float | (low, high) | list
    join: str = "AND"    # 'AND' | 'OR' (ignored on first clause)

OP_SQL = {
    '=': '{col} = %s',
    '!=': '{col} != %s',
    '>': '{col} > %s',
    '>=': '{col} >= %s',
    '<': '{col} < %s',
    '<=': '{col} <= %s',
    'contains': '{col} ILIKE %s',           # %value%
    'in': '{col} IN ({ph})',                # list
    'between': '{col} BETWEEN %s AND %s',
    'isnull': '{col} IS NULL',
    'notnull': '{col} IS NOT NULL',
}

def build_where(clauses: List[Clause]) -> Tuple[str, List[Any]]:
    if not clauses:
        return "", []
    parts, params = [], []
    for i, c in enumerate(clauses):
        col = quote_ident(c.column)
        if c.op == 'in':
            vals = c.value if isinstance(c.value, list) else [c.value]
            ph = ','.join(['%s']*len(vals))
            sql = OP_SQL['in'].format(col=col, ph=ph)
            p = vals
        elif c.op == 'contains':
            sql = OP_SQL['contains'].format(col=col)
            p = [f"%{c.value}%"]
        elif c.op == 'between':
            sql = OP_SQL['between'].format(col=col)
            low, high = c.value
            p = [low, high]
        elif c.op in ('isnull','notnull'):
            sql = OP_SQL[c.op].format(col=col)
            p = []
        else:
            sql = OP_SQL[c.op].format(col=col)
            p = [c.value]

        if i > 0:
            parts.append(c.join)
        parts.append(f"({sql})")
        params.extend(p)

    return "WHERE " + " ".join(parts), params

def fetch_data(clauses: List[Clause], columns: List[str] | None = None, limit: int = 5000):
    where, params = build_where(clauses)
    cols_sql = ", ".join(quote_ident(c) for c in columns) if columns else "*"
    sql = f"SELECT {cols_sql} FROM {quote_ident(TABLE)} {where} LIMIT %s"
    cur = conn.cursor()
    cur.execute(sql, params + [limit])
    df = pd.DataFrame.from_records(iter(cur), columns=[c[0] for c in cur.description])
    cur.close()
    return df

def fetch_count(clauses: List[Clause]):
    where, params = build_where(clauses)
    sql = f"SELECT COUNT(*) FROM {quote_ident(TABLE)} {where}"
    cur = conn.cursor()
    cur.execute(sql, params)
    n = cur.fetchone()[0]
    cur.close()
    return int(n)
# ----------------------------------------

# ---------------- data tab --------------
data_tab, viz_tab, summary_tab = st.tabs(["Data Table", "Visualizations", "Summary Stats"])

with data_tab:
    st.write("### PilotData-WholeGlobe_Banked+Pending")

    # Legacy global search: removed for now (Cortex later)

    # session for UI-built clauses
    if "clauses" not in st.session_state:
        st.session_state.clauses: list[Clause] = []

    # ===== Sidebar: Smart Filters =====
    with st.sidebar:
        st.subheader("Smart Filters")
        st.caption("AND/OR chaining • type-aware • Snowflake-backed")

        # add / clear
        c1, c2 = st.columns(2)
        with c1:
            if st.button("➕ Add filter", use_container_width=True):
                st.session_state.clauses.append(Clause(column="", op="=", value=None, join="AND"))
        with c2:
            if st.button("🧹 Clear all", use_container_width=True):
                st.session_state.clauses.clear()
                st.experimental_rerun()

        # render each filter
        for i, cl in enumerate(st.session_state.clauses):
            with st.expander(f"Filter {i+1}", expanded=True):
                # join (except first)
                if i > 0:
                    cl.join = st.radio("Join with previous", ["AND","OR"], index=0 if cl.join=="AND" else 1,
                                       key=f"join_{i}", horizontal=True)

                # searchable column picker
                q = st.text_input("Find a column", key=f"q_{i}", placeholder="type to search…")
                subset = META if not q else META[META['column'].str.contains(q, case=False, na=False)]
                cl.column = st.selectbox("Column", options=subset['column'].tolist(),
                                         index=(subset['column'].tolist().index(cl.column)
                                                if cl.column in subset['column'].tolist() else 0) if len(subset)>0 else None,
                                         key=f"col_{i}")

                if not cl.column:
                    continue

                kind = META.loc[META['column']==cl.column, 'kind'].iloc[0]

                # operator + value widgets
                if kind == 'number':
                    ops = ['between','=', '!=', '>', '>=', '<', '<=', 'isnull', 'notnull']
                    cl.op = st.selectbox("Operator", ops, index=ops.index(cl.op) if cl.op in ops else 0, key=f"op_n_{i}")
                    if cl.op == 'between':
                        lo, hi = col_minmax(cl.column)
                        lo = float(lo or 0.0); hi = float(hi or 0.0 if lo==0 else hi)
                        cl.value = st.slider("Range", lo, hi, (lo, hi), key=f"rng_{i}")
                    elif cl.op not in ('isnull','notnull'):
                        cl.value = st.number_input("Value", value=float(cl.value or 0.0), key=f"val_n_{i}")

                elif kind == 'datetime':
                    ops = ['between','=', '>=', '<=', 'isnull', 'notnull']
                    cl.op = st.selectbox("Operator", ops, index=ops.index(cl.op) if cl.op in ops else 0, key=f"op_d_{i}")
                    if cl.op == 'between':
                        lo, hi = col_minmax(cl.column)
                        lo = pd.to_datetime(lo) if lo is not None else pd.Timestamp("1900-01-01")
                        hi = pd.to_datetime(hi) if hi is not None else pd.Timestamp.today().normalize()
                        d1, d2 = st.date_input("Date range", (lo.date(), hi.date()), key=f"dt_{i}")
                        cl.value = (pd.to_datetime(d1), pd.to_datetime(d2))
                    elif cl.op not in ('isnull','notnull'):
                        d = st.date_input("Date", key=f"dt1_{i}")
                        cl.value = pd.to_datetime(d)

                else:  # text
                    ops = ['contains','=', '!=', 'in', 'isnull', 'notnull']
                    cl.op = st.selectbox("Operator", ops, index=ops.index(cl.op) if cl.op in ops else 0, key=f"op_t_{i}")
                    if cl.op == 'in':
                        options = col_distinct(cl.column)
                        sel = st.multiselect("Values", options=options, default=cl.value or [], key=f"in_{i}")
                        cl.value = sel
                    elif cl.op not in ('isnull','notnull'):
                        cl.value = st.text_input("Value", value=str(cl.value or ""), key=f"val_t_{i}")

                # remove button
                if st.button(f"Remove filter #{i+1}", key=f"rm_{i}"):
                    st.session_state.clauses.pop(i)
                    st.experimental_rerun()

    # ===== Fetch & show data =====
    # (Optional) visible columns picker to keep tables light
    with st.expander("Visible columns", expanded=False):
        default_cols = ['SAMPLE_ID','OD_OS','GENDER','Patient Race']
        existing_defaults = [c for c in default_cols if c in META['column'].tolist()]
        visible_cols = st.multiselect("Columns to select (optional)", options=META['column'].tolist(),
                                      default=existing_defaults)
        if len(visible_cols) == 0:
            visible_cols = None  # selects all

    filtered_df = fetch_data(st.session_state.clauses, columns=visible_cols, limit=5000)
    total_rows = fetch_count(st.session_state.clauses)

    st.write(f"Rows matched: **{total_rows}**  |  Showing: **{len(filtered_df)}**")
    st.dataframe(filtered_df, use_container_width=True, height=420)
# -------------- end data tab -----------

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
