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
import itertools
import streamlit.components.v1 as components

# Machine learning imports for clustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
import numpy as np

# UMAP is optional
try:
    import umap
    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False
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

@st.cache_data(show_spinner=False)
def col_profile(col: str):
    kind = META.loc[META['column']==col, 'kind'].iloc[0]
    if kind == 'number':
        lo, hi = col_minmax(col)
        return {'kind':'number', 'min': lo, 'max': hi}
    if kind == 'datetime':
        lo, hi = col_minmax(col)
        lo = pd.to_datetime(lo) if lo is not None else None
        hi = pd.to_datetime(hi) if hi is not None else None
        return {'kind':'datetime','min': lo, 'max': hi}
    # text
    vals = col_distinct(col, limit=2000)
    lower = {v.strip().lower() for v in vals}
    is_bool = lower.issubset({'0','1','yes','no','true','false'}) and len(lower) <= 4
    samples = vals[:20]
    return {'kind':'text','n_distinct': len(vals), 'samples': samples, 'is_bool': is_bool}
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
    valid_seen = False
    for i, c in enumerate(clauses):
        if not c.column or not c.op:
            continue
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

        if valid_seen:
            parts.append(c.join)
        parts.append(f"({sql})")
        params.extend(p)
        valid_seen = True
    return ("WHERE " + " ".join(parts)) if valid_seen else "", params

def fetch_data(clauses: List[Clause], columns: List[str] | None = None, limit: int = 5000):
    where, params = build_where(clauses)
    cols_sql = ", ".join(quote_ident(c) for c in columns) if columns else "*"
    # ❗ interpolate LIMIT directly (no bound param)
    sql = f"SELECT {cols_sql} FROM {quote_ident(TABLE)} {where} LIMIT {int(limit)}"
    cur = conn.cursor()
    cur.execute(sql, params)  # only data params are bound
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
data_tab, viz_tab, summary_tab, co_tab, clusters_tab = st.tabs(["Data Table", "Visualizations", "Summary Stats", "Co-occurrence Network", "Clusters"])

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
                q = st.text_input("Find a column", key=f"q_{i}", placeholder="type 2–3 letters…")
                subset = META if not q else META[META['column'].str.contains(q, case=False, na=False)]
                # Show live suggestions (first 20) to feel "predictive"
                if q:
                    st.caption("Suggestions: " + ", ".join(subset['column'].head(20).tolist()) if len(subset) else "No matches")

                if subset.empty:
                    st.warning("No columns match your search.")
                    continue  # don't render operator/value for this filter

                cl.column = st.selectbox(
                    "Column",
                    options=subset['column'].tolist(),
                    index=0 if cl.column not in subset['column'].tolist() else subset['column'].tolist().index(cl.column),
                    key=f"col_{i}"
                )

                if not cl.column:
                    continue

                profile = col_profile(cl.column)
                if profile['kind'] == 'number':
                    st.caption(f"Range: **{profile['min']} – {profile['max']}**")
                elif profile['kind'] == 'datetime':
                    lo = profile['min'].date() if profile['min'] is not None else '—'
                    hi = profile['max'].date() if profile['max'] is not None else '—'
                    st.caption(f"Date range: **{lo} – {hi}**")
                else:
                    if profile.get('is_bool'):
                        st.caption("Values: **Yes/No / True/False / 0/1**")
                    else:
                        st.caption(f"Distinct values: **{profile['n_distinct']}**; samples: {', '.join(map(str, profile['samples']))}")

                # operator + value widgets
                if profile['kind'] == 'number':
                    ops = ['between','=', '!=', '>', '>=', '<', '<=', 'isnull', 'notnull']
                elif profile['kind'] == 'datetime':
                    ops = ['between','=', '>=', '<=', 'isnull', 'notnull']
                else:
                    if profile.get('is_bool'):
                        ops = ['=','!=','isnull','notnull']
                    else:
                        ops = ['contains','=', '!=', 'in','isnull','notnull']
                cl.op = st.selectbox("Operator", ops, index=ops.index(cl.op) if cl.op in ops else 0, key=f"op_{i}")

                if profile['kind'] == 'number':
                    if cl.op == 'between':
                        lo, hi = col_minmax(cl.column)
                        lo = float(lo or 0.0); hi = float(hi or 0.0 if lo==0 else hi)
                        cl.value = st.slider("Range", lo, hi, (lo, hi), key=f"rng_{i}")
                    elif cl.op not in ('isnull','notnull'):
                        cl.value = st.number_input("Value", value=float(cl.value or 0.0), key=f"val_n_{i}")

                elif profile['kind'] == 'datetime':
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
                    if profile['kind']=='text' and profile.get('is_bool') and cl.op in ('=','!='):
                        cl.value = st.radio("Value", ['Yes','No'], horizontal=True, key=f"bool_{i}")
                    elif cl.op == 'in':
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

# ---------- Viz helpers ----------

def is_binary_text_values(vals: list[str]) -> bool:
    lower = {str(v).strip().lower() for v in vals}
    lower = {v for v in lower if v not in ("", "nan", "none", "null")}
    return lower.issubset({"0","1","yes","no","true","false"}) and 1 <= len(lower) <= 4

@st.cache_data(show_spinner=False)
def is_binary_numeric(col: str) -> bool:
    lo, hi = col_minmax(col)
    # treat as binary if min/max are 0/1 (allow None for all-null)
    return (lo in (0, 0.0) and hi in (1, 1.0))

def numeric_candidates_nonbinary():
    nums = META.loc[META['kind']=='number','column'].tolist()
    safe = []
    for c in nums:
        try:
            if not is_binary_numeric(c):
                safe.append(c)
        except Exception:
            # if profiling fails, keep it tentatively
            safe.append(c)
    return sorted(list(dict.fromkeys(safe)))

def categorical_candidates():
    # Start with known flags and standard categoricals
    cats = []
    for c, k in META[['column','kind']].values:
        if c.startswith(("Hx_","Sx_","Med_")):
            cats.append(c)
        elif k == 'datetime':
            continue
        elif k == 'number':
            try:
                if is_binary_numeric(c):
                    cats.append(c)
            except Exception:
                pass
        else:
            cats.append(c)
    return sorted(list(dict.fromkeys(cats)))

def fetch_cols_for_viz(cols: list[str], row_cap: int = 50000):
    # Reuse current filters but only pull the viz columns
    return fetch_data(st.session_state.clauses, columns=cols, limit=row_cap)

# Server-side category counts (accurate & fast)
def category_counts(col: str, top_n: int | None):
    where, params = build_where(st.session_state.clauses)
    colq = quote_ident(col)
    sql = f"SELECT {colq} AS v, COUNT(*) AS n FROM {quote_ident(TABLE)} {where} GROUP BY 1 ORDER BY n DESC"
    if top_n:
        sql += f" LIMIT {int(top_n)}"
    cur = conn.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()
    cur.close()
    return pd.DataFrame(rows, columns=['value','count'])

# Helpers to fetch numeric data with safe casts
def numeric_kind(col: str) -> str:
    # 'number' or 'text' from META
    return META.loc[META['column']==col, 'kind'].iloc[0]

def numeric_expr(col: str) -> str:
    # If stored as text, try to parse as double
    return f"TRY_TO_DOUBLE({quote_ident(col)})" if numeric_kind(col) == 'text' else quote_ident(col)

def fetch_numeric_series(col: str, cap: int = 50000):
    where, params = build_where(st.session_state.clauses)
    expr = numeric_expr(col)
    sql = f"""
        SELECT {expr} AS v
        FROM {quote_ident(TABLE)} {where}
        WHERE {expr} IS NOT NULL
        LIMIT {int(cap)}
    """
    cur = conn.cursor(); cur.execute(sql, params)
    df = pd.DataFrame(cur.fetchall(), columns=['v']); cur.close()
    return df

def fetch_numeric_xy(x_col: str, y_col: str, color_col: str | None, cap: int):
    where, params = build_where(st.session_state.clauses)
    x_expr, y_expr = numeric_expr(x_col), numeric_expr(y_col)
    color_sel = f", {quote_ident(color_col)} AS color" if color_col else ""
    sql = f"""
        SELECT {x_expr} AS x, {y_expr} AS y{color_sel}
        FROM {quote_ident(TABLE)} {where}
        WHERE {x_expr} IS NOT NULL AND {y_expr} IS NOT NULL
        LIMIT {int(cap)}
    """
    cur = conn.cursor(); cur.execute(sql, params)
    cols = ['x','y'] + (['color'] if color_col else [])
    df = pd.DataFrame(cur.fetchall(), columns=cols); cur.close()
    return df

# ---------- Visualizations ----------

with viz_tab:
    st.subheader("Visualizations")

    # Choose chart family
    chart_type = st.radio(
        "Pick a chart",
        ["Bar (count by category)", "Histogram (numeric)", "Scatter (numeric × numeric)", "Heatmap (category × category)"],
        horizontal=False
    )

    if chart_type == "Bar (count by category)":
        cats = categorical_candidates()
        if not cats:
            st.info("No categorical columns available.")
        else:
            cat = st.selectbox("Category column", cats, index=0)
            use_top = st.checkbox("Show top N categories only (for long tails)", value=True)
            topn = st.slider("Top N", 5, 100, 20) if use_top else None

            counts = category_counts(cat, topn)
            if counts.empty:
                st.warning("No data for this category and current filters.")
            else:
                st.caption("Counts computed on the full filtered set" + (f", showing top {topn}." if topn else "."))
                bar = alt.Chart(counts).mark_bar().encode(
                    x=alt.X('value:N', sort='-y', title=cat),
                    y=alt.Y('count:Q', title='Count'),
                    tooltip=['value', 'count']
                ).properties(title=f"Count of {cat}").interactive()
                st.altair_chart(bar, use_container_width=True)

    elif chart_type == "Histogram (numeric)":
        # Build numeric candidates
        strict_nums = META.loc[META['kind']=='number','column'].tolist()

        include_text_as_number = st.checkbox(
            "Include numeric-like text columns (auto-cast with TRY_TO_DOUBLE)",
            value=False,
            help="Enable to see text columns that contain mostly numbers (e.g., IDs)."
        )

        if include_text_as_number:
            # Offer a search box to keep the dropdown manageable
            qn = st.text_input("Filter numeric column list", placeholder="type to filter column names…")
            text_cols = META.loc[META['kind']=='text','column'].tolist()
            if qn:
                text_cols = [c for c in text_cols if qn.lower() in c.lower()]
            numeric_list = sorted(set(strict_nums + text_cols))
        else:
            numeric_list = sorted(strict_nums)

        if not numeric_list:
            st.info("No numeric (non-binary) columns found.")
        else:
            num = st.selectbox("Numeric column", numeric_list, index=numeric_list.index('Age') if 'Age' in numeric_list else 0)
            bins = st.slider("Bins", 5, 100, 30)
            # Fetch only this column (big row cap but limited to one column)
            dfh = fetch_numeric_series(num)  # <-- server-side cast if needed
            if dfh.empty or dfh['v'].dropna().empty:
                st.warning("No numeric data to plot.")
            else:
                hist = alt.Chart(dfh).mark_bar().encode(
                    x=alt.X('v:Q', bin=alt.Bin(maxbins=bins), title=num),
                    y='count()',
                    tooltip=['count()']
                ).properties(title=f"Histogram of {num}").interactive()
                st.altair_chart(hist, use_container_width=True)

    elif chart_type == "Scatter (numeric × numeric)":
        # Build numeric candidates
        strict_nums = META.loc[META['kind']=='number','column'].tolist()

        include_text_as_number = st.checkbox(
            "Include numeric-like text columns (auto-cast with TRY_TO_DOUBLE)",
            value=False,
            help="Enable to see text columns that contain mostly numbers (e.g., IDs)."
        )

        if include_text_as_number:
            # Offer a search box to keep the dropdown manageable
            qn = st.text_input("Filter numeric column list", placeholder="type to filter column names…")
            text_cols = META.loc[META['kind']=='text','column'].tolist()
            if qn:
                text_cols = [c for c in text_cols if qn.lower() in c.lower()]
            numeric_list = sorted(set(strict_nums + text_cols))
        else:
            numeric_list = sorted(strict_nums)

        if len(numeric_list) < 2:
            st.info("Pick filters that include at least two numeric columns (or enable numeric-like text).")
        else:
            c1, c2 = st.columns(2)
            with c1:
                x = st.selectbox("X", numeric_list, index=numeric_list.index('Age') if 'Age' in numeric_list else 0)
            with c2:
                y = st.selectbox("Y", [n for n in numeric_list if n != x] or numeric_list)

            color_by = st.selectbox("Color by (optional category)", ["(none)"] + categorical_candidates())
            cap = st.slider("Max points", 1000, 50000, 10000)

            df2 = fetch_numeric_xy(x, y, None if color_by=="(none)" else color_by, cap)
            if df2.empty:
                st.warning("No data to plot for these axes.")
            else:
                enc = {'x': alt.X('x:Q', title=x), 'y': alt.Y('y:Q', title=y), 'tooltip': ['x','y']}
                if color_by != "(none)":
                    enc['color'] = alt.Color('color:N', title=color_by)
                    enc['tooltip'].append('color')
                scatter = alt.Chart(df2).mark_circle(opacity=0.7).encode(**enc).properties(title=f"{x} vs {y}").interactive()
                try:
                    trend = alt.Chart(df2).transform_regression('x','y').mark_line()
                    st.altair_chart(scatter + trend, use_container_width=True)
                except Exception:
                    st.altair_chart(scatter, use_container_width=True)
                try:
                    r = df2[['x','y']].corr(numeric_only=True).iloc[0,1]
                    st.caption(f"Pearson r = **{r:.3f}**")
                except Exception:
                    pass

    else:  # Heatmap
        cats = categorical_candidates()
        if len(cats) < 2:
            st.info("Need at least two categorical columns for a heatmap.")
        else:
            row_cat = st.selectbox("Rows", cats)
            col_cat = st.selectbox("Columns", [c for c in cats if c != row_cat] or cats)
            where, params = build_where(st.session_state.clauses)
            sql = f"""
                SELECT {quote_ident(row_cat)} AS r, {quote_ident(col_cat)} AS c, COUNT(*) AS n
                FROM {quote_ident(TABLE)} {where}
                GROUP BY 1,2
            """
            cur = conn.cursor(); cur.execute(sql, params)
            rows = cur.fetchall(); cur.close()
            dfh = pd.DataFrame(rows, columns=['r','c','n'])
            if dfh.empty:
                st.warning("No data for these categories.")
            else:
                heat = alt.Chart(dfh).mark_rect().encode(
                    x=alt.X('c:N', title=col_cat, sort='-y'),
                    y=alt.Y('r:N', title=row_cat, sort='-x'),
                    color=alt.Color('n:Q', title='Count'),
                    tooltip=[alt.Tooltip('r:N', title=row_cat), alt.Tooltip('c:N', title=col_cat), 'n:Q']
                ).properties(title=f"Count heatmap: {row_cat} × {col_cat}")
                st.altair_chart(heat, use_container_width=True)

with summary_tab:
    st.write("### Summary Statistics (based on filtered data)")
    try:
        st.write(filtered_df.describe(include='all'))
    except Exception as e:
        st.warning(f"Error generating summary statistics: {str(e)}")

with co_tab:
    st.subheader("Disease / Medication Co-occurrence")

    # Check if pyvis is available
    try:
        from pyvis.network import Network
        pyvis_available = True
    except ImportError as e:
        st.error(f"⚠️ Pyvis library not available: {str(e)}")
        st.info("This feature requires the pyvis library for network visualizations.")
        st.code("pip install pyvis>=0.3.2")
        pyvis_available = False
    except Exception as e:
        st.error(f"⚠️ Error loading pyvis: {str(e)}")
        st.info("This feature requires the pyvis library for network visualizations.")
        pyvis_available = False

    if not pyvis_available:
        st.stop()

    try:
        # pick flag columns
        flag_cols = [c for c in META['column'].tolist() if c.startswith(("Hx_","Sx_","Med_"))]
        if not flag_cols:
            st.info("No Hx_/Sx_/Med_ columns found.")
        else:
            # Right-size the "Max rows" slider + add one-line help
            total_rows = fetch_count(st.session_state.clauses)
            row_cap_max = min(total_rows, 2000)  # <= 2k as you asked
            row_cap = st.slider(
                "Max rows to analyze",
                min_value=200,
                max_value=int(row_cap_max) if row_cap_max >= 200 else int(total_rows),
                value=min(1000, int(row_cap_max)),
                step=100,
                help="Upper limit of filtered rows used to compute co-occurrence. Smaller = faster; larger = more stable."
            )
            
            df_flags = fetch_cols_for_viz(flag_cols, row_cap=row_cap)

            if df_flags.empty:
                st.warning("No data available for analysis with current filters.")
                st.stop()

            # normalize to boolean
            def to_bool(s: pd.Series) -> pd.Series:
                s2 = s.astype(str).str.strip().str.lower()
                return s2.isin(["1","yes","true","y","t"])

            B = pd.DataFrame({c: to_bool(df_flags[c]) for c in df_flags.columns})

            # prevalence and pick top-N for readability
            prev = B.mean(numeric_only=True).sort_values(ascending=False)
            topN = st.slider(
                "Top N flags by prevalence",
                10, min(200, len(prev)), 50,
                help="Show only the N most common flags (nodes). Use 30–60 for a readable graph."
            )
            top_flags = prev.head(topN).index.tolist()
            B = B[top_flags]

            # co-occurrence via Jaccard
            counts = B.sum()
            edges = []
            for a, b in itertools.combinations(top_flags, 2):
                inter = (B[a] & B[b]).sum()
                union = (B[a] | B[b]).sum()
                j = (inter / union) if union else 0.0
                if j > 0:
                    edges.append((a, b, j, int(inter)))

            if not edges:
                st.info("No meaningful co-occurrences in the current selection.")
            else:
                min_j = st.slider(
                    "Min Jaccard to show",
                    0.0, 1.0, 0.10, 0.01,
                    help="Hide weak edges. Jaccard = (rows with A & B) / (rows with A or B). 0.10–0.15 is a good start."
                )
                edges = [e for e in edges if e[2] >= min_j]

                if not edges:
                    st.info("No edges meet the minimum Jaccard threshold.")
                else:
                    # build network
                    net = Network(height="650px", width="100%", notebook=True, cdn_resources="in_line", directed=False)
                    # nodes with size by prevalence
                    for n in top_flags:
                        size = 10 + 40 * float(prev[n])  # scale
                        net.add_node(n, label=n, title=f"Prevalence: {prev[n]:.2%}", value=size)
                    # edges with width by jaccard
                    for a,b,j,co in edges:
                        net.add_edge(a, b, value=1 + 10*j, title=f"Co-occurrence: {co} | Jaccard {j:.2f}")

                    # render
                    html = net.generate_html()
                    components.html(html, height=680, scrolling=True)
                    st.caption("Node size = prevalence in filtered rows. Edge width = Jaccard similarity. Hover for values.")
    except Exception as e:
        st.error(f"Error in co-occurrence analysis: {str(e)}")
        st.info("Please try adjusting your filters or contact support if the issue persists.")

# ============ CLUSTERS TAB ============
with clusters_tab:
    st.subheader("Dimensionality, Clusters & Similarity")

    with st.expander("What is this?"):
        st.markdown(
            """
- **Goal:** place each sample in a 2-D map so you can *see cohorts*, *spot outliers*, and *recommend similar or diverse samples*.
- **How it works:** we convert Hx_/Sx_/Med_ flags to 0/1 and combine them with numeric features (e.g., Age).  
  We standardize the matrix and project it with **PCA** (linear, interpretable) or **UMAP** (non-linear, better separation).  
  We then overlay **K-Means** clusters.
- **Use cases:**  
  • **Similar picks** – "This sold sample lives in Cluster 3; recommend 5 nearest neighbors."  
  • **Diversity picks** – "Give me two *very different* samples (far apart)."  
  • **Explainability** – PCA lists **top contributing features** (flags/meds) driving axes.  
            """
        )

    total_rows = fetch_count(st.session_state.clauses)

    # --- choose features ---
    flag_cols_all = [c for c in META['column'].tolist() if c.startswith(("Hx_","Sx_","Med_"))]
    num_cols_all  = META.loc[META['kind']=='number','column'].tolist()

    row_cap = st.slider(
        "Max rows to pull",
        min_value=300,
        max_value=int(min(2000, max(300, total_rows))),  # cap to 2k as requested
        value=int(min(1000, total_rows)),
        step=100,
        help="Rows sampled (server-side) from the current filters for analysis. Larger = more stable, slower."
    )

    topN_flags = st.slider(
        "Top N flags by prevalence",
        min_value=20,
        max_value=int(min(300, len(flag_cols_all) or 20)),
        value=100,
        help="Use the most common flags to reduce sparsity/noise (30–150 is a good range)."
    )

    default_nums = [c for c in ["Age","Death To Preservation","Amount"] if c in num_cols_all]
    picked_nums = st.multiselect(
        "Numeric features (optional)",
        options=sorted(num_cols_all),
        default=default_nums,
        help="Add numeric columns to the feature matrix (scaled automatically)."
    )

    color_by = st.selectbox(
        "Color points by (category)",
        ["(none)","GENDER","OD_OS","Patient Race","Research Project"] + flag_cols_all,
        help="Purely visual grouping; doesn't affect clustering."
    )

    algo = st.selectbox(
        "Projection algorithm",
        ["PCA","UMAP (non-linear)"] if HAVE_UMAP else ["PCA"],
        help="Start with PCA to understand variance and drivers; switch to UMAP for tighter cohort separation."
    )

    k_clusters = st.slider(
        "K-Means: number of clusters (k)",
        2, 15, 5, help="Overlay k clusters on the 2-D map."
    )

    n_neighbors = st.slider(
        "Neighbors to recommend (similar samples)",
        1, 25, 8, help="Number of nearest neighbors to return for a selected sample."
    )

    # --- SQL helpers for building feature matrix ---
    def flag_expr(col):  # Yes/True/1 -> 1 else 0
        return f"CASE WHEN LOWER(TRIM({quote_ident(col)})) IN ('1','yes','true','y','t') THEN 1 ELSE 0 END AS {quote_ident(col)}"

    def num_expr(col):
        return f"{quote_ident(col)} AS {quote_ident(col)}"

    # prevalence to pick top-N flags (fast SQL AVG of 0/1)
    where, params = build_where(st.session_state.clauses)
    if flag_cols_all:
        col_list = ", ".join(
            [f"AVG(CASE WHEN LOWER(TRIM({quote_ident(c)})) IN ('1','yes','true','y','t') THEN 1 ELSE 0 END) AS {quote_ident(c)}"
             for c in flag_cols_all]
        )
        sql_prev = f"SELECT {col_list} FROM {quote_ident(TABLE)} {where}"
        cur = conn.cursor(); cur.execute(sql_prev, params); prev_row = cur.fetchone(); cur.close()
        prev = pd.Series(prev_row, index=flag_cols_all).sort_values(ascending=False)
        top_flags = prev.head(topN_flags).index.tolist()
    else:
        prev = pd.Series(dtype=float)
        top_flags = []

    # Build SELECT for the working dataset
    select_bits = []
    # A stable ID helps drill-down later
    sid_col = "SAMPLE_ID" if "SAMPLE_ID" in META['column'].tolist() else META['column'].tolist()[0]
    select_bits.append(f"{quote_ident(sid_col)} AS SID")
    for c in top_flags:
        select_bits.append(flag_expr(c))
    for c in picked_nums:
        select_bits.append(num_expr(c))
    if color_by and color_by != "(none)":
        select_bits.append(f"{quote_ident(color_by)} AS COLOR")

    if len(select_bits) <= 1:
        st.info("Pick at least one feature (flags and/or numeric).")
        st.stop()

    sql_data = f"""
        SELECT {", ".join(select_bits)}
        FROM {quote_ident(TABLE)} {where}
        LIMIT {int(row_cap)}
    """
    cur = conn.cursor(); cur.execute(sql_data, params)
    cols = [c[0] for c in cur.description]
    DF = pd.DataFrame.from_records(iter(cur), columns=cols); cur.close()

    if DF.empty:
        st.warning("No rows returned for the current filters.")
        st.stop()

    # --- Build X matrix ---
    feat_cols = top_flags + picked_nums
    X = DF[feat_cols].copy()
    # Coerce to numeric (flags are already 0/1 from SQL)
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0.0)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)

    # --- Projection ---
    if algo.startswith("PCA"):
        pca = PCA(n_components=min(10, Xs.shape[1]), random_state=42)
        Z = pca.fit_transform(Xs)
        expl = pca.explained_variance_ratio_

        with st.expander("PCA • explained variance & loadings", expanded=False):
            scree = pd.DataFrame({"PC": [f"PC{i+1}" for i in range(len(expl))], "Variance": expl})
            sc = alt.Chart(scree).mark_bar().encode(
                x='PC', y=alt.Y('Variance:Q', axis=alt.Axis(format='%')),
                tooltip=[alt.Tooltip('Variance:Q', format='.1%'), 'PC']
            ).properties(title="Scree plot")
            st.altair_chart(sc, use_container_width=True)

            loadings = pd.DataFrame(pca.components_[:2, :], columns=feat_cols, index=['PC1','PC2']).T
            top_load = (loadings.abs().sum(axis=1).sort_values(ascending=False)).head(20).index.tolist()
            st.write("Top contributing features to PC1/PC2")
            st.dataframe(loadings.loc[top_load])
        coords = pd.DataFrame({"X": Z[:,0], "Y": Z[:,1], "SID": DF["SID"]})

    else:  # UMAP
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
        Z2 = reducer.fit_transform(Xs)
        coords = pd.DataFrame({"X": Z2[:,0], "Y": Z2[:,1], "SID": DF["SID"]})
        st.caption("UMAP axes are unitless; use them for pattern discovery rather than interpretation.")

    # Coloring / tooltip
    if "COLOR" in DF.columns:
        coords["COLOR"] = DF["COLOR"]
        color_field = "COLOR"
    else:
        color_field = None

    # --- KMeans overlay ---
    km = KMeans(n_clusters=k_clusters, n_init="auto", random_state=42)
    coords["Cluster"] = km.fit_predict(coords[["X","Y"]])

    # Plot
    enc = {
        "x": alt.X("X:Q", title="Component 1"),
        "y": alt.Y("Y:Q", title="Component 2"),
        "shape": alt.Shape("Cluster:N"),
        "tooltip": ["SID","Cluster"]
    }
    if color_field:
        enc["color"] = alt.Color(f"{color_field}:N", legend=alt.Legend(title=color_field))
        enc["tooltip"].append(color_field)

    chart = alt.Chart(coords).mark_circle(size=70, opacity=0.75).encode(**enc)\
        .properties(title=f"{algo}: 2-D projection with K-Means overlay").interactive()
    st.altair_chart(chart, use_container_width=True)

    # --- Cluster sizes ---
    ccounts = coords["Cluster"].value_counts().rename_axis("Cluster").reset_index(name="Count").sort_values("Cluster")
    st.write("Cluster sizes")
    st.dataframe(ccounts, use_container_width=True)

    # --- Similar samples (Nearest Neighbors) ---
    st.markdown("### Recommend similar samples")
    sid_options = coords["SID"].astype(str).tolist()
    sid_pick = st.selectbox(
        "Pick a sample (SID) to find nearest neighbors",
        sid_options,
        help="Returns nearest neighbors in the 2-D space (PCA/UMAP)."
    )
    nbrs = NearestNeighbors(n_neighbors=min(n_neighbors+1, len(coords)), metric='euclidean')
    nbrs.fit(coords[["X","Y"]].values)
    idx = coords.index[coords["SID"].astype(str) == sid_pick][0]
    dists, inds = nbrs.kneighbors(coords.loc[[idx], ["X","Y"]].values)
    inds = inds[0].tolist()
    dists = dists[0].tolist()
    # drop self
    if idx in inds:
        j = inds.index(idx)
        inds.pop(j); dists.pop(j)
    rec = coords.iloc[inds].copy()
    rec["distance"] = dists
    rec = rec[["SID","Cluster"] + ([color_field] if color_field else []) + ["distance","X","Y"]]
    st.dataframe(rec, use_container_width=True)
    st.download_button("Download neighbors (CSV)", rec.to_csv(index=False).encode("utf-8"), file_name=f"neighbors_{sid_pick}.csv")

    # --- Diversity pick: find a very different pair ---
    st.markdown("### Find a very different pair (diverse pick)")
    st.caption("We look among the most extreme points along the axes and return a far pair (approximate, fast).")
    extreme_k = st.slider("Extreme pool size", 20, min(200, len(coords)), 80,
                          help="We take this many extreme points (by |X|+|Y|) and search the farthest pair inside.")
    coords["_ext"] = (coords["X"].abs() + coords["Y"].abs())
    pool = coords.nlargest(extreme_k, "_ext")[["SID","X","Y","Cluster"] + ([color_field] if color_field else [])].reset_index(drop=True)
    if len(pool) >= 2:
        M = pool[["X","Y"]].values
        D = pairwise_distances(M, metric="euclidean")
        i, j = np.unravel_index(np.argmax(D), D.shape)
        far = pool.iloc[[i,j]].copy()
        far["pair_distance"] = D[i, j]
        st.dataframe(far)
        st.download_button("Download pair (CSV)", far.to_csv(index=False).encode("utf-8"), file_name="diverse_pair.csv")

    # --- Enrichment: what characterizes each cluster? (quick lift over overall) ---
    st.markdown("### Cluster enrichment (top flags per cluster)")
    # We have flags in X; map rows back to clusters
    W = pd.DataFrame(X, columns=feat_cols)  # numeric matrix (flags are 0/1)
    W["Cluster"] = coords["Cluster"].values
    overall = W[top_flags].mean().replace(0, np.nan)  # avoid div/0
    show_k = st.slider("Top features to display per cluster", 3, 15, 8)
    for cl in sorted(coords["Cluster"].unique()):
        sub = W[W["Cluster"] == cl]
        prev_cl = sub[top_flags].mean()
        lift = (prev_cl / overall).sort_values(ascending=False).head(show_k)
        st.write(f"**Cluster {cl}** — top features (lift over overall prevalence)")
        st.dataframe(pd.DataFrame({"prevalence": prev_cl[lift.index].round(3), "lift": lift.round(2)}))
