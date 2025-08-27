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

with viz_tab:
    st.subheader("Explore (based on current filters)")

    if filtered_df.empty:
        st.info("No data to visualize. Adjust filters.")
    else:
        cols = filtered_df.columns.tolist()
        num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(filtered_df[c])]
        cat_cols = [c for c in cols if filtered_df[c].nunique(dropna=True) <= 50]

        chart_type = st.selectbox(
            "Chart type",
            ["Bar (count by category)", "Histogram", "Scatter", "Heatmap (count)"]
        )

        if chart_type == "Bar (count by category)":
            if not cat_cols:
                st.warning("No categorical columns (≤50 distinct) in the current data.")
            else:
                cat = st.selectbox("Category", cat_cols)
                topk = st.slider("Top K", 5, 100, 20)
                vc = (filtered_df[cat].astype(str)
                      .value_counts(dropna=True)
                      .head(topk)
                      .reset_index())
                vc.columns = [cat, 'count']
                bar = alt.Chart(vc).mark_bar().encode(
                    x=alt.X(cat, sort='-y'),
                    y='count',
                    tooltip=[cat, 'count']
                ).properties(title=f"Count of {cat} (Top {topk})").interactive()
                st.altair_chart(bar, use_container_width=True)

        elif chart_type == "Histogram":
            if not num_cols:
                st.warning("No numeric columns in the current data.")
            else:
                num = st.selectbox("Numeric", num_cols)
                bins = st.slider("Bins", 5, 100, 30)
                hist = alt.Chart(filtered_df).mark_bar().encode(
                    x=alt.X(num, bin=alt.Bin(maxbins=bins)),
                    y='count()',
                    tooltip=['count()']
                ).properties(title=f"Histogram of {num}").interactive()
                st.altair_chart(hist, use_container_width=True)

        elif chart_type == "Scatter":
            if len(num_cols) < 2:
                st.warning("Need at least two numeric columns for a scatter plot.")
            else:
                x = st.selectbox("X (numeric)", num_cols)
                y = st.selectbox("Y (numeric)", [c for c in num_cols if c != x] or num_cols)
                color_opt = ['(none)'] + cat_cols
                color = st.selectbox("Color by (optional)", color_opt)
                enc = {'x': x, 'y': y, 'tooltip': [x, y]}
                if color != '(none)':
                    enc['color'] = color
                    enc['tooltip'].append(color)
                scatter = alt.Chart(filtered_df).mark_circle().encode(**enc)\
                    .properties(title=f"{x} vs {y}").interactive()
                st.altair_chart(scatter, use_container_width=True)

        else:  # Heatmap
            if len(cat_cols) < 2:
                st.warning("Need at least two categorical columns (≤50 distinct) for a heatmap.")
            else:
                r = st.selectbox("Rows (category)", cat_cols)
                c = st.selectbox("Columns (category)", [k for k in cat_cols if k != r] or cat_cols)
                pivot = filtered_df.pivot_table(index=r, columns=c, aggfunc='size', fill_value=0)
                melt = pivot.reset_index().melt(id_vars=r, var_name=c, value_name='count')
                heat = alt.Chart(melt).mark_rect().encode(
                    x=alt.X(c, sort='-y'),
                    y=alt.Y(r, sort='-x'),
                    color='count',
                    tooltip=[r, c, 'count']
                ).properties(title=f"Count heatmap: {r} × {c}")
                st.altair_chart(heat, use_container_width=True)

with summary_tab:
    st.write("### Summary Statistics (based on filtered data)")
    try:
        st.write(filtered_df.describe(include='all'))
    except Exception as e:
        st.warning(f"Error generating summary statistics: {str(e)}")
