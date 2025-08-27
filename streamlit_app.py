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
from pyvis.network import Network
import streamlit.components.v1 as components
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
data_tab, viz_tab, summary_tab, co_tab = st.tabs(["Data Table", "Visualizations", "Summary Stats", "Co-occurrence Network"])

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
        nums = numeric_candidates_nonbinary()
        if not nums:
            st.info("No numeric (non-binary) columns found.")
        else:
            num = st.selectbox("Numeric column", nums, index=nums.index('Age') if 'Age' in nums else 0)
            bins = st.slider("Bins", 5, 100, 30)
            # Fetch only this column (big row cap but limited to one column)
            dfh = fetch_cols_for_viz([num])
            if dfh.empty or dfh[num].dropna().empty:
                st.warning("No numeric data to plot.")
            else:
                hist = alt.Chart(dfh).mark_bar().encode(
                    x=alt.X(num, bin=alt.Bin(maxbins=bins)),
                    y='count()',
                    tooltip=['count()']
                ).properties(title=f"Histogram of {num}").interactive()
                st.altair_chart(hist, use_container_width=True)

    elif chart_type == "Scatter (numeric × numeric)":
        nums = numeric_candidates_nonbinary()
        if len(nums) < 2:
            st.info("Pick filters that include at least two non-binary numeric columns.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                x = st.selectbox("X", nums, index=nums.index('Age') if 'Age' in nums else 0)
            with c2:
                y = st.selectbox("Y", [n for n in nums if n != x] or nums)

            color_by = st.selectbox("Color by (optional category)", ["(none)"] + categorical_candidates())
            sample_cap = st.slider("Max points", 1000, 50000, 10000, help="Data pulled server-side; higher = denser plot")

            df2 = fetch_cols_for_viz([x, y] + ([] if color_by == "(none)" else [color_by]), row_cap=sample_cap)
            df2 = df2.dropna(subset=[x,y])
            if df2.empty:
                st.warning("No data to plot for these axes.")
            else:
                enc = {'x': x, 'y': y, 'tooltip': [x, y]}
                if color_by != "(none)":
                    enc['color'] = color_by
                    enc['tooltip'].append(color_by)

                scatter = alt.Chart(df2).mark_circle(opacity=0.7).encode(**enc)\
                    .properties(title=f"{x} vs {y}").interactive()

                # Trendline + correlation
                try:
                    trend = alt.Chart(df2).transform_regression(x, y).mark_line()
                    st.altair_chart(scatter + trend, use_container_width=True)
                except Exception:
                    st.altair_chart(scatter, use_container_width=True)

                # Pearson r
                try:
                    r = df2[[x,y]].corr(numeric_only=True).iloc[0,1]
                    st.caption(f"Pearson r = **{r:.3f}** (based on sampled rows)")
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

    # pick flag columns
    flag_cols = [c for c in META['column'].tolist() if c.startswith(("Hx_","Sx_","Med_"))]
    if not flag_cols:
        st.info("No Hx_/Sx_/Med_ columns found.")
    else:
        # fetch a sample of flags (wide but capped rows)
        row_cap = st.slider("Max rows to analyze", 500, 20000, 5000, step=500)
        df_flags = fetch_cols_for_viz(flag_cols, row_cap=row_cap)

        # normalize to boolean
        def to_bool(s: pd.Series) -> pd.Series:
            s2 = s.astype(str).str.strip().str.lower()
            return s2.isin(["1","yes","true","y","t"])

        B = pd.DataFrame({c: to_bool(df_flags[c]) for c in df_flags.columns})

        # prevalence and pick top-N for readability
        prev = B.mean(numeric_only=True).sort_values(ascending=False)
        topN = st.slider("Top N flags by prevalence", 10, min(200, len(prev)), 50)
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
            min_j = st.slider("Min Jaccard to show", 0.0, 1.0, 0.1, 0.01)
            edges = [e for e in edges if e[2] >= min_j]

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
            st.caption("Node size = prevalence; edge width = co-occurrence strength (Jaccard).")
