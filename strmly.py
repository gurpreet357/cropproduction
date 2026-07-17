import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import base64
import os

# ---------------------------------------------------------
# 1. PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(
    page_title="Crop Production Analysis | Intelligence Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

PLOTLY_TEMPLATE = "plotly_dark"
COLOR_SEQ = ["#F2A93B", "#E8622C", "#E9C46A", "#C9502C", "#8A9A5B", "#B9791F"]

NUMERIC_COLS = ["N", "P", "K", "pH", "rainfall", "temperature",
                 "Area_in_hectares", "Production_in_tons", "Yield_ton_per_hec"]

REQUIRED_COLS = {"State_Name", "Crop_Type", "Crop", "N", "P", "K", "pH", "rainfall",
                  "temperature", "Area_in_hectares", "Production_in_tons", "Yield_ton_per_hec"}

# ---------------------------------------------------------
# 2. LOAD + CLEAN DATA
# ---------------------------------------------------------
@st.cache_data
def load_data(path: str = "Crop_production.csv"):
    raw = pd.read_csv(path)
    if "Unnamed: 0" in raw.columns:
        raw = raw.drop(columns=["Unnamed: 0"])

    missing = REQUIRED_COLS - set(raw.columns)
    if missing:
        raise ValueError(f"Dataset is missing required column(s): {sorted(missing)}")

    df = raw.copy()
    df["State_Name"] = df["State_Name"].astype(str).str.strip().str.lower()
    df["Crop_Type"] = df["Crop_Type"].astype(str).str.strip().str.lower()
    df["Crop"] = df["Crop"].astype(str).str.strip().str.lower()

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=NUMERIC_COLS)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Bucketed helper columns used across the Nutrient / Explorer pages
    df["rainfall_level"] = pd.qcut(df["rainfall"], 5, labels=["Very Low", "Low", "Medium", "High", "Very High"], duplicates="drop")
    df["nitrogen_level"] = pd.qcut(df["N"], 5, labels=["Very Low", "Low", "Medium", "High", "Very High"], duplicates="drop")
    df["ph_group"] = pd.cut(df["pH"], bins=[-np.inf, 6.0, 7.5, np.inf], labels=["Acidic", "Neutral", "Alkaline"])

    return df, raw


try:
    df_clean, df_original = load_data()
except FileNotFoundError:
    st.error(
        "Couldn't find **crop_production.csv**. Place it in the same folder as "
        "this script, then rerun."
    )
    st.stop()
except ValueError as e:
    st.error(f"Dataset format problem: {e}")
    st.stop()
except Exception as e:  # noqa: BLE001
    st.error(f"Couldn't load the dataset: {e}")
    st.stop()

if df_clean.empty:
    st.error("The dataset loaded but contains no usable rows after cleaning.")
    st.stop()

# ---------------------------------------------------------
# 3. BACKGROUND IMAGES — main area + sidebar, both optional
# ---------------------------------------------------------
def set_backgrounds(main_png, sidebar_png):
    css = "<style>"
    if os.path.exists(main_png):
        with open(main_png, "rb") as f:
            main_b64 = base64.b64encode(f.read()).decode()
        css += f"""
        [data-testid="stAppViewContainer"] {{
            background-image:
                linear-gradient(rgba(14,16,22,0.72), rgba(14,16,22,0.78)),
                url("data:image/png;base64,{main_b64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        """
    if os.path.exists(sidebar_png):
        with open(sidebar_png, "rb") as f:
            side_b64 = base64.b64encode(f.read()).decode()
        css += f"""
        [data-testid="stSidebar"] {{
            background-image:
                linear-gradient(rgba(10,12,16,0.65), rgba(10,12,16,0.72)),
                url("data:image/png;base64,{side_b64}");
            background-size: cover;
            background-position: center;
        }}
        """
    css += "</style>"
    st.markdown(css, unsafe_allow_html=True)


# Drop background.png / sidebar_background.png next to this script to enable —
# the app runs fine without them (falls back to the plain dark theme).
set_backgrounds("background.png", "sidebar_background.png")

# ---------------------------------------------------------
# 4. SIDEBAR — NAVIGATION + FILTERS
# ---------------------------------------------------------
st.sidebar.title("🌾 Crop Production Analysis")
st.sidebar.caption("Agricultural Intelligence System")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "🧹 Data Cleaning", "📈 Analytics", "🌱 Crop Explorer",
     "🧪 Nutrient Analysis", "💡 Insights", "⬇️ Export"],
)

st.sidebar.divider()
st.sidebar.subheader("Filters")

states_all = sorted(df_clean.State_Name.unique())
crop_types_all = sorted(df_clean.Crop_Type.unique())

state_sel = st.sidebar.multiselect(
    "State", states_all, default=states_all, format_func=lambda x: x.title()
)
crop_type_sel = st.sidebar.multiselect(
    "Crop Season", crop_types_all, default=crop_types_all, format_func=lambda x: x.title()
)

crop_pool = df_clean if not crop_type_sel else df_clean[df_clean.Crop_Type.isin(crop_type_sel)]
crops_all = sorted(crop_pool.Crop.unique())
crop_sel = st.sidebar.multiselect(
    "Crop", crops_all, default=crops_all, format_func=lambda x: x.title()
)

exclude_outliers = st.sidebar.checkbox(
    "Exclude yield outliers (> 30 t/ha)", value=True,
    help="A small number of records have implausible yields that distort averages and chart scales.",
)

df = df_clean[
    df_clean.State_Name.isin(state_sel) &
    df_clean.Crop_Type.isin(crop_type_sel) &
    df_clean.Crop.isin(crop_sel)
]
if exclude_outliers:
    df = df[df.Yield_ton_per_hec <= 30]

st.sidebar.divider()
st.sidebar.caption(f"{len(df):,} of {len(df_clean):,} records match filters")

if df.empty:
    st.warning("No records match the current filters. Try widening your selection.")
    st.stop()

# ===========================================================
# PAGE: OVERVIEW
# ===========================================================
if page == "📊 Overview":
    st.title("🌾 CROP PRODUCTION INTELLIGENCE")
    st.caption("YIELD & PRODUCTION PATTERN DETECTION SYSTEM")

    st.markdown("""Understanding what, where, and how crops are produced.

This project analyzes:

🌱 Crop & Season Distribution

🗺️ State-wise Production & Yield

🧪 Soil Nutrient Levels (N, P, K, pH)

🌦️ Climate Factors (rainfall, temperature)

📦 Area vs. Production Efficiency""")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("States Covered", df.State_Name.nunique())
    c3.metric("Crops Covered", df.Crop.nunique())
    c4.metric("Avg Yield", f"{df.Yield_ton_per_hec.mean():.2f} t/ha")
    c5.metric("% High Yield (>10 t/ha)", f"{(df.Yield_ton_per_hec > 10).mean()*100:.1f}%")

    st.divider()

    st.subheader("Top Crops by Total Production")
    top_crops = (
        df.groupby("Crop")["Production_in_tons"].sum()
        .sort_values(ascending=False).head(8).reset_index()
    )
    top_crops["Crop"] = top_crops["Crop"].str.title()
    fig = px.bar(top_crops, x="Production_in_tons", y="Crop", orientation="h",
                 template=PLOTLY_TEMPLATE, color="Crop", color_discrete_sequence=COLOR_SEQ)
    fig.update_layout(showlegend=True, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, width="stretch")

    st.subheader("Average Yield by Rainfall Level")
    by_rain = df.groupby("rainfall_level", observed=True)["Yield_ton_per_hec"].mean().reset_index()
    fig = px.line(by_rain, x="rainfall_level", y="Yield_ton_per_hec", template=PLOTLY_TEMPLATE,
                  markers=True, color_discrete_sequence=["#E8622C"])
    fig.update_layout(xaxis_title="Rainfall Level", yaxis_title="Average Yield (t/ha)")
    st.plotly_chart(fig, width="stretch")

# ===========================================================
# PAGE: DATA CLEANING
# ===========================================================
elif page == "🧹 Data Cleaning":
    st.title("Data Cleaning")

    c1, c2, c3 = st.columns(3)
    c1.metric("Original Rows", f"{len(df_original):,}")
    c2.metric("Missing Values", int(df_original.isna().sum().sum()))
    c3.metric("Duplicate Rows", int(df_original.duplicated().sum()))

    st.divider()
    st.subheader("Cleaning Pipeline")
    st.markdown("""
    - Strip whitespace and lowercase `State_Name`, `Crop_Type`, `Crop`
    - Coerce all numeric columns (`N`, `P`, `K`, `pH`, `rainfall`, `temperature`,
      `Area_in_hectares`, `Production_in_tons`, `Yield_ton_per_hec`) with `errors="coerce"`
    - Drop rows with missing values in any numeric column
    - Remove exact duplicate rows
    - Engineer `rainfall_level`, `nitrogen_level` (quintile buckets) and `ph_group`
      (Acidic / Neutral / Alkaline) from raw columns
    """)

    if st.button("▶️ Run Cleaning Summary", type="primary"):
        st.subheader("Cleaning Report")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Rows After Clean", f"{len(df_clean):,}")
        r2.metric("Rows Removed", int(len(df_original) - len(df_clean)))
        r3.metric("Numeric Columns", df_clean.select_dtypes("number").shape[1])
        r4.metric("Categorical Columns", df_clean.select_dtypes("object").shape[1])
        st.success("Cleaning summary generated.")

# ===========================================================
# PAGE: ANALYTICS
# ===========================================================
elif page == "📈 Analytics":
    st.title("Production Analytics")
    st.caption(f"Analyzing {len(df):,} records after filters")

    t1, t2, t3, t4 = st.tabs(["By Crop Season", "By State", "Rainfall vs Yield", "Correlation"])

    with t1:
        top_types = df.Crop_Type.value_counts().head(6).index
        subset = df[df.Crop_Type.isin(top_types)].copy()
        subset["Crop_Type"] = subset["Crop_Type"].str.title()
        fig = px.box(subset, x="Crop_Type", y="Yield_ton_per_hec", color="Crop_Type",
                     template=PLOTLY_TEMPLATE, color_discrete_sequence=COLOR_SEQ)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")

    with t2:
        counts = (
            df.groupby("State_Name")["Production_in_tons"].sum()
            .sort_values(ascending=False).head(8).reset_index()
        )
        counts["State_Name"] = counts["State_Name"].str.title()
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(counts, names="State_Name", values="Production_in_tons", hole=0.55,
                         color_discrete_sequence=COLOR_SEQ, template=PLOTLY_TEMPLATE)
            st.plotly_chart(fig, width="stretch")
        with col2:
            fig = px.bar(counts, x="State_Name", y="Production_in_tons", color="State_Name",
                         color_discrete_sequence=COLOR_SEQ, template=PLOTLY_TEMPLATE)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width="stretch")

    with t3:
        fig = px.box(df, x="rainfall_level", y="Yield_ton_per_hec", color="rainfall_level",
                     category_orders={"rainfall_level": ["Very Low", "Low", "Medium", "High", "Very High"]},
                     color_discrete_sequence=COLOR_SEQ, template=PLOTLY_TEMPLATE)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")

    with t4:
        corr = df[NUMERIC_COLS].corr()
        fig = px.imshow(corr, text_auto=".2f", template=PLOTLY_TEMPLATE,
                         color_continuous_scale=["#1A1A24", "#F2A93B", "#E8622C"], zmin=-1, zmax=1)
        st.plotly_chart(fig, width="stretch")

# ===========================================================
# PAGE: CROP EXPLORER
# ===========================================================
elif page == "🌱 Crop Explorer":
    st.title("Crop Explorer")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Yield by Crop Season")
        top_types = df.Crop_Type.value_counts().head(5).index
        subset = df[df.Crop_Type.isin(top_types)].copy()
        subset["Crop_Type"] = subset["Crop_Type"].str.title()
        fig = px.violin(subset, x="Crop_Type", y="Yield_ton_per_hec", color="Crop_Type", box=True,
                         template=PLOTLY_TEMPLATE, color_discrete_sequence=COLOR_SEQ)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.subheader("Yield by Soil pH Group")
        fig = px.box(df, x="ph_group", y="Yield_ton_per_hec", color="ph_group",
                     category_orders={"ph_group": ["Acidic", "Neutral", "Alkaline"]},
                     template=PLOTLY_TEMPLATE, color_discrete_sequence=COLOR_SEQ)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("Highest-Yield Crop Records")
    top_yield = df.nlargest(10, "Yield_ton_per_hec")[
        ["State_Name", "Crop", "Crop_Type", "N", "P", "K", "pH",
         "rainfall", "temperature", "Yield_ton_per_hec"]
    ].copy()
    top_yield["State_Name"] = top_yield["State_Name"].str.title()
    top_yield["Crop"] = top_yield["Crop"].str.title()
    top_yield["Crop_Type"] = top_yield["Crop_Type"].str.title()
    st.dataframe(top_yield, width="stretch")

# ===========================================================
# PAGE: NUTRIENT ANALYSIS
# ===========================================================
elif page == "🧪 Nutrient Analysis":
    st.title("Nutrient & Climate Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Avg Yield by Nitrogen Level")
        by_n = df.groupby("nitrogen_level", observed=True)["Yield_ton_per_hec"].mean().reset_index()
        fig = px.bar(by_n, x="nitrogen_level", y="Yield_ton_per_hec", template=PLOTLY_TEMPLATE,
                     color="Yield_ton_per_hec", color_continuous_scale=["#1A1A24", "#F2A93B", "#E8622C"])
        fig.update_layout(height=380, coloraxis_showscale=False,
                           xaxis_title="Nitrogen Level", yaxis_title="Average Yield (t/ha)")
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.subheader("Avg Yield by Rainfall Level")
        by_rain2 = df.groupby("rainfall_level", observed=True)["Yield_ton_per_hec"].mean().reset_index()
        fig = px.line(by_rain2, x="rainfall_level", y="Yield_ton_per_hec", template=PLOTLY_TEMPLATE,
                      markers=True, color_discrete_sequence=["#F2A93B"])
        fig.update_traces(line=dict(width=3), marker=dict(size=7))
        fig.update_layout(height=380, xaxis_title="Rainfall Level", yaxis_title="Average Yield (t/ha)")
        st.plotly_chart(fig, width="stretch")

# ===========================================================
# PAGE: INSIGHTS
# ===========================================================
elif page == "💡 Insights":
    st.title("Analytical Insights")

    if st.button("⚡ Generate Insights", type="primary"):
        top_crop = df.Crop.value_counts().idxmax()
        best_state = df.groupby("State_Name")["Yield_ton_per_hec"].mean().idxmax()
        high_yield_pct = (df.Yield_ton_per_hec > 10).mean() * 100
        corr_val = df["rainfall"].corr(df["Yield_ton_per_hec"])
        direction = "negatively" if corr_val < 0 else "positively"
        median_rain = df["rainfall"].median()
        low_rain_yield = df[df.rainfall <= median_rain]["Yield_ton_per_hec"].mean()
        high_rain_yield = df[df.rainfall > median_rain]["Yield_ton_per_hec"].mean()
        top_season = df.Crop_Type.value_counts().idxmax()
        top_season_pct = (df.Crop_Type == top_season).mean() * 100

        st.subheader("Key Findings")
        st.info(f"🌾 **{top_crop.title()}** is the most frequently recorded crop — "
                f"{(df.Crop == top_crop).mean()*100:.1f}% of records in this selection.")
        st.warning(f"🗺️ **{best_state.title()}** shows the highest average yield "
                   f"({df[df.State_Name==best_state]['Yield_ton_per_hec'].mean():.2f} t/ha) in this filtered set.")
        st.error(f"📈 **{high_yield_pct:.1f}%** of records exceed 10 t/ha yield.")
        st.info(f"🌧️ Rainfall is {direction} correlated with yield (r = {corr_val:.2f}).")
        st.warning(f"☔ Below-median rainfall records average **{low_rain_yield:.2f} t/ha**, "
                   f"vs **{high_rain_yield:.2f} t/ha** above median rainfall.")
        st.error(f"🌱 **{top_season.title()}** is the dominant crop season — "
                 f"**{top_season_pct:.1f}%** of records.")
    else:
        st.info("Click **Generate Insights** to compute key findings from the current filtered data.")

# ===========================================================
# PAGE: EXPORT
# ===========================================================
elif page == "⬇️ Export":
    st.title("Export Data")
    st.write(f"Exporting **{len(df):,}** of **{len(df_clean):,}** total records based on active filters.")
    table_height = min(35 * len(df) + 38, 800)
    st.dataframe(df, width="stretch", height=table_height)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Filtered CSV", data=csv,
                        file_name="crop_production_filtered.csv", mime="text/csv")
