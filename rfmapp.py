import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import streamlit as st

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Universal RFM Customer Segmentation Engine",
    layout="wide",
)


# --- HELPER FUNCTION FOR SYNTHETIC DEMO DATA GENERATION ---
def generate_synthetic_data(num_rows=1000, num_customers=150):
    """Generates a clean preprocessed transaction dataset for demonstration."""
    np.random.seed(42)

    customer_ids = [f"CUST-{1000 + i}" for i in range(num_customers)]
    order_ids = [f"INV-{50000 + i}" for i in range(300)]
    dates = pd.date_range(end="2026-06-30", periods=365, freq="D")

    data = {
        "CustomerID": np.random.choice(customer_ids, num_rows),
        "OrderDate": np.random.choice(dates, num_rows),
        "OrderID": np.random.choice(order_ids, num_rows),
        "UnitPrice": np.round(
            np.random.exponential(scale=25.0, size=num_rows) + 1.5, 2
        ),
        "Quantity": np.random.randint(1, 12, size=num_rows),
    }

    return pd.DataFrame(data)


# --- HELPER FUNCTION FOR SMART COLUMN MATCHING ---
def find_default(candidates, cols):
    """Finds the index of the first matching column name based on keyword candidates."""
    for c in cols:
        if any(cand in c.lower() for cand in candidates):
            return cols.index(c)
    return 0


# --- MODEL OPTIMIZATION CALCULATIONS (ELBOW & SILHOUETTE) ---
def calculate_optimization_metrics(X_scaled, max_k=8):
    """Computes WCSS (Inertia) and Silhouette Scores for K=2 to max_k."""
    wcss = []
    silhouette_scores = []
    k_range = list(range(2, max_k + 1))

    # K=1 WCSS calculation
    km1 = KMeans(n_clusters=1, random_state=42, n_init=10)
    km1.fit(X_scaled)
    wcss_all = [km1.inertia_]

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        wcss_all.append(km.inertia_)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)

    df_elbow = pd.DataFrame(
        {"Number of Clusters (K)": list(range(1, max_k + 1)), "WCSS": wcss_all}
    )

    df_silhouette = pd.DataFrame(
        {
            "Number of Clusters (K)": k_range,
            "Silhouette Score": silhouette_scores,
        }
    )

    return df_elbow, df_silhouette


# --- DYNAMIC RFM & K-MEANS PIPELINE ---
def process_dynamic_rfm(
    df, id_col, date_col, order_col, price_col, qty_col, n_clusters=4
):
    """Aggregates RFM metrics, log-transforms, scales, and fits K-Means dynamically."""
    mapped_cols = [id_col, date_col, order_col, price_col, qty_col]

    if len(set(mapped_cols)) < 5:
        st.error(
            "⚠️ **Mapping Error:** Please select 5 unique columns. Two or more dropdowns are set to the same header."
        )
        return None, None, None

    clean_df = df[mapped_cols].copy()
    clean_df.columns = [
        "CustomerID",
        "OrderDate",
        "OrderID",
        "UnitPrice",
        "Quantity",
    ]

    missing_count = clean_df.isna().sum().sum()
    if missing_count > 0:
        st.error(
            f"❌ **Data Ingestion Error:** Found {missing_count} missing (NaN) values in mapped columns."
        )
        return None, None, None

    clean_df["OrderDate"] = pd.to_datetime(clean_df["OrderDate"], errors="coerce")
    if clean_df["OrderDate"].isna().sum() > 0:
        st.error(
            "❌ **Date Formatting Error:** Some values in your Order Date column could not be parsed as dates."
        )
        return None, None, None

    clean_df["SalesAmount"] = clean_df["UnitPrice"] * clean_df["Quantity"]
    reference_date = clean_df["OrderDate"].max() + pd.Timedelta(days=1)

    rfm = (
        clean_df.groupby("CustomerID")
        .agg(
            {
                "OrderDate": lambda x: (reference_date - x.max()).days,
                "OrderID": "nunique",
                "SalesAmount": "sum",
            }
        )
        .reset_index()
    )

    rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]

    if rfm.empty:
        st.error("❌ Aggregation failed: No valid records found.")
        return None, None, None

    # Feature Transformation
    X = rfm[["Recency", "Frequency", "Monetary"]]
    X_log = np.log1p(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)

    # Optimization calculations (Elbow & Silhouette)
    df_elbow, df_silhouette = calculate_optimization_metrics(X_scaled, max_k=8)

    # Dynamic K-Means Fitting
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm["Cluster"] = kmeans.fit_predict(X_scaled)

    # Flexible Persona Mapping based on monetary rank
    monetary_rank = (
        rfm.groupby("Cluster")["Monetary"].mean().sort_values().index.tolist()
    )

    if n_clusters == 4:
        persona_names = [
            "Lost / Very Lapsed Customers",
            "Hibernating / Slipping Customers",
            "Warm Leads / Potential Churners",
            "Champions / High-Value Loyal Customers",
        ]
    elif n_clusters == 3:
        persona_names = [
            "Lost / Low-Value Customers",
            "Mid-Tier / Warm Leads",
            "Champions / High-Value Loyal Customers",
        ]
    elif n_clusters == 2:
        persona_names = ["Low-Value / Inactive", "High-Value / Active"]
    else:
        persona_names = [
            f"Segment Tier {i+1} (Avg Spend: Low-to-High)"
            for i in range(n_clusters)
        ]

    persona_map = {
        monetary_rank[i]: persona_names[i] for i in range(n_clusters)
    }
    rfm["Customer Segment"] = rfm["Cluster"].map(persona_map)

    return rfm, df_elbow, df_silhouette


# --- FRONTEND UI ---
st.title("Universal RFM Customer Segmentation Engine")
st.markdown(
    "A dynamic machine learning dashboard that fits a K-Means model on the fly for any uploaded e-commerce dataset."
)

st.divider()

# --- USER GUIDELINES CARD ---
with st.expander(
    "📋 **Dataset Requirements & Preparation Instructions (Click to Expand)**",
    expanded=False,
):
    st.markdown("""
    To ensure successful analysis, please ensure your uploaded CSV is **pre-cleaned** and structured according to the following guidelines:
    
    * **No Missing Values:** Ensure there are no empty (`NaN` / `null`) cells in your key transaction columns.
    * **Positive Numeric Values:** `UnitPrice` and `Quantity` must be numeric, positive numbers ($> 0$). Remove currency symbols or commas prior to upload.
    * **Valid Dates:** Dates should be formatted consistently (e.g., `YYYY-MM-DD`).
    * **No Cancellations or Returns:** Exclude negative quantity transactions or refund invoice numbers beforehand.
    """)

st.divider()

# Sidebar Setup
st.sidebar.header("📁 Data Input & Model Parameters")
uploaded_file = st.sidebar.file_uploader(
    "Upload Clean Transaction CSV", type=["csv"]
)

st.sidebar.markdown("**OR**")
if st.sidebar.button("🎲 Load Sample Demo Data"):
    st.session_state["raw_df"] = generate_synthetic_data()
    st.session_state.pop("rfm_results", None)
    st.sidebar.success("Loaded 1,000 clean synthetic transactions!")

if uploaded_file is not None:
    st.session_state["raw_df"] = pd.read_csv(uploaded_file)

# Dynamic K-Slider
st.sidebar.divider()
st.sidebar.subheader("⚙️ Model Settings")
n_clusters = st.sidebar.slider(
    "Select Number of Clusters (K)",
    min_value=2,
    max_value=6,
    value=4,
    help="Adjust K to segment customers into more or fewer behavioral tiers.",
)

# Main App Execution
if "raw_df" in st.session_state:
    raw_df = st.session_state["raw_df"]
    columns = list(raw_df.columns)

    # Column Mapping Section
    st.subheader("Map Your Data Columns")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        id_col = st.selectbox(
            "Customer ID",
            columns,
            index=find_default(["cust", "user", "client"], columns),
        )
    with col2:
        date_col = st.selectbox(
            "Order Date",
            columns,
            index=find_default(["date", "time", "day"], columns),
        )
    with col3:
        order_col = st.selectbox(
            "Order ID",
            columns,
            index=find_default(["order", "txn", "invoice"], columns),
        )
    with col4:
        price_col = st.selectbox(
            "Unit Price",
            columns,
            index=find_default(["price", "rate", "cost"], columns),
        )
    with col5:
        qty_col = st.selectbox(
            "Quantity",
            columns,
            index=find_default(["qty", "quantity", "count"], columns),
        )

    st.divider()

    if st.button("🚀 Process & Segment Data", type="primary"):
        with st.spinner("Calculating metrics, evaluating K, and fitting model..."):
            rfm_results, df_elbow, df_silhouette = process_dynamic_rfm(
                raw_df,
                id_col,
                date_col,
                order_col,
                price_col,
                qty_col,
                n_clusters=n_clusters,
            )
            st.session_state["rfm_results"] = rfm_results
            st.session_state["df_elbow"] = df_elbow
            st.session_state["df_silhouette"] = df_silhouette

# Display Results & Analytics Dashboards
if "rfm_results" in st.session_state and st.session_state["rfm_results"] is not None:
    rfm_df = st.session_state["rfm_results"]
    df_elbow = st.session_state["df_elbow"]
    df_silhouette = st.session_state["df_silhouette"]

    if not rfm_df.empty:
        st.divider()

        # Section Tabs
        main_tab, opt_tab = st.tabs(
            ["📊 Segmentation Dashboards", "📐 Cluster Optimization (Elbow & Silhouette)"]
        )

        # TAB 1: SEGMENTATION RESULTS
        with main_tab:
            st.subheader("📌 Key Business Insights")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Customers", f"{len(rfm_df):,}")
            c2.metric("Avg Recency (Days)", f"{rfm_df['Recency'].mean():.1f}")
            c3.metric(
                "Avg Frequency (Orders)", f"{rfm_df['Frequency'].mean():.1f}"
            )
            c4.metric("Avg Spend ($)", f"${rfm_df['Monetary'].mean():,.2f}")

            st.divider()

            col_pie, col_scatter = st.columns(2)

            with col_pie:
                st.subheader("Customer Segment Distribution")
                fig_pie = px.pie(
                    rfm_df,
                    names="Customer Segment",
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_scatter:
                st.subheader("Frequency vs Spend (Log Scale)")
                fig_scatter = px.scatter(
                    rfm_df,
                    x="Frequency",
                    y="Monetary",
                    color="Customer Segment",
                    size="Recency",
                    hover_data=["CustomerID"],
                    log_x=True,
                    log_y=True,
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

            st.divider()
            st.subheader("Segmented Customer Table")
            st.dataframe(
                rfm_df[
                    [
                        "CustomerID",
                        "Customer Segment",
                        "Recency",
                        "Frequency",
                        "Monetary",
                    ]
                ],
                use_container_width=True,
            )

            csv_data = rfm_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Segmented Results (CSV)",
                data=csv_data,
                file_name="rfm_segmented_customers.csv",
                mime="text/csv",
            )

        # TAB 2: CLUSTER OPTIMIZATION (ELBOW + SILHOUETTE)
        with opt_tab:
            st.subheader("Determine Optimal Number of Clusters")
            st.markdown(
                "Use these diagnostic charts to select the mathematically optimal value for $K$."
            )

            opt_col1, opt_col2 = st.columns(2)

            with opt_col1:
                st.markdown("### 1. Elbow Method (WCSS)")
                fig_elbow = px.line(
                    df_elbow,
                    x="Number of Clusters (K)",
                    y="WCSS",
                    markers=True,
                    title="Elbow Curve (Within-Cluster Sum of Squares)",
                )
                fig_elbow.add_vline(
                    x=n_clusters,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Selected K={n_clusters}",
                )
                st.plotly_chart(fig_elbow, use_container_width=True)
                st.info(
                    "💡 **How to read:** Look for the 'bend' or elbow point where WCSS drops off significantly less."
                )

            with opt_col2:
                st.markdown("### 2. Silhouette Analysis")
                fig_sil = px.bar(
                    df_silhouette,
                    x="Number of Clusters (K)",
                    y="Silhouette Score",
                    title="Average Silhouette Score by K",
                    color="Silhouette Score",
                    color_continuous_scale="Blues",
                )
                fig_sil.add_vline(
                    x=n_clusters,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Selected K={n_clusters}",
                )
                st.plotly_chart(fig_sil, use_container_width=True)

                best_k = df_silhouette.loc[
                    df_silhouette["Silhouette Score"].idxmax(),
                    "Number of Clusters (K)",
                ]
                best_score = df_silhouette["Silhouette Score"].max()

                st.success(
                    f"🏆 **Highest Silhouette Score:** $K = {int(best_k)}$ (Score: {best_score:.3f}). "
                    f"This indicates the cleanest cluster separation for this dataset."
                )

else:
    if "raw_df" not in st.session_state:
        st.info(
            "👈 Upload a transaction CSV file or click 'Load Sample Demo Data' in the sidebar to get started!"
        )

# Footer
st.markdown(
    """
    <div style='text-align: center; color: #888888; padding-top: 20px;'>
        © 2026 • Built with Streamlit & Python by Ranithri Hewasiliyan • All Rights Reserved
    </div>
    """,
    unsafe_allow_html=True,
)
