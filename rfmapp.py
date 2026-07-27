import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
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


# --- DYNAMIC RFM & K-MEANS PIPELINE ---
def process_dynamic_rfm(
    df, id_col, date_col, order_col, price_col, qty_col, n_clusters=4
):
    """Aggregates RFM metrics, log-transforms, scales, and fits K-Means dynamically."""
    # 1. Column Selection & Validation Check
    mapped_cols = [id_col, date_col, order_col, price_col, qty_col]

    # Guardrail: Ensure unique column mappings
    if len(set(mapped_cols)) < 5:
        st.error(
            "⚠️ **Mapping Error:** Please select 5 unique columns. Two or more dropdowns are set to the same header."
        )
        return None

    clean_df = df[mapped_cols].copy()
    clean_df.columns = [
        "CustomerID",
        "OrderDate",
        "OrderID",
        "UnitPrice",
        "Quantity",
    ]

    # Guardrail: Check for missing values
    missing_count = clean_df.isna().sum().sum()
    if missing_count > 0:
        st.error(
            f"❌ **Data Ingestion Error:** Found {missing_count} missing (NaN) values in mapped columns. "
            "Please upload a dataset with zero missing values or clean your file before uploading."
        )
        return None

    # Parse dates safely
    clean_df["OrderDate"] = pd.to_datetime(clean_df["OrderDate"], errors="coerce")
    if clean_df["OrderDate"].isna().sum() > 0:
        st.error(
            "❌ **Date Formatting Error:** Some values in your Order Date column could not be parsed as dates."
        )
        return None

    # Calculate Total Sales Amount
    clean_df["SalesAmount"] = clean_df["UnitPrice"] * clean_df["Quantity"]

    # 2. Calculate RFM Metrics
    reference_date = clean_df["OrderDate"].max() + pd.Timedelta(days=1)

    rfm = (
        clean_df.groupby("CustomerID")
        .agg(
            {
                "OrderDate": lambda x: (reference_date - x.max()).days,  # Recency
                "OrderID": "nunique",  # Frequency
                "SalesAmount": "sum",  # Monetary
            }
        )
        .reset_index()
    )

    rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]

    if rfm.empty:
        st.error("❌ Aggregation failed: No valid records found.")
        return None

    # 3. Feature Transformation (Log1p -> Dynamic Standard Scaling)
    X = rfm[["Recency", "Frequency", "Monetary"]]

    # Log Transformation to handle skewness
    X_log = np.log1p(X)

    # Dynamic Standard Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)

    # 4. Dynamic K-Means Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm["Cluster"] = kmeans.fit_predict(X_scaled)

    # 5. Relative Persona Mapping (Ranks clusters by average Monetary spend)
    monetary_rank = (
        rfm.groupby("Cluster")["Monetary"].mean().sort_values().index.tolist()
    )

    persona_map = {
        monetary_rank[0]: "Lost / Very Lapsed Customers",
        monetary_rank[1]: "Hibernating / Slipping Customers",
        monetary_rank[2]: "Warm Leads / Potential Churners",
        monetary_rank[3]: "Champions / High-Value Loyal Customers",
    }

    rfm["Customer Segment"] = rfm["Cluster"].map(persona_map)

    return rfm


# --- FRONTEND UI ---
st.title("Universal RFM Customer Segmentation Engine")
st.markdown(
    "A dynamic machine learning dashboard that fits a K-Means model on the fly for any uploaded e-commerce dataset."
)

st.divider()

# --- USER GUIDELINES CARD ---
with st.expander("📋 **Dataset Requirements & Preparation Instructions (Click to Expand)**", expanded=True):
    st.markdown("""
    To ensure successful analysis, please ensure your uploaded CSV is **pre-cleaned** and structured according to the following guidelines:
    
    * **No Missing Values:** Ensure there are no empty (`NaN` / `null`) cells in your key transaction columns.
    * **Positive Numeric Values:** `UnitPrice` and `Quantity` must be numeric, positive numbers ($> 0$). Remove currency symbols (e.g., `$`) or commas prior to upload.
    * **Valid Dates:** Dates should be formatted consistently (e.g., `YYYY-MM-DD` or `YYYY-MM-DD HH:MM:SS`).
    * **No Cancellations or Returns:** Exclude negative quantity transactions, returned item logs, or refund invoice numbers beforehand.
    * **Required Information:** Your file must contain columns representing: **Customer ID**, **Order Date**, **Order ID / Invoice Number**, **Unit Price**, and **Quantity**.
    """)

st.divider()

# Sidebar Setup
st.sidebar.header("📁 Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Clean Transaction CSV", type=["csv"])

st.sidebar.markdown("**OR**")
if st.sidebar.button("🎲 Load Sample Demo Data"):
    st.session_state["raw_df"] = generate_synthetic_data()
    st.session_state.pop("rfm_results", None)
    st.sidebar.success("Loaded 1,000 clean synthetic transactions!")

if uploaded_file is not None:
    st.session_state["raw_df"] = pd.read_csv(uploaded_file)

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

    # Data Preview & Quality Summary
    st.divider()
    st.subheader("🔍 Dataset Overview & Health Check")

    tab1, tab2 = st.tabs(["Raw Data Preview", "Data Summary"])

    with tab1:
        st.dataframe(raw_df.head(10), use_container_width=True)

    with tab2:
        m1, m2, m3 = st.columns(3)
        total_rows = len(raw_df)
        total_missing = raw_df[[id_col, date_col, order_col, price_col, qty_col]].isna().sum().sum()

        m1.metric("Total Rows", f"{total_rows:,}")
        m2.metric("Total Columns", f"{len(columns):,}")
        m3.metric("Missing Cells (Mapped Fields)", f"{total_missing:,}")

        mapped_fields = {
            "Customer ID": id_col,
            "Order Date": date_col,
            "Order ID": order_col,
            "Unit Price": price_col,
            "Quantity": qty_col,
        }

        health_data = []
        for field_label, col_name in mapped_fields.items():
            missing_count = raw_df[col_name].isna().sum()
            health_data.append({
                "Mapped Field": field_label,
                "CSV Column Name": col_name,
                "Data Type": str(raw_df[col_name].dtype),
                "Missing Values": missing_count,
            })

        st.dataframe(pd.DataFrame(health_data), use_container_width=True)

    st.divider()

    if st.button("🚀 Process & Segment Data", type="primary"):
        with st.spinner("Training dynamic K-Means model & generating clusters..."):
            rfm_results = process_dynamic_rfm(
                raw_df, id_col, date_col, order_col, price_col, qty_col
            )
            st.session_state["rfm_results"] = rfm_results

# Display Results & Dashboards
if "rfm_results" in st.session_state and st.session_state["rfm_results"] is not None:
    rfm_df = st.session_state["rfm_results"]

    if not rfm_df.empty:
        st.divider()
        st.subheader("📌 Key Business Insights")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Customers", f"{len(rfm_df):,}")
        c2.metric("Avg Recency (Days)", f"{rfm_df['Recency'].mean():.1f}")
        c3.metric("Avg Frequency (Orders)", f"{rfm_df['Frequency'].mean():.1f}")
        c4.metric("Avg Spend ($)", f"${rfm_df['Monetary'].mean():,.2f}")

        st.divider()

        # Visualizations
        st.subheader("Customer Segment Distribution")
        fig_pie = px.pie(
            rfm_df,
            names="Customer Segment",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        st.divider()

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

        # Marketing Strategies Section
        st.divider()
        st.subheader("Recommended Marketing Strategies")

        strat_col1, strat_col2 = st.columns(2)

        with strat_col1:
            with st.expander("🏆 **Champions / High-Value Loyal Customers**"):
                st.markdown("""
                * **Reward Loyalty:** Offer exclusive discounts, early access to new products, or VIP loyalty programs.
                * **Encourage Referrals:** Ask them to refer friends and family with referral bonuses.
                * **Personalized Communication:** Send highly relevant product recommendations and updates.
                """)

            with st.expander("⏳ **Hibernating / Slipping Customers**"):
                st.markdown("""
                * **Win-Back Campaigns:** Offer incentives to reactivate, such as significant discounts or free shipping.
                * **Personalized Outreach:** Reach out with personalized messages based on their past purchase history.
                * **Customer Surveys:** Understand reasons for reduced activity to tailor future strategies.
                """)

        with strat_col2:
            with st.expander("🔥 **Warm Leads / Potential Churners**"):
                st.markdown("""
                * **Re-engagement Campaigns:** Send targeted emails with special offers or new product announcements to encourage repeat purchases.
                * **Value Proposition Reminders:** Highlight benefits of past purchases or new features.
                * **Feedback Collection:** Ask for feedback to understand potential barriers to purchase.
                """)

            with st.expander("💤 **Lost / Very Lapsed Customers**"):
                st.markdown("""
                * **Last-Ditch Offers:** Present aggressive, limited-time offers to entice a return (e.g., steep discounts, significant freebies).
                * **Focus on Brand Awareness:** If win-back fails, focus on subtle brand reminders without high investment.
                * **Consider Re-acquisition:** For some, treat as new customers if they haven't responded to win-back efforts.
                """)

        # Data Export Table
        st.divider()
        st.subheader("Segmented Customer Results")
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
