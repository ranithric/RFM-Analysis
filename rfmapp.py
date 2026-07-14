import warnings
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import sklearn  # Pre-loading sklearn prevents joblib load freezes
import streamlit as st

# Suppress version mismatch warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="RFM Customer Segmentation Engine",
    layout="wide",
)

# --- LOAD TRAINED ARTIFACTS ---


@st.cache_resource
def load_models():
    scaler = joblib.load("scaler.pkl")
    kmeans = joblib.load("kmeans_model.pkl")
    return scaler, kmeans


scaler, kmeans = load_models()


# --- HELPER FUNCTION FOR SYNTHETIC DATA GENERATION ---
def generate_synthetic_data(num_rows=1000, num_customers=150):
    """Generates realistic e-commerce transaction data for demonstration."""
    np.random.seed(42)

    customer_ids = [f"CUST-{1000 + i}" for i in range(num_customers)]
    order_ids = [f"INV-{50000 + i}" for i in range(300)]

    dates = pd.date_range(end="2026-06-30", periods=365, freq="D")

    data = {
        "InvoiceNo": np.random.choice(order_ids, num_rows),
        "CustomerID": np.random.choice(customer_ids, num_rows),
        "InvoiceDate": np.random.choice(dates, num_rows),
        "UnitPrice": np.round(np.random.exponential(scale=25.0, size=num_rows) + 1.5, 2),
        "Quantity": np.random.randint(1, 12, size=num_rows),
    }

    df = pd.DataFrame(data)
    # Inject a few cancellation invoices for realism
    cancel_indices = np.random.choice(df.index, size=int(num_rows * 0.05), replace=False)
    df.loc[cancel_indices, "InvoiceNo"] = "C" + df.loc[cancel_indices, "InvoiceNo"].astype(str)

    return df


# --- HELPER FUNCTION FOR SMART COLUMN MATCHING ---
def find_default(candidates, cols):
    """Finds the index of the first matching column name based on keyword candidates."""
    for c in cols:
        if any(cand in c.lower() for cand in candidates):
            return cols.index(c)
    return 0


# --- BACKEND PROCESSING & PREPROCESSING FUNCTION ---
def process_mapped_sales_data(
    df, id_col, date_col, order_col, price_col, qty_col, scaler_obj, kmeans_model_obj
):
    # 1. Select and Rename Columns
    clean_df = df[[id_col, date_col, order_col, price_col, qty_col]].copy()
    clean_df.columns = [
        "CustomerID",
        "OrderDate",
        "OrderID",
        "UnitPrice",
        "Quantity",
    ]

    # 2. Data Preprocessing Pipeline (Notebook Alignment)
    # A. Remove returned products (Invoice numbers starting with / containing 'C')
    clean_df["OrderID"] = clean_df["OrderID"].astype(str).str.strip()
    clean_df = clean_df[~clean_df["OrderID"].str.contains("C", case=False, na=False)]

    # B. Remove missing values from the dataset
    clean_df = clean_df.dropna()

    # C. Clean UnitPrice & Quantity fields
    if clean_df["UnitPrice"].dtype == "object":
        clean_df["UnitPrice"] = (
            clean_df["UnitPrice"]
            .astype(str)
            .str.replace(r"[^\d.-]", "", regex=True)
        )
    clean_df["UnitPrice"] = pd.to_numeric(clean_df["UnitPrice"], errors="coerce")

    if clean_df["Quantity"].dtype == "object":
        clean_df["Quantity"] = (
            clean_df["Quantity"]
            .astype(str)
            .str.replace(r"[^\d.-]", "", regex=True)
        )
    clean_df["Quantity"] = pd.to_numeric(clean_df["Quantity"], errors="coerce")

    # D. Drop any NaNs resulting from numeric coercion & filter out negative/zero values
    clean_df = clean_df.dropna()
    clean_df["CustomerID"] = clean_df["CustomerID"].astype(str).str.strip()
    clean_df = clean_df[(clean_df["Quantity"] > 0) & (clean_df["UnitPrice"] > 0)]

    # E. Parse Dates Safely
    clean_df["OrderDate"] = pd.to_datetime(clean_df["OrderDate"], errors="coerce")
    clean_df = clean_df.dropna(subset=["OrderDate"])

    # Verify rows remain after cleaning
    if clean_df.empty:
        st.error(
            "❌ Cleaning failed: No valid data remaining after filtering cancellations, invalid dates, prices, or quantities."
        )
        return None

    # F. Compute Total Sales Amount
    clean_df["SalesAmount"] = clean_df["UnitPrice"] * clean_df["Quantity"]

    # 3. Calculate RFM Metrics
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
        st.error("❌ Aggregation failed: No customer records produced.")
        return None

    # 4. Transform & Predict Clusters (Matches notebook sequence)
    X = rfm[["Recency", "Frequency", "Monetary"]]

    # Step 1: Scale features using loaded scaler
    rfm_scaled_array = scaler_obj.transform(X)
    rfm_normalized = pd.DataFrame(rfm_scaled_array, columns=X.columns, index=X.index)

    # Step 2: Log transform scaled values (log1p)
    rfm_log_transformed = np.log1p(rfm_normalized)

    # Step 3: Predict clusters using KMeans model
    rfm["Cluster"] = kmeans_model_obj.predict(rfm_log_transformed)

    # 4-Cluster Persona Definitions
    cluster_labels = {
        0: "Warm Leads / Potential Churners",
        1: "Lost / Very Lapsed Customers",
        2: "Champions / High-Value Loyal Customers",
        3: "Hibernating / Slipping Customers",
    }

    # Map cluster numbers into human-readable persona names
    rfm["Customer Segment"] = rfm["Cluster"].map(cluster_labels).fillna(
        rfm["Cluster"].apply(lambda x: f"Cluster {x}")
    )

    return rfm


# --- FRONTEND UI ---
st.title("🚀 RFM Customer Segmentation Dashboard")
st.markdown(
    "Upload raw transaction data to automatically clean, aggregate, and group customers into ML segments."
)

st.divider()

# Sidebar Setup
st.sidebar.header("📁 Data Input")
uploaded_file = st.sidebar.file_uploader(
    "Upload Transaction CSV", type=["csv"]
)

# Demo / Synthetic Data Option
st.sidebar.markdown("**OR**")
if st.sidebar.button("🎲 Load Sample Demo Data"):
    st.session_state["raw_df"] = generate_synthetic_data()
    st.session_state.pop("rfm_results", None)  # Reset previous results
    st.sidebar.success("Loaded 1,000 synthetic transactions!")

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

    # Data Preview & Health Check Section
    st.divider()
    st.subheader("🔍 Dataset Overview & Health Check")

    tab1, tab2 = st.tabs(["Raw Data Preview", "Data Quality & Types Summary"])

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
        target_types = {
            "Customer ID": "String",
            "Order Date": "Datetime",
            "Order ID": "String",
            "Unit Price": "Float / Numeric",
            "Quantity": "Integer / Numeric",
        }

        for field_label, col_name in mapped_fields.items():
            missing_count = raw_df[col_name].isna().sum()
            missing_pct = (missing_count / total_rows) * 100
            curr_type = str(raw_df[col_name].dtype)

            health_data.append({
                "Mapped Field": field_label,
                "CSV Column Name": col_name,
                "Original Data Type": curr_type,
                "Target/Corrected Type": target_types[field_label],
                "Missing Values": missing_count,
                "Missing %": f"{missing_pct:.1f}%",
            })

        st.dataframe(pd.DataFrame(health_data), use_container_width=True)

    st.divider()

    if st.button("🚀 Process & Segment Data", type="primary"):
        with st.spinner("Cleaning formats & calculating clusters..."):
            rfm_results = process_mapped_sales_data(
                raw_df, id_col, date_col, order_col, price_col, qty_col, scaler, kmeans
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