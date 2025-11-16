import streamlit as st
import pandas as pd
import lifetimes
import numpy as np
import altair as alt
from sklearn.cluster import KMeans
from lifetimes import ParetoNBDFitter, GammaGammaFitter
from lifetimes.plotting import plot_frequency_recency_matrix, plot_probability_alive_matrix
import warnings
import matplotlib.pyplot as plt
from io import BytesIO

# --- App Configuration ---
st.set_page_config(page_title="Customer Lifetime Value Predictor", layout="wide")

# --- UI Elements ---
def show_sidebar():
    """Configure and display the sidebar elements"""
    st.sidebar.image(
        "http://logok.org/wp-content/uploads/2014/06/City-of-Melbourne-logo-M.png", 
        width=120
    )
    st.sidebar.markdown("**Made by Akshit Mahajan**")
    st.sidebar.title("Input Features ")
    
    # Input sliders
    days = st.sidebar.slider(
        "Select The No. Of Days", 
        min_value=1, 
        max_value=365, 
        value=30
    )
    profit = st.sidebar.slider(
        "Profit Margin (%)", 
        min_value=1.0, 
        max_value=9.0, 
        value=5.0
    ) / 100  # Convert to decimal
    
    # Display selected features
    st.sidebar.subheader("Selected Input Features ")
    st.sidebar.write(pd.DataFrame({"Days": [days], "Profit": [f"{profit:.1%}"]}))
    
    st.sidebar.markdown("""
    **Instructions:**
    - Select input parameters first
    - Upload a CSV file with RFM data
    - Only CSV files are supported
    """)
    
    return days, profit

# --- Visualization Functions ---
def plot_customer_distributions(result):
    """Plot various customer distribution charts"""
    col1, col2 = st.columns(2)
    
    with col1:
        # Monetary Value Distribution
        st.subheader("Monetary Value Distribution")
        monetary_chart = alt.Chart(result).mark_bar().encode(
            alt.X("monetary_value:Q", bin=True, title="Monetary Value"),
            y="count()",
            tooltip=["count()", "monetary_value"]
        ).interactive()
        st.altair_chart(monetary_chart, use_container_width=True)
    
    with col2:
        # Frequency Distribution
        st.subheader("Purchase Frequency Distribution")
        freq_chart = alt.Chart(result).mark_bar().encode(
            alt.X("frequency:Q", bin=True, title="Purchase Frequency"),
            y="count()",
            tooltip=["count()", "frequency"]
        ).interactive()
        st.altair_chart(freq_chart, use_container_width=True)

def plot_model_matrices(pareto_model, result):
    """Plot model matrices"""
    st.subheader("Customer Behavior Matrices")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Frequency-Recency Matrix
        st.markdown("**Frequency-Recency Matrix**")
        fig = plt.figure(figsize=(10, 6))
        plot_frequency_recency_matrix(pareto_model)
        st.pyplot(fig)
    
    with col2:
        # Probability Alive Matrix
        st.markdown("**Probability Alive Matrix**")
        fig = plt.figure(figsize=(10, 6))
        plot_probability_alive_matrix(pareto_model)
        st.pyplot(fig)

def show_summary_stats(result):
    """Display summary statistics cards"""
    st.subheader("Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", len(result))
    
    with col2:
        st.metric("Average CLV", f"${result['predicted_clv'].mean():.2f}")
    
    with col3:
        st.metric("Total Predicted Value", f"${result['predicted_clv'].sum():,.2f}")
    
    with col4:
        st.metric("Avg. Purchase Frequency", f"{result['frequency'].mean():.1f}")

# --- Main Processing Functions ---
def process_data(input_data, days, profit):
    """Process input data through the CLV prediction pipeline"""
    # Pareto/NBD Model
    pareto_model = ParetoNBDFitter(penalizer_coef=0.1)
    pareto_model.fit(input_data["frequency"], input_data["recency"], input_data["T"])
    
    input_data["p_alive"] = pareto_model.conditional_probability_alive(
        input_data["frequency"], input_data["recency"], input_data["T"]
    )
    
    input_data["predicted_purchases"] = pareto_model.conditional_expected_number_of_purchases_up_to_time(
        days, input_data["frequency"], input_data["recency"], input_data["T"]
    )
    
    # Gamma-Gamma Model (only for customers with frequency > 0)
    input_data = input_data[
        (input_data["frequency"] > 0) & (input_data["monetary_value"] > 0)
    ].copy()
    
    ggf_model = GammaGammaFitter(penalizer_coef=0.1)
    ggf_model.fit(input_data["frequency"], input_data["monetary_value"])
    
    input_data["expected_avg_sales"] = ggf_model.conditional_expected_average_profit(
        input_data["frequency"], input_data["monetary_value"]
    )
    
    input_data["predicted_clv"] = ggf_model.customer_lifetime_value(
        pareto_model,
        input_data["frequency"],
        input_data["recency"],
        input_data["T"],
        input_data["monetary_value"],
        time=days,
        freq='D',
        discount_rate=0.01
    )
    
    input_data["profit_margin"] = input_data["predicted_clv"] * profit
    
    # Customer Segmentation
    features = ["predicted_purchases", "expected_avg_sales", "predicted_clv", "profit_margin"]
    k_model = KMeans(n_clusters=4, init="k-means++", max_iter=1000, random_state=42)
    input_data["segment"] = k_model.fit_predict(input_data[features])
    
    # Map segments to meaningful labels
    segment_map = {0: "Low", 1: "Medium", 2: "High", 3: "Very High"}
    input_data["segment"] = input_data["segment"].map(segment_map)
    
    return input_data, pareto_model

# --- Main App Function ---
def main():
    """Main application function"""
    st.title("Customer Lifetime Value Predictor ")
    st.markdown("""
    Upload your RFM (Recency, Frequency, Monetary) data to get customer lifetime value predictions.
    """)
    
    st.image(
        "https://sarasanalytics.com/wp-content/uploads/2019/11/Customer-Lifetime-value-new-1.jpg",
        use_column_width=True
    )
    
    # Get user inputs
    days, profit = show_sidebar()
    
    # File upload
    uploaded_file = st.file_uploader("Upload RFM Data (CSV)", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read and process data
            data = pd.read_csv(uploaded_file)
            
            # Handle unnamed index column if present
            if data.columns[0].startswith("Unnamed"):
                data = data.iloc[:, 1:]
            
            # Process data through models
            result, pareto_model = process_data(data, days, profit)
            
            # Show summary statistics
            show_summary_stats(result)
            
            # Display results
            st.subheader("Prediction Results ")
            st.dataframe(result.head())
            
            # Customer distributions
            plot_customer_distributions(result)
            
            # Model matrices
            plot_model_matrices(pareto_model, result)
            
            # Customer segments
            st.subheader("Customer Segments")
            chart = alt.Chart(result).mark_bar().encode(
                y=alt.Y("segment:N", sort="-x"),
                x="count(segment):Q",
                color="segment:N",
                tooltip=["count(segment)"]
            ).interactive()
            st.altair_chart(chart, use_container_width=True)
            
            # Download button
            csv = result.to_csv(index=False).encode('utf-8')
            st.download_button(
                " Download Results",
                csv,
                "clv_predictions.csv",
                "text/csv"
            )
            
            st.success(" Analysis complete!")
            
        except Exception as e:
            st.error(f" Error processing file: {str(e)}")
    else:
        st.info(" Please upload a CSV file to begin")

if __name__ == "__main__":
    main()
