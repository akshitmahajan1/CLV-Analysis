
# #adding necessary libraries
# import streamlit as st
# import pandas as pd
# import lifetimes
# import math
# import numpy as np
# import xlrd
# import datetime
# np.random.seed(42)
# import altair as alt
# import time
# import warnings
# warnings.filterwarnings("ignore")
# from math import sqrt
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# from lifetimes.plotting import plot_frequency_recency_matrix
# from lifetimes.plotting import plot_probability_alive_matrix
# from lifetimes.plotting import plot_period_transactions
# from lifetimes.utils import calibration_and_holdout_data
# from lifetimes import ParetoNBDFitter
# from lifetimes.plotting import plot_history_alive
# from sklearn.metrics import mean_squared_error, r2_score

# st.markdown(""" # Customer Lifetime Prediction App 👋


# Upload the RFM data and get your customer lifetime prediction on the fly !!! :smile:

# 	""")


# st.image("https://sarasanalytics.com/wp-content/uploads/2019/11/Customer-Lifetime-value-new-1.jpg", use_column_width = True)


# data = st.file_uploader("File Uploader")

# st.sidebar.image("http://logok.org/wp-content/uploads/2014/06/City-of-Melbourne-logo-M.png", width = 120)
# st.sidebar.markdown(""" **Made with :heart: by Mukul Singhal** """)


# st.sidebar.title("Input Features :pencil:")


# st.sidebar.markdown("""

# [Example CSV Input File](https://raw.githubusercontent.com/mukulsinghal001/customer-lifetime-prediction-using-python/main/model_deployment/sample_file.csv)

# 	""")

# days = st.sidebar.slider("Select The No. Of Days", min_value = 1, max_value = 365, step = 1, value = 30)

# profit = st.sidebar.slider("Select the Profit Margin", min_value = 0.01, max_value = 0.09, step = 0.01, value = 0.05)


# t_days = days

# profit_m = profit

# slider_data = {
# 	"Days": t_days,
# 	"Profit": profit_m
# }

# st.sidebar.markdown("""

# ### Selected Input Features :page_with_curl:

# 	""")

# features = pd.DataFrame(slider_data, index = [0])

# st.sidebar.write(features)

# st.sidebar.markdown("""

# Before uploading the file, please select the input features first.

# Also, please make sure the columns are in proper format. For reference you can download the [dummy data](https://raw.githubusercontent.com/mukulsinghal001/customer-lifetime-prediction-using-python/main/model_deployment/sample_file.csv).

# **Note:** Only Use "CSV" File.

# 	""")


# if data is not None:

# 	def load_data(data, day = t_days, profit = profit_m):

# 		input_data = pd.read_csv(data)

# 		input_data = pd.DataFrame(input_data.iloc[:, 1:])

#         #Pareto Model

# 		pareto_model = lifetimes.ParetoNBDFitter(penalizer_coef = 0.1)
# 		pareto_model.fit(input_data["frequency"],input_data["recency"],input_data["T"])
# 		input_data["p_not_alive"] = 1-pareto_model.conditional_probability_alive(input_data["frequency"], input_data["recency"], input_data["T"])
# 		input_data["p_alive"] = pareto_model.conditional_probability_alive(input_data["frequency"], input_data["recency"], input_data["T"])
# 		t = days
# 		input_data["predicted_purchases"] = pareto_model.conditional_expected_number_of_purchases_up_to_time(t, input_data["frequency"], input_data["recency"], input_data["T"])
        

#         #Gamma Gamma Model

# 		idx = input_data[(input_data["frequency"] <= 0.0)]
# 		idx = idx.index
# 		input_data = input_data.drop(idx, axis = 0)
# 		m_idx = input_data[(input_data["monetary_value"] <= 0.0)].index
# 		input_data = input_data.drop(m_idx, axis = 0)


# 		input_data.reset_index().drop("index", axis = 1, inplace = True)

# 		ggf_model =  lifetimes.GammaGammaFitter(penalizer_coef=0.1)

# 		ggf_model.fit(input_data["frequency"], input_data["monetary_value"])

# 		input_data["expected_avg_sales_"] = ggf_model.conditional_expected_average_profit(input_data["frequency"], input_data["monetary_value"])

# 		input_data["predicted_clv"] = ggf_model.customer_lifetime_value(pareto_model, input_data["frequency"], input_data["recency"], input_data["T"], input_data["monetary_value"], time = 30, freq = 'D', discount_rate = 0.01)

# 		input_data["profit_margin"] = input_data["predicted_clv"]*profit

# 		input_data = input_data.reset_index().drop("index", axis = 1)

# 		#K-Means Model

# 		col = ["predicted_purchases", "expected_avg_sales_", "predicted_clv", "profit_margin"]

# 		new_df = input_data[col]

# 		k_model = KMeans(n_clusters = 4, init = "k-means++", n_jobs = -1, max_iter = 1000).fit(new_df)

# 		labels = k_model.labels_

# 		labels = pd.Series(labels, name = "Labels")

# 		input_data = pd.concat([input_data, labels], axis = 1)

# 		label_mapper = dict({0 : "Low", 3: "Medium", 1: "High", 2: "V_High"})

# 		input_data["Labels"] = input_data["Labels"].map(label_mapper)

# 		#saving the input data in the separate variable 

# 		download = input_data

# 		st.write(input_data)

# 		#adding a count bar chart

# 		fig = alt.Chart(input_data).mark_bar().encode(

# 			y = "Labels:N",
# 			x = "count(Labels):Q"

# 			)

# 		#adding a annotation to the chart

# 		text = fig.mark_text(

# 			align = "left",
# 			baseline = "middle",
# 			dx = 3

# 			).encode(

# 			text = "count(Labels):Q"

# 			)


# 		chart = (fig+text)

# 		#showing the chart

# 		st.altair_chart(chart, use_container_width = True)

# 		#creating a button to download the result

# 		if st.button("Download"):
# 			st.write("Successfully Downloaded!!! Please Check Your Default Download Location...:smile:" )
# 			return download.to_csv("customer_lifetime_prediction_result.csv")


# 	#calling the function		

# 	st.markdown("""

# 		## Customer Lifetime Prediction Result :bar_chart:

# 		""")

# 	load_data(data)

# else:
# 	st.text("Please Upload the CSV File")



# ==============================
# ✅ Customer Lifetime Value Streamlit App
# ==============================

# --- Import necessary libraries ---
import streamlit as st
import pandas as pd
import lifetimes
import numpy as np
import datetime
import altair as alt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from lifetimes import ParetoNBDFitter, GammaGammaFitter
from lifetimes.plotting import plot_frequency_recency_matrix, plot_probability_alive_matrix, plot_period_transactions
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)

# --- Streamlit app title ---
st.markdown("""
# Customer Lifetime Prediction App 👋

Upload the RFM data and get your customer lifetime prediction on the fly !!! 😄
""")

st.image(
    "https://sarasanalytics.com/wp-content/uploads/2019/11/Customer-Lifetime-value-new-1.jpg",
    use_column_width=True
)

# --- File uploader ---
data = st.file_uploader("Upload your RFM CSV file")

# --- Sidebar content ---
st.sidebar.image("http://logok.org/wp-content/uploads/2014/06/City-of-Melbourne-logo-M.png", width=120)
st.sidebar.markdown("**Made by Akshit Mahajan**")
st.sidebar.title("Input Features ✏️")

# st.sidebar.markdown("""
# [Example CSV Input File](https://raw.githubusercontent.com/mukulsinghal001/customer-lifetime-prediction-using-python/main/model_deployment/sample_file.csv)
# """)

days = st.sidebar.slider("Select The No. Of Days", min_value=1, max_value=365, step=1, value=30)
profit = st.sidebar.slider("Select the Profit Margin", min_value=0.01, max_value=0.09, step=0.01, value=0.05)

# Display sidebar summary
features = pd.DataFrame({"Days": [days], "Profit": [profit]})
st.sidebar.subheader("Selected Input Features 📄")
st.sidebar.write(features)

st.sidebar.markdown("""
Before uploading the file, please select the input features first.


**Note:** Only use CSV files.
""")

# --- Main processing ---
if data is not None:

    def load_data(data, day=days, profit=profit):
        # Read data
        # input_data = pd.read_csv(data)
        input_data = pd.read_csv(data, encoding='cp1252')
        input_data = pd.DataFrame(input_data.iloc[:, 1:])  # ignore first unnamed index column if present

        # =====================
        # 1️⃣ Pareto/NBD Model
        # =====================
        pareto_model = ParetoNBDFitter(penalizer_coef=0.1)
        pareto_model.fit(input_data["frequency"], input_data["recency"], input_data["T"])

        input_data["p_not_alive"] = 1 - pareto_model.conditional_probability_alive(
            input_data["frequency"], input_data["recency"], input_data["T"]
        )
        input_data["p_alive"] = pareto_model.conditional_probability_alive(
            input_data["frequency"], input_data["recency"], input_data["T"]
        )

        input_data["predicted_purchases"] = pareto_model.conditional_expected_number_of_purchases_up_to_time(
            day, input_data["frequency"], input_data["recency"], input_data["T"]
        )

        # =====================
        # 2️⃣ Gamma-Gamma Model
        # =====================
        input_data = input_data[(input_data["frequency"] > 0) & (input_data["monetary_value"] > 0)].copy()
        ggf_model = GammaGammaFitter(penalizer_coef=0.1)
        ggf_model.fit(input_data["frequency"], input_data["monetary_value"])

        input_data["expected_avg_sales_"] = ggf_model.conditional_expected_average_profit(
            input_data["frequency"], input_data["monetary_value"]
        )

        input_data["predicted_clv"] = ggf_model.customer_lifetime_value(
            pareto_model,
            input_data["frequency"],
            input_data["recency"],
            input_data["T"],
            input_data["monetary_value"],
            time=day,
            freq='D',
            discount_rate=0.01
        )

        input_data["profit_margin"] = input_data["predicted_clv"] * profit

        # =====================
        # 3️⃣ K-Means Clustering
        # =====================
        col = ["predicted_purchases", "expected_avg_sales_", "predicted_clv", "profit_margin"]
        new_df = input_data[col]

        # ✅ Removed deprecated n_jobs param
        k_model = KMeans(n_clusters=4, init="k-means++", max_iter=1000, random_state=42).fit(new_df)

        input_data["Labels"] = pd.Series(k_model.labels_, name="Labels")

        label_mapper = {0: "Low", 1: "Medium", 2: "High", 3: "Very High"}
        input_data["Labels"] = input_data["Labels"].map(label_mapper)

        # =====================
        # ✅ Output + Visualization
        # =====================
        st.write("### Predicted Customer Lifetime Metrics")
        st.dataframe(input_data.head())

        # Bar chart
        fig = alt.Chart(input_data).mark_bar().encode(
            y=alt.Y("Labels:N", sort="-x"),
            x="count(Labels):Q"
        )

        text = fig.mark_text(align="left", baseline="middle", dx=3).encode(text="count(Labels):Q")
        chart = fig + text
        st.altair_chart(chart, use_container_width=True)

        # Download button
        csv = input_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Prediction Results",
            data=csv,
            file_name="customer_lifetime_prediction_result.csv",
            mime="text/csv"
        )

        st.success("✅ Model predictions and clustering complete!")

    # --- Execute ---
    st.subheader("## Customer Lifetime Prediction Result 📊")
    load_data(data)

else:
    st.info("Please upload a CSV file to begin.")
