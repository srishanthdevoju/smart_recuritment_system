import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.express as px


# ==================================================
# CUSTOM POSITIONAL ENCODING LAYER
# ==================================================

@tf.keras.utils.register_keras_serializable()
class PositionalEncoding(tf.keras.layers.Layer):

    def call(self, x):
        seq_len = tf.shape(x)[1]

        positions = tf.range(
            start=0,
            limit=seq_len,
            delta=1
        )

        positions = tf.cast(
            positions,
            tf.float32
        )

        return x + tf.expand_dims(
            positions,
            axis=-1
        )

    def get_config(self):
        return super().get_config()


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Fraud Intelligence Dashboard",
    page_icon="💳",
    layout="wide"
)
def load_css():

    st.markdown("""
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"]{
        font-family:'Outfit',sans-serif;
    }

    .stApp{

        background:
        radial-gradient(circle at top left,
        rgba(0,255,255,0.15),
        transparent 30%),

        radial-gradient(circle at top right,
        rgba(59,130,246,0.15),
        transparent 30%),

        radial-gradient(circle at bottom,
        rgba(99,102,241,0.15),
        transparent 35%),

        #020617;

        color:white;
    }

    #MainMenu{
        visibility:hidden;
    }

    footer{
        visibility:hidden;
    }

    header{
        visibility:hidden;
    }

    [data-testid="stSidebar"]{

        background:
        linear-gradient(
            180deg,
            #0f172a,
            #111827
        );

        border-right:
        1px solid rgba(255,255,255,0.08);

        backdrop-filter:blur(25px);
    }

    [data-testid="stSidebar"] *{
        color:white !important;
    }

    h1{

        font-size:3rem !important;

        font-weight:800 !important;

        background:
        linear-gradient(
            90deg,
            #00e5ff,
            #60a5fa,
            #6366f1
        );

        -webkit-background-clip:text;

        -webkit-text-fill-color:transparent;
    }

    h2,h3{
        color:#f8fafc;
    }

    div[data-testid="metric-container"]{

        background:
        rgba(255,255,255,0.04);

        backdrop-filter:
        blur(18px);

        border:
        1px solid rgba(255,255,255,0.08);

        border-radius:20px;

        padding:20px;

        transition:0.4s;
    }

    div[data-testid="metric-container"]:hover{

        transform:
        translateY(-8px);

        box-shadow:
        0 15px 40px rgba(0,229,255,0.25);
    }

    div[data-testid="metric-container"] label{

        color:#94a3b8 !important;
    }

    div[data-testid="metric-container"] div{

        color:#00e5ff !important;

        font-size:34px;
    }

    .stButton button{

        width:100%;

        border:none;

        border-radius:16px;

        background:
        linear-gradient(
            135deg,
            #00e5ff,
            #2563eb
        );

        color:white;

        font-weight:700;

        height:55px;

        transition:0.35s;
    }

    .stButton button:hover{

        transform:scale(1.03);

        box-shadow:
        0 0 25px rgba(0,229,255,0.6);
    }

    [data-testid="stFileUploader"]{

        border:
        2px dashed #00e5ff;

        border-radius:20px;

        padding:25px;

        background:
        rgba(0,229,255,0.05);
    }

    [data-testid="stDataFrame"]{

        border-radius:18px;

        overflow:hidden;

        border:
        1px solid rgba(255,255,255,0.08);
    }

    .js-plotly-plot{

        border-radius:18px;
    }

    .stAlert{

        border-radius:18px;
    }

    ::-webkit-scrollbar{
        width:8px;
    }

    ::-webkit-scrollbar-thumb{

        background:#00e5ff;

        border-radius:10px;
    }

    @keyframes glow{

        0%{
            box-shadow:0 0 10px rgba(0,229,255,0.2);
        }

        50%{
            box-shadow:0 0 30px rgba(0,229,255,0.5);
        }

        100%{
            box-shadow:0 0 10px rgba(0,229,255,0.2);
        }
    }

    .hero-card{

        background:
        rgba(255,255,255,0.05);

        backdrop-filter:
        blur(20px);

        border:
        1px solid rgba(255,255,255,0.08);

        border-radius:24px;

        padding:25px;

        animation:glow 4s infinite;
    }

    </style>
    """, unsafe_allow_html=True)

load_css()

# ==================================================
# LOAD MODEL
# ==================================================

@st.cache_resource
def load_model():

    import os

    st.write("Current Files:")
    st.write(os.listdir("."))

    try:

        model = tf.keras.models.load_model(
            "attention_model.keras",
            custom_objects={
                "PositionalEncoding": PositionalEncoding
            },
            compile=False
        )

        st.success("Model Loaded Successfully")

        return model

    except Exception as e:

        st.error("MODEL LOAD FAILED")
        st.exception(e)

        raise e

@st.cache_resource
def load_scaler():
    return joblib.load("scaler .pkl")


model = load_model()
scaler = load_scaler()


# ==================================================
# HEADER
# ==================================================
st.markdown("""
<div class="hero-card">
    <h1>💳 Fraud Intelligence Dashboard</h1>
    <p style="font-size:18px;color:#cbd5e1;">
        LSTM • Self Attention • Positional Encoding • Real-Time Fraud Detection
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    """
    Upload a transaction CSV file and predict
    potentially fraudulent transactions using
    LSTM + Attention + Positional Encoding.
    """
)


# ==================================================
# FILE UPLOADER
# ==================================================

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)


# ==================================================
# PREDICTION
# ==================================================

if uploaded_file is not None:

    try:

        df = pd.read_csv(uploaded_file)

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.write("Rows:", df.shape[0])
        st.write("Columns:", df.shape[1])

        if st.button("Analyze Transactions"):

            working_df = df.copy()

            if "Class" in working_df.columns:
                features_df = working_df.drop(
                    columns=["Class"]
                )
            else:
                features_df = working_df

            feature_values = features_df.values

            # Scale Features
            scaled_features = scaler.transform(
                feature_values
            )

            # Create Sequences
            SEQ_LEN = 5

            sequences = []

            for i in range(
                len(scaled_features) - SEQ_LEN
            ):
                sequences.append(
                    scaled_features[
                        i:i + SEQ_LEN
                    ]
                )

            X = np.array(sequences)

            if len(X) == 0:
                st.error(
                    "Dataset must contain at least 6 rows."
                )
                st.stop()

            # Predict
            predictions = model.predict(
                X,
                verbose=0
            )

            fraud_prob = predictions.flatten()

            results = df.iloc[
                SEQ_LEN:
            ].copy()

            results[
                "Fraud_Probability"
            ] = fraud_prob

            # Metrics
            avg_prob = fraud_prob.mean() * 100

            high_risk = results[
                results[
                    "Fraud_Probability"
                ] > 0.80
            ]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Transactions",
                    len(results)
                )

            with col2:
                st.metric(
                    "Average Fraud Risk",
                    f"{avg_prob:.2f}%"
                )

            with col3:
                st.metric(
                    "High Risk Transactions",
                    len(high_risk)
                )

            # High Risk Table
            st.subheader(
                "🚨 High Risk Transactions"
            )

            if len(high_risk) > 0:
                st.dataframe(high_risk)
            else:
                st.success(
                    "No high-risk transactions detected."
                )

            # Histogram
            st.subheader(
                "Fraud Probability Distribution"
            )

            fig = px.histogram(
                results,
                x="Fraud_Probability",
                nbins=30,
                title="Fraud Probability Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # Top 10 Risky Transactions
            st.subheader(
                "Top 10 Riskiest Transactions"
            )

            top10 = results.sort_values(
                by="Fraud_Probability",
                ascending=False
            ).head(10)

            st.dataframe(top10)

            # Download Results
            csv = results.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download Predictions",
                data=csv,
                file_name="fraud_predictions.csv",
                mime="text/csv"
            )

    except Exception as e:

        st.error(
            f"Prediction Error: {str(e)}"
        )
