import streamlit as st
import numpy as np
import pandas as pd
import pickle
import json
import tensorflow as tf
import xgboost as xgb
from tensorflow.keras.models import load_model
import tldextract
from urllib.parse import urlparse
import re
from collections import Counter

# ------------------------------
# Load All Artifacts
# ------------------------------
with open("deploy_artifacts/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("deploy_artifacts/ordinal_encoder.pkl", "rb") as f:
    ordinal_encoder = pickle.load(f)

with open("deploy_artifacts/selected_features.json", "r") as f:
    selected_features = json.load(f)

cnn_model = load_model("deploy_artifacts/cnn_model.h5")
feature_extractor = load_model("deploy_artifacts/cnn_feature_extractor.h5")
lstm_model = load_model("deploy_artifacts/lstm_model.h5")
fcnn_model = load_model("deploy_artifacts/fcnn_meta_learner.h5")

xgb_model = xgb.XGBClassifier()
xgb_model.load_model("deploy_artifacts/xgboost_model.json")

# ------------------------------
# Feature Extraction from URL
# ------------------------------
def entropy(s):
    p, lns = Counter(s), float(len(s))
    return -sum(count/lns * np.log2(count/lns) for count in p.values())

def extract_url_features(url: str) -> dict:
    parsed = urlparse(url)
    ext = tldextract.extract(url)

    domain = ext.domain + '.' + ext.suffix if ext.suffix else ext.domain
    tld = ext.suffix or ''

    features = {
        'Domain': domain,
        'TLD': tld,
        'URLLength': len(url),
        'HasHTTPS': int('https' in parsed.scheme),
        'NumDots': url.count('.'),
        'NumHyphens': url.count('-'),
        'HasAtSymbol': int('@' in url),
        'HasIPAddress': int(bool(re.search(r'\d+\.\d+\.\d+\.\d+', parsed.netloc))),
        'NumSubdomains': len(ext.subdomain.split('.')) if ext.subdomain else 0,
        'URL_Entropy': entropy(url),
        'HasSpecialChars': sum([url.count(sym) for sym in ['@', '&', '%', '=', '?']]),
        # Fill in reasonable defaults for missing engineered features
        'URLSimilarityIndex': 0.5,
        'NoOfJS': 0,
        'HasSocialNet': 0,
        'DomainTitleMatchScore': 0.5,
        'DomainEntropy': 0.5
    }

    return features

# ------------------------------
# Feature Processing
# ------------------------------
def extract_input_features(input_dict):
    df = pd.DataFrame([input_dict])
    df[['Domain', 'TLD']] = ordinal_encoder.transform(df[['Domain', 'TLD']])
    df = df[selected_features]
    scaled = scaler.transform(df)
    return scaled

# ------------------------------
# Predict Function
# ------------------------------
def predict_phishing(input_features):
    input_cnn = np.expand_dims(input_features, axis=2)
    cnn_proba = cnn_model.predict(input_cnn)[0][0]

    cnn_feats = feature_extractor.predict(input_cnn)

    lstm_input = np.expand_dims(cnn_feats, axis=1)
    lstm_proba = lstm_model.predict(lstm_input)[0][0]

    xgb_proba = xgb_model.predict_proba(cnn_feats)[0][1]

    stacked_input = np.array([[cnn_proba, lstm_proba, xgb_proba]])
    final_proba = fcnn_model.predict(stacked_input)[0][0]

    return final_proba

# ------------------------------
# Streamlit Interface
# ------------------------------
st.title("Phishing URL Detection")
st.write("Enter a URL. The system will extract features and predict if it's phishing.")

with st.form("url_form"):
    url = st.text_input("Enter a URL", "https://example.com")
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        input_dict = extract_url_features(url)
        scaled_input = extract_input_features(input_dict)
        prediction = predict_phishing(scaled_input)
        label = "Phishing" if prediction >= 0.5 else "Legitimate"
        st.success(f"Prediction: {label} ({prediction:.2f} confidence)")
    except Exception as e:
        st.error(f"Error: {str(e)}")
