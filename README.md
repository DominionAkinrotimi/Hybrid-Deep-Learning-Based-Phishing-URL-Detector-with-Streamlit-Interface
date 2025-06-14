# Phishing URL Detection (Streamlit App)

This project is a real-time phishing detection system powered by a hybrid deep learning framework. Users simply paste a URL, and the system extracts relevant features, processes them using a pre-trained ensemble model, and returns a prediction on whether the URL is legitimate or phishing.

## 🚀 Built with:
- Python & Streamlit
- TensorFlow (CNN, LSTM, FCNN)
- XGBoost
- Scikit-learn
- URL parsing & feature engineering (tldextract, urllib)

## 🔍 How It Works
1. User enters a raw URL (e.g., `https://secure-login.example.com`).
2. The app automatically extracts structural features like:
   - **Domain and TLD**
   - **URL length, special characters**
   - **HTTPS usage, IP address presence**
   - **Entropy, subdomain count**
3. These features are scaled and encoded using previously trained preprocessing tools.
4. The CNN model extracts high-level features.
5. These are passed to:
   - LSTM (for sequential patterns)
   - XGBoost (for structured decision boundaries)
6. Their outputs are combined using a Fully Connected Neural Network (FCNN).
7. The app outputs a phishing probability and final label.

## 🧠 Model Architecture
- CNN → Feature Extractor
- LSTM → Temporal Sequence Detector
- XGBoost → Structured Pattern Learner
- FCNN → Stacked Ensemble Meta-Learner
_All models are pre-trained and saved in `deploy_artifacts/`._

## 📦 Installation
Clone this repository and install dependencies:
```
git clone https://github.com/yourusername/phishing-detector.git
cd phishing-detector
pip install -r requirements.txt
```

## Run the app:
```
streamlit run app.py
```

## 📁 Folder Structure
│
├── app.py # Streamlit app
├── requirements.txt # Project dependencies
├── README.md # You're here
└── deploy_artifacts/ # Pretrained models and encoders
├── cnn_model.h5
├── cnn_feature_extractor.h5
├── lstm_model.h5
├── fcnn_meta_learner.h5
├── xgboost_model.json
├── scaler.pkl
├── ordinal_encoder.pkl
└── selected_features.json

## Requirements
Python 3.9+
Install dependencies via:
```
pip install -r requirements.txt
```

## 👨‍💻 Author
Built by Dominion Akinrotimi – Data Scientist, Research Enthusiast, and ML Developer from Nigeria.

📫 Reach out on LinkedIn: www.linkedin.com/in/dominion-akinrotimi
