# Phishing Website Detector

MSc Dissertation demo — *A Deep Learning Framework for Phishing Website Detection Using URL and HTML Features.*
Student: Sushma Singahalli Parashuramappa (250638286). Supervisor: Bo Wei.

A Streamlit app that classifies a URL as **Phishing** or **Legitimate**, using an
XGBoost model trained on a combined dataset of 21,260 URLs from two independent
sources (test accuracy 91.0%, ROC-AUC 0.966). Classification uses only 37 lexical
features extracted from the URL string itself — no page is fetched or visited.

Model weights are hosted for free on the Hugging Face Hub:
[Twinkytuffy/phishing-url-detector-model](https://huggingface.co/Twinkytuffy/phishing-url-detector-model)

## Run locally
```
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub (already done if you're reading this on GitHub).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. "New app" → select this repo → main file path: `streamlit_app.py` → Deploy.

## Files
- `streamlit_app.py` - the Streamlit UI and prediction logic.
- `url_features.py` - the lexical URL feature extractor (identical to the one used
  to build the training data).
- `requirements.txt` - pinned dependencies.

## Known limitation
The training data's legitimate class skews toward short, bare domains rather than
`www.`-prefixed global brands, so some well-known root-domain-only URLs can land
close to the 50% decision boundary. This is a dataset-composition limitation of the
lexical-only feature set, not a bug.
