"""
Phishing Website Detector - Streamlit app.

Paste a URL; the app extracts 37 lexical URL features (no network calls, no page
fetch needed - purely from the URL string) using the same extractor used to build
the training data, scales them with the saved StandardScaler, and classifies with
the trained XGBoost model hosted on Hugging Face
(Twinkytuffy/phishing-url-detector-model): 91.0% accuracy, 0.966 ROC-AUC on the
held-out combined-dataset test set.
"""
import joblib
import numpy as np
import streamlit as st
from huggingface_hub import hf_hub_download

from url_features import extract_url_features

MODEL_REPO = "Twinkytuffy/phishing-url-detector-model"

st.set_page_config(page_title="Phishing Website Detector", page_icon="\U0001F3A3")


@st.cache_resource
def load_model():
    model = joblib.load(hf_hub_download(MODEL_REPO, "xgboost.joblib"))
    scaler = joblib.load(hf_hub_download(MODEL_REPO, "scaler.joblib"))
    feature_names = joblib.load(hf_hub_download(MODEL_REPO, "feature_names.joblib"))
    return model, scaler, feature_names


model, scaler, FEATURE_NAMES = load_model()

st.title("\U0001F3A3 Phishing Website Detector")
st.markdown(
    """
    MSc Dissertation demo — *A Deep Learning Framework for Phishing Website
    Detection Using URL and HTML Features*.

    Paste any URL below. The model (XGBoost, trained on a combined dataset of
    21,260 URLs from two independent sources) extracts 37 lexical URL features
    and classifies the site as **Phishing** or **Legitimate**.

    No page is fetched or visited — classification is based only on the URL
    string itself.
    """
)

url = st.text_input(
    "URL to check",
    placeholder="e.g. http://secure-update-account-verify.example-login.com/webscr",
)

examples = {
    "-- pick an example --": "",
    "Legitimate: Wikipedia article": "https://en.wikipedia.org/wiki/Phishing",
    "Legitimate: Google": "https://www.google.com",
    "Phishing: fake PayPal login": "http://paypal-secure-login.account-verify-update.com/webscr?cmd=login",
    "Phishing: IP-based login form": "http://192.168.1.1/admin/login.php?redirect=account&verify=1",
}
choice = st.selectbox("Or try an example", list(examples.keys()))
if choice != "-- pick an example --" and not url:
    url = examples[choice]

if st.button("Check URL", type="primary") or url:
    if not url or not url.strip():
        st.info("Enter a URL above to check it.")
    else:
        feats = extract_url_features(url)
        x = np.array([[feats[name] for name in FEATURE_NAMES]])
        x_scaled = scaler.transform(x)
        proba_phishing = float(model.predict_proba(x_scaled)[0, 1])
        label = "Phishing" if proba_phishing >= 0.5 else "Legitimate"
        confidence = proba_phishing if label == "Phishing" else 1 - proba_phishing

        if label == "Phishing":
            st.error(f"### \U0001F6A8 {label}")
        else:
            st.success(f"### ✅ {label}")
        st.write(
            f"Confidence: **{confidence * 100:.1f}%**  "
            f"(phishing probability: {proba_phishing * 100:.1f}%)"
        )

        with st.expander("Details"):
            st.write("**Phishing probability (0-1):**", round(proba_phishing, 4))
            st.write("**Key extracted features:**")
            st.markdown(
                f"""
                - **URL length:** {feats['url_length']}
                - **Hostname length:** {feats['hostname_length']}
                - **Has IP address as host:** {'yes' if feats['has_ip_host'] else 'no'}
                - **Uses HTTPS:** {'yes' if feats['has_https'] else 'no'}
                - **Subdomain count:** {feats['qty_subdomains']}
                - **Suspicious keywords found:** {feats['suspicious_word_count']}
                - **URL entropy:** {feats['url_entropy']}
                - **'@' symbol present:** {'yes' if feats['has_at_symbol'] else 'no'}
                """
            )

st.markdown(
    """
    ---
    **Note:** this model was trained on lexical URL features only (no HTML content
    or live page data), so accuracy on the held-out combined test set was 91.0%
    (ROC-AUC 0.966) — lower than models using richer pre-engineered HTML/URL
    features (up to 97.7% on the UCI dataset), because less discriminative
    information is available from the URL string alone. Treat predictions as
    indicative, not definitive.

    **Known limitation:** the training data's legitimate class skews toward short,
    bare domains (e.g. `example.edu/`) rather than `www.`-prefixed global brands, so
    some well-known root-domain URLs (e.g. plain `amazon.com` or `wikipedia.org`
    without a path) can land close to the 50% decision boundary. Adding a specific
    page path, as in the Wikipedia example above, resolves this. This reflects a
    genuine dataset-composition limitation of the lexical-only feature set, not a
    bug in the model or pipeline.
    """
)
