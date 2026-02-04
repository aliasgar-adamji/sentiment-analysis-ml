import streamlit as st
import pickle
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# =====================================================
# PAGE CONFIG (must be first Streamlit command)
# =====================================================
st.set_page_config(
    page_title="Flipkart Sentiment Analyzer",
    page_icon="🛍️",
    layout="centered"
)

# =====================================================
# DOWNLOAD NLTK DATA (AWS SAFE METHOD)
# Stores locally inside project instead of system path
# =====================================================
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
nltk.data.path.append(nltk_data_path)


@st.cache_resource
def setup_nltk():
    nltk.download("stopwords", download_dir=nltk_data_path)
    nltk.download("wordnet", download_dir=nltk_data_path)
    nltk.download("punkt", download_dir=nltk_data_path)
    nltk.download("omw-1.4", download_dir=nltk_data_path)


setup_nltk()

# =====================================================
# LOAD MODEL + VECTORIZER (cached for speed)
# =====================================================
@st.cache_resource
def load_model():
    with open("sentiment_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


model, vectorizer = load_model()

# =====================================================
# NLP PREPROCESSING
# =====================================================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text(text):
    text = clean_text(text)
    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]
    return " ".join(tokens)


# =====================================================
# UI HEADER
# =====================================================
st.title("🛍️ Flipkart Product Review Sentiment Analyzer")
st.markdown("### Analyze customer reviews instantly — Positive or Negative")

st.markdown("---")

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.header("📊 About")

    st.info(
        """
        This app classifies Flipkart reviews:

        ✅ Positive (4–5 ⭐)  
        ❌ Negative (1–2 ⭐)

        Model: Logistic Regression  
        Features: TF-IDF  
        """
    )

    st.header("🔧 Tech Stack")
    st.markdown(
        """
        - Python  
        - Scikit-learn  
        - NLTK  
        - Streamlit  
        - AWS EC2 (deployment)
        """
    )


# =====================================================
# MAIN INPUT
# =====================================================
st.subheader("📝 Enter Product Review")

user_input = st.text_area(
    "Type or paste review below:",
    height=150,
    placeholder="Example: This product is amazing! Best purchase ever."
)

analyze_button = st.button(
    "🔍 Analyze Sentiment",
    use_container_width=True
)

# =====================================================
# PREDICTION LOGIC
# =====================================================
if analyze_button:

    if not user_input.strip():
        st.warning("⚠️ Please enter a review first")
        st.stop()

    with st.spinner("Analyzing sentiment..."):

        processed_text = preprocess_text(user_input)

        review_vector = vectorizer.transform([processed_text])

        prediction = model.predict(review_vector)[0]

        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(review_vector)[0]

    st.markdown("---")
    st.subheader("📊 Analysis Result")

    # Result
    if prediction == 1:
        st.success("### ✅ POSITIVE Sentiment")
        st.balloons()
    else:
        st.error("### ❌ NEGATIVE Sentiment")

    # Confidence
    if proba is not None:
        confidence = max(proba) * 100
        st.metric("Confidence", f"{confidence:.2f}%")
        st.progress(int(confidence))

    # Processed text
    with st.expander("🔍 See processed text"):
        st.code(processed_text)


# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.caption("Built by Aliasgar Alihusain  | Deployed on AWS EC2")

