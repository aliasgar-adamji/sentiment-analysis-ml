
# Problem Statement

Customer reviews contain valuable insights about product quality, delivery experience, and overall satisfaction. However, analyzing thousands of textual reviews manually is inefficient and impractical for businesses.

This project aims to develop an automated **Sentiment Analysis system** that leverages Natural Language Processing (NLP) and Machine Learning to classify Flipkart product reviews as Positive or Negative.

The system performs:
- Text preprocessing and cleaning
- Feature extraction using TF-IDF and Bag-of-Words
- Model training and evaluation using classical ML algorithms
- Deployment via Streamlit for real-time predictions

The final solution enables fast, scalable, and data-driven understanding of customer feedback.

------------------------------------------------------------------------

##  Flipkart Review Sentiment Analysis

A machine learning web app that predicts whether a Flipkart product
review is **Positive ** or **Negative ** using NLP and classical ML
models.

Built using TF-IDF + Logistic Regression and deployed with
Streamlit.

------------------------------------------------------------------------

##  Features

-   Text preprocessing (cleaning, stopwords removal, lemmatization)
-   TF-IDF & Bag-of-Words embeddings
-   Multiple ML models comparison
-   Cross-validation based model selection
-   Streamlit web interface
-   Ready for AWS EC2 deployment

------------------------------------------------------------------------

##  Tech Stack

-   Python
-   Pandas, NumPy
-   Scikit-learn
-   NLTK
-   Streamlit

------------------------------------------------------------------------

##  Project Structure

    flipkart-sentiment-analysis/
    │── data/             # Raw & Pre-processed
    ├── notebooks/        # EDA & experiments              
    ├── models/           # saved model + vectorizer
    ├── app/              # Streamlit app
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

##  Run Locally

### 1. Create virtual environment

``` bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install packages

``` bash
pip install -r requirements.txt
```

### 3. Run Streamlit

``` bash
streamlit run app/app.py
```

Open browser:

    http://localhost:8501

------------------------------------------------------------------------

##  Deploy on AWS (EC2)

``` bash
pip install streamlit
streamlit run app/app.py --server.port 8501 --server.address 0.0.0.0
```

Open:

    http://18.183.80.16:8501/

    In case of any issue arises during testing of app, connect with author : https://www.linkedin.com/in/aliasgar-alihusain-9a852121a/

------------------------------------------------------------------------

##  Model Performance (Example)

  Model                 F1 Score
  --------------------- ----------
  Logistic Regression   0.95
  Naive Bayes           0.96
  SVM                   0.94

------------------------------------------------------------------------



## 👤 Author

Aliasgar Alihusain\
Data Science \| ML \| NLP

------------------------------------------------------------------------

If you like this project, feel free to star the repo!
