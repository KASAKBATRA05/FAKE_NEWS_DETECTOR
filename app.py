import streamlit as st
import joblib
import numpy as np

model = joblib.load("tfidf_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector")
st.markdown("Enter a news article or headline below to check if it's **Real** or **Fake**.")

news_text = st.text_area("📝 News Text", height=200)

if st.button("Detect"):
    if news_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        vect = vectorizer.transform([news_text])
        prediction = model.predict(vect)[0]
        prob = model.predict_proba(vect)[0]

        label = "🟢 Real News" if prediction == 1 else "🔴 Fake News"
        confidence = f"{np.max(prob)*100:.2f}% confidence"

        st.subheader("Prediction:")
        st.success(label)
        st.write(f"Model confidence: **{confidence}**")
