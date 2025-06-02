import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load('tfidf_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# UI
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.markdown("## 📰 Fake News Detector")
st.markdown("Enter a news article or headline below to check if it's **Real** or **Fake**.")

news_text = st.text_area("📄 News Text", height=200)
if st.button("Detect"):
    if news_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        vect = vectorizer.transform([news_text])
        prediction = model.predict(vect)[0]
        proba = model.predict_proba(vect)[0]

        st.markdown("### Prediction:")
        if prediction == 1:
            st.success("🟢 Real News")
        else:
            st.error("🔴 Fake News")

        st.markdown(f"**Model confidence:** `{max(proba)*100:.2f}% confidence`")
