import streamlit as st
import joblib

# Load trained models
tfidf = joblib.load("tfidf_vectorizer.pkl")
category_model = joblib.load("category_model.pkl")
priority_model = joblib.load("priority_model.pkl")

# Page title
st.title("🎫 Customer IT Support Ticket Classifier")

st.write("Enter a customer support ticket to predict its category and priority.")

# Input
ticket = st.text_area(
    "Customer Ticket",
    placeholder="Example: I was charged twice for my subscription and need a refund."
)

# Predict button
if st.button("Predict"):
    if ticket.strip() == "":
        st.warning("Please enter a ticket.")
    else:
        # Convert text to TF-IDF
        ticket_tfidf = tfidf.transform([ticket])

        # Predictions
        predicted_category = category_model.predict(ticket_tfidf)[0]
        predicted_priority = priority_model.predict(ticket_tfidf)[0]

        # Probabilities
        category_probabilities = category_model.predict_proba(ticket_tfidf)[0]
        priority_probabilities = priority_model.predict_proba(ticket_tfidf)[0]

        category_confidence = category_probabilities.max()
        priority_confidence = priority_probabilities.max()

        # Overall confidence
        overall_confidence = (
            category_confidence + priority_confidence
        ) / 2

        # Display results
        st.subheader("Prediction Results")

        st.success(f"Category: {predicted_category}")
        st.info(f"Priority: {predicted_priority}")

        st.write(
            f"Category Confidence: {category_confidence * 100:.2f}%"
        )

        st.write(
            f"Priority Confidence: {priority_confidence * 100:.2f}%"
        )

        st.write(
            f"Overall Confidence: {overall_confidence * 100:.2f}%"
        )

        # Human review
        if overall_confidence < 0.60:
            st.warning("⚠️ Low confidence — Human review required.")
        else:
            st.success("✅ Confidence acceptable — Automatic routing.")