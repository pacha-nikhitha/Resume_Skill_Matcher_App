import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf(file):
    """Extracts text from an uploaded PDF file."""
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def calculate_similarity(resume_text, jd_text):
    """Calculates the cosine similarity between two strings."""
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([resume_text, jd_text])
    similarity = cosine_similarity(vectors)
    return similarity[0][1]

# UI Setup
st.set_page_config(page_title="Pro Skill Matcher", page_icon="🎯")
st.title("🎯 Resume-JD Skill Matcher")
st.markdown("Upload your resume and paste the job description to see how well you match.")

# Inputs
uploaded_file = st.file_uploader("Upload Resume (PDF format)", type=["pdf"])
job_description = st.text_area("Paste the Job Description here:", height=250)

# Action Button
if st.button("Skill Check"):
    if uploaded_file is not None and job_description.strip() != "":
        with st.spinner('Analyzing your skills...'):
            # Processing on the server
            resume_content = extract_text_from_pdf(uploaded_file)
            score = calculate_similarity(resume_content, job_description)
            percentage = round(score * 100, 2)
            
            # Results display
            st.divider()
            st.subheader(f"Match Score: {percentage}%")
            
            if percentage > 75:
                st.success("🌟 Excellent Match! Your profile is highly relevant.")
            elif percentage > 45:
                st.warning("⚖️ Good Match. Consider adding a few more keywords from the JD.")
            else:
                st.error("📉 Low Match. You may need to tailor your resume significantly.")
    else:
        st.error("Please ensure you have uploaded a PDF and pasted a job description.")
