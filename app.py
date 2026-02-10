import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(text1, text2):
    # Vectorize the text to convert words into numbers
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([text1, text2])
    
    # Calculate the Cosine Similarity
    similarity = cosine_similarity(vectors)
    return similarity[0][1]

# Streamlit UI
st.set_page_config(page_title="Resume Skill Matcher", page_icon="📄")
st.title("📄 Resume Skill Matcher")
st.write("Compare your resume against a job description using local NLP.")

col1, col2 = st.columns(2)

with col1:
    resume_text = st.text_area("Paste your Resume here:", height=300)

with col2:
    job_desc_text = st.text_area("Paste the Job Description here:", height=300)

if st.button("Calculate Match Score"):
    if resume_text and job_desc_text:
        score = calculate_similarity(resume_text, job_desc_text)
        percentage = round(score * 100, 2)
        
        st.subheader(f"Match Score: {percentage}%")
        
        if percentage > 70:
            st.success("🔥 High Match! Your resume aligns well with this role.")
        elif percentage > 40:
            st.warning("⚡ Moderate Match. Consider adding more keywords from the JD.")
        else:
            st.error("❄️ Low Match. You might need to tailor your resume more.")
    else:
        st.info("Please paste both the resume and the job description to continue.")
