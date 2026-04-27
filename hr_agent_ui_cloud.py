import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate

# -------------------------------------------------
# 🔑 OPENAI API KEY (hard-coded for demo)
# -------------------------------------------------
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


# -------------------------------------------------
# 🖥️ Streamlit App Config
# -------------------------------------------------
st.set_page_config(page_title="HR Resume Screening AI Agent", layout="wide")
st.title("🤖 HR Resume Screening AI Agent")
st.write(
    "Upload resumes, paste a Job Description, and let the AI agent "
    "shortlist candidates using GenAI + Agentic reasoning."
)

# -------------------------------------------------
# 📤 Upload Resume PDFs (UI ONLY)
# -------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload multiple resume PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

# -------------------------------------------------
# 📝 Job Description Input
# -------------------------------------------------
job_description = st.text_area(
    "Paste Job Description",
    height=180,
    placeholder="Example: Looking for a Data Scientist with Python, ML, SQL, and GenAI exposure..."
)

# -------------------------------------------------
# ▶️ Run Button
# -------------------------------------------------
if st.button("Run AI Resume Screening"):

    if not uploaded_files:
        st.warning("Please upload at least one resume PDF.")
        st.stop()

    if not job_description.strip():
        st.warning("Please provide a Job Description.")
        st.stop()

    # -------------------------------------------------
    # 📄 Load Uploaded Resumes
    # -------------------------------------------------
    documents = []

    with st.spinner("Loading resumes..."):
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            for doc in docs:
                doc.metadata["candidate_file"] = file.name

            documents.extend(docs)
            os.remove(tmp_path)

    st.success(f"Loaded {len(documents)} resume pages")

    # -------------------------------------------------
    # 🧠 Create Vector Store (AI Memory)
    # -------------------------------------------------
    with st.spinner("Creating AI memory (vector database)..."):
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(documents, embeddings)

    # -------------------------------------------------
    # 🔍 Retrieve Top Matching Resumes
    # -------------------------------------------------
    with st.spinner("Finding best matching candidates..."):
        top_docs = vectorstore.similarity_search(job_description, k=5)

    st.info(f"Evaluating top {len(top_docs)} candidates")

    # -------------------------------------------------
    # 🤖 HR AI Agent Prompt
    # -------------------------------------------------
    prompt = PromptTemplate(
        input_variables=["resume", "jd"],
        template="""
You are an HR AI Agent.

Goal: Evaluate the resume against the Job Description.

Job Description:
{jd}

Resume:
{resume}

Tasks:
1. Score the candidate out of 10
2. Mention key strengths
3. Mention missing or weak areas
4. Final recommendation: Shortlist / Maybe / Reject

Be concise, unbiased, and professional.
"""
    )

    llm = ChatOpenAI(temperature=0)

    # -------------------------------------------------
    # 📊 Evaluate Candidates
    # -------------------------------------------------
    st.subheader("📊 AI Evaluation Results")

    for idx, doc in enumerate(top_docs, start=1):
        candidate = doc.metadata.get("candidate_file", "Unknown")

        with st.expander(f"Candidate {idx} — {candidate}"):
            with st.spinner("Evaluating candidate..."):
                response = llm.invoke(
                    prompt.format(
                        resume=doc.page_content,
                        jd=job_description
                    )
                )

            st.write(response.content)

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption("Agentic AI Demo | HR Resume Screening | Python · LangChain · Streamlit")
