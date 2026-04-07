# UIcore.py — JobSage with a Streamlit web interface
# ----------------------------------------------------
# Run with: streamlit run UIcore.py
# Requirements: pip install streamlit langchain-google-genai pydantic python-dotenv

import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional

load_dotenv()

# -------------------- Page Setup --------------------
st.set_page_config(page_title="JobSage", page_icon="🧑‍💼", layout="centered")

st.title("🧑‍💼 JobSage")
st.subheader("Paste any job description — get clean structured data instantly")
st.caption("Powered by Google Gemini + LangChain")

# -------------------- Model --------------------
@st.cache_resource
def get_model():
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

model = get_model()

# -------------------- Output Schema --------------------
class JobPosting(BaseModel):
    job_title: str = Field(description="The job title or role")
    company_name: Optional[str] = Field(description="The company or organisation name")
    location: Optional[str] = Field(description="City, country or 'Remote'")
    job_type: Optional[str] = Field(description="Full-time, Part-time, Contract, Internship, etc.")
    required_skills: List[str] = Field(description="List of required technical skills or technologies")
    nice_to_have_skills: List[str] = Field(description="Optional skills mentioned as a bonus")
    experience_required: Optional[str] = Field(description="Years or level of experience needed")
    salary_range: Optional[str] = Field(description="Salary range if mentioned, else null")
    summary: str = Field(description="A 1-2 sentence plain English summary of the role")

parser = PydanticOutputParser(pydantic_object=JobPosting)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at reading job descriptions and extracting structured information.
Extract all relevant details from the job posting below.
Be precise — only extract what is explicitly mentioned.
{format_instructions}
"""),
    ("human", "Job posting:\n\n{job_description}")
])

# -------------------- Input --------------------
st.divider()
job_text = st.text_area(
    "Paste the full job description here",
    height=250,
    placeholder="e.g. We are looking for a Senior Python Developer at TechCorp in Berlin..."
)

# -------------------- Example Button --------------------
if st.button("📋 Load Example Job Posting"):
    st.session_state["example_loaded"] = True

if st.session_state.get("example_loaded"):
    job_text = """
We are hiring a Senior Machine Learning Engineer at DataVision AI, based in Berlin (Hybrid).
This is a full-time position. We are looking for someone with 4+ years of experience in ML.

Required skills: Python, PyTorch, TensorFlow, scikit-learn, SQL, Docker, Kubernetes.
Nice to have: experience with LLMs, LangChain, MLflow, or AWS SageMaker.

You will design and deploy ML models, collaborate with data engineers, and mentor junior team members.
Salary: €80,000 – €110,000 per year.

Apply if you're passionate about turning data into real-world AI solutions!
""".strip()
    st.text_area("Paste the full job description here", value=job_text, height=250, key="example_text")


# -------------------- Extract Button --------------------
st.divider()
if st.button("🔍 Extract Job Information", type="primary"):
    if not job_text.strip():
        st.warning("Please paste a job description first.")
    else:
        with st.spinner("Reading the job posting..."):
            try:
                final_prompt = prompt.invoke({
                    "job_description": job_text,
                    "format_instructions": parser.get_format_instructions()
                })
                response = model.invoke(final_prompt)
                job = parser.parse(response.content)

                # -------------------- Results --------------------
                st.success("✅ Job info extracted successfully!")
                st.divider()

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Role", job.job_title)
                    st.metric("Company", job.company_name or "Not mentioned")
                    st.metric("Location", job.location or "Not mentioned")
                with col2:
                    st.metric("Job Type", job.job_type or "Not mentioned")
                    st.metric("Experience", job.experience_required or "Not mentioned")
                    st.metric("Salary", job.salary_range or "Not mentioned")

                st.divider()

                col3, col4 = st.columns(2)
                with col3:
                    st.subheader("🔧 Required Skills")
                    for skill in job.required_skills:
                        st.markdown(f"- {skill}")
                with col4:
                    st.subheader("⭐ Nice to Have")
                    if job.nice_to_have_skills:
                        for skill in job.nice_to_have_skills:
                            st.markdown(f"- {skill}")
                    else:
                        st.write("None mentioned")

                st.divider()
                st.subheader("📝 Summary")
                st.info(job.summary)

                st.divider()
                with st.expander("🔍 Raw JSON output"):
                    st.json(job.model_dump())

            except Exception as e:
                st.error("Failed to parse the job posting. Try again or check the format.")
                st.exception(e)
