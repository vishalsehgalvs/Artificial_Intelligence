# JobSage — Extract structured job info from any job posting
# ------------------------------------------------------------
# You paste a raw job description (the messy paragraph from a job site),
# and the AI parses it into clean, structured data:
#   title, company, location, skills, salary, experience, job type
#
# Key concepts shown here:
#   - ChatPromptTemplate  → reusable prompt with placeholders
#   - Pydantic BaseModel  → defines the output structure (like a form)
#   - PydanticOutputParser → tells the LLM to fill in that form
#   - Structured output   → guaranteed machine-readable JSON response

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional

# -------------------- Model --------------------
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
# temperature=0 because we want consistent, structured output — not creativity

# -------------------- Output Schema --------------------
# Think of this as the form we want the AI to fill in.
# The Field descriptions tell the AI what each field means.
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

# -------------------- Parser --------------------
# The parser reads the model's response and turns it into a JobPosting object
parser = PydanticOutputParser(pydantic_object=JobPosting)

# -------------------- Prompt --------------------
# {format_instructions} is auto-populated by the parser
# It tells the LLM exactly what JSON structure to produce
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at reading job descriptions and extracting structured information.
Extract all relevant details from the job posting below.
Be precise — only extract what is explicitly mentioned, don't make things up.
{format_instructions}
"""),
    ("human", "Job posting:\n\n{job_description}")
])

# -------------------- Run --------------------
print("=" * 60)
print("🧑‍💼 JobSage — Job Description Parser")
print("=" * 60)
print("\nPaste a job description below (press Enter twice when done):\n")

lines = []
while True:
    line = input()
    if line == "":
        if lines:
            break
    else:
        lines.append(line)

job_description = "\n".join(lines)

print("\n⏳ Analysing job posting...\n")

# Fill in the prompt template
final_prompt = prompt.invoke({
    "job_description": job_description,
    "format_instructions": parser.get_format_instructions()
})

# Call the model
response = model.invoke(final_prompt)

# Parse the JSON response into a Python object
job = parser.parse(response.content)

# -------------------- Display --------------------
print("=" * 60)
print("✅ Extracted Job Information")
print("=" * 60)
print(f"\n📌 Role:         {job.job_title}")
print(f"🏢 Company:      {job.company_name or 'Not mentioned'}")
print(f"📍 Location:     {job.location or 'Not mentioned'}")
print(f"💼 Type:         {job.job_type or 'Not mentioned'}")
print(f"📅 Experience:   {job.experience_required or 'Not mentioned'}")
print(f"💰 Salary:       {job.salary_range or 'Not mentioned'}")
print(f"\n🔧 Required Skills:")
for skill in job.required_skills:
    print(f"   • {skill}")
if job.nice_to_have_skills:
    print(f"\n⭐ Nice to Have:")
    for skill in job.nice_to_have_skills:
        print(f"   • {skill}")
print(f"\n📝 Summary: {job.summary}")
print()
