import math
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import spacy
from openai import OpenAI
from typing import Dict, Any
from dotenv import load_dotenv
import asyncio
import textstat 
import re
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

app = FastAPI(title="Resume Analyzer API with OpenAI and spaCy")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for the request and response
class AnalysisRequest(BaseModel):
    resume_text: str
    job_text: str

class ResumeAnalysis(BaseModel):
    fit_score: float
    spacy_analysis: Dict[str, Any]
    openai_feedback: str
    total_score: float

def analyze_with_spacy(resume_text: str, job_text: str) -> Dict[str, Any]:
    """
    Analyzes resume text and job description using spaCy and additional logic.
    Metrics are computed dynamically:
      - Fit score: cosine similarity between the resume and job description.
      - Word count, average sentence length.
      - Readability via Flesch Reading Ease (using textstat).
      - Keyword optimization: ratio of common keywords.
      - Structure: detection of standard resume sections.
      - Achievements: based on numeric quantifiers.
      - Professionalism: a simple heuristic based on punctuation.
      - Formatting: check for newline usage.
    All scores are scaled (mostly out of 10) and summed to a total score.
    """
    resume_doc = nlp(resume_text)
    job_doc = nlp(job_text)

    # 1. Overall Fit Score (using spaCy similarity)
    fit_score = resume_doc.similarity(job_doc) * 100

    # 2. Basic Text Metrics
    words = [token.text for token in resume_doc if token.is_alpha]
    num_words = len(words)
    sentences = list(resume_doc.sents)
    num_sentences = len(sentences)
    avg_sentence_length = num_words / num_sentences if num_sentences > 0 else num_words

    # 3. Readability: Use Flesch Reading Ease score
    flesch_score = textstat.flesch_reading_ease(resume_text)

    # 4. Keyword Optimization
    # Extract lemmas from resume and job description (ignore stop words)
    job_keywords = {token.lemma_.lower() for token in job_doc if token.is_alpha and not token.is_stop}
    resume_keywords = {token.lemma_.lower() for token in resume_doc if token.is_alpha and not token.is_stop}
    common_keywords = resume_keywords.intersection(job_keywords)
    keyword_count = len(common_keywords)
    keyword_ratio = keyword_count / len(job_keywords) if job_keywords else 0
    # Scale ratio to a 10-point score (capped at 10)
    keyword_optimization = min(keyword_ratio * 10, 10)

    # 5. Structure
    # Check for common resume sections (case-insensitive search)
    sections = ["work experience", "education", "skills", "projects", "certifications"]
    found_sections = [section for section in sections if section in resume_text.lower()]
    structure_score = (len(found_sections) / len(sections)) * 10

    # 6. Achievements
    # Count numeric values in resume as an indicator of quantifiable achievements.
    numbers = re.findall(r'\d+', resume_text)
    achievements_score = min(len(numbers) * 0.5, 10)

    # 7. Professionalism
    # Simple heuristic: fewer exclamation marks equals higher professionalism.
    exclamations = resume_text.count("!")
    professionalism_score = max(10 - exclamations, 0)

    # 8. Formatting
    # Check if the resume contains line breaks (an indicator of structured text)
    formatting_score = 10 if "\n" in resume_text else 5

    # 9. Content Quality
    # Based on word count; more words (if above a threshold) indicate richer content.
    if num_words < 200:
        content_quality = 4
    elif num_words < 400:
        content_quality = 7
    else:
        content_quality = 10

    # 10. Readability Score Scaling
    # Typical Flesch reading ease: around 60-70 is acceptable.
    if flesch_score > 80:
        readability = 10
    elif flesch_score > 70:
        readability = 8
    elif flesch_score > 60:
        readability = 6
    elif flesch_score > 50:
        readability = 4
    else:
        readability = 2

    # Total dynamic score is the sum of all categories
    total_score = (
        formatting_score +
        content_quality +
        structure_score +
        keyword_optimization +
        readability +
        achievements_score +
        professionalism_score
    )

    return {
        "fit_score": round(fit_score, 1),
        "num_words": num_words,
        "avg_sentence_length": round(avg_sentence_length, 1),
        "flesch_score": round(flesch_score, 1),
        "keyword_count": keyword_count,
        "keyword_ratio": round(keyword_ratio, 2),
        "common_keywords": list(common_keywords),
        "scores": {
            "formatting": formatting_score,
            "content_quality": content_quality,
            "structure": structure_score,
            "keyword_optimization": math.floor(keyword_optimization),
            "readability": readability,
            "achievements": achievements_score,
            "professionalism": professionalism_score
        },
        "total_score": round(total_score, 1)
    }


async def analyze_with_openai(resume_text: str, job_text: str) -> str:
    prompt = f"""
You are an expert career advisor. Analyze the following resume and job description, and provide detailed feedback.

Resume:
{resume_text}

Job Description:
{job_text}

Please provide:
1. An overall fit score (0 to 100) based on how well the resume matches the job description.
2. Specific improvement tips (e.g., missing keywords, formatting suggestions, projects to add).
Your response should be clear, actionable, and concise.
"""
    try:
        # Run the synchronous API call in a separate thread
        response = await asyncio.to_thread(
            lambda: client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful career advisor."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,
                temperature=0.7,
            )
        )
        openai_feedback = response.choices[0].message.content.strip()
        return openai_feedback
    except Exception as e:
        return f"Error calling OpenAI API: {e}"



@app.post("/analyze-resume", response_model=ResumeAnalysis)
async def analyze_resume(request: AnalysisRequest):
    if not request.resume_text or not request.job_text:
        raise HTTPException(status_code=400, detail="Both resume and job description texts are required.")

    spacy_result = analyze_with_spacy(request.resume_text, request.job_text)
    openai_feedback = await analyze_with_openai(request.resume_text, request.job_text)

    return ResumeAnalysis(
        fit_score=spacy_result["fit_score"],
        spacy_analysis=spacy_result,
        openai_feedback=openai_feedback,
        total_score=spacy_result["total_score"],
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
