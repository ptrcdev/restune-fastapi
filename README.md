# Resume Analyzer API

A FastAPI-based backend service that leverages natural language processing (NLP) with spaCy and the OpenAI API to analyze resumes against job descriptions. This service generates an overall fit score, detailed metric breakdowns, and actionable improvement suggestions to help job seekers optimize their resumes.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies](#technologies)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Deployment](#deployment)
- [Future Improvements](#future-improvements)

---

## Overview

The Resume Analyzer API processes resume and job description texts to produce an in-depth analysis. It computes several metrics using spaCy (such as word count, sentence length, readability, keyword analysis, etc.) and then uses OpenAI's GPT-3.5-turbo model to provide qualitative feedback and improvement suggestions. The API returns a structured JSON response that can be integrated with frontend applications for a seamless user experience.

---

## Features

- **Comprehensive Analysis:**  
  Evaluates resume content, formatting, structure, keyword optimization, readability, achievements, and overall professionalism.
  
- **NLP Integration:**  
  Uses spaCy to process text and compute detailed metrics.
  
- **AI Feedback:**  
  Integrates OpenAI to generate qualitative feedback and actionable improvement tips.
  
- **Flexible Input:**  
  Supports resume text sent directly as JSON and can be extended to support file uploads via integration with a separate file-processing service.
  
- **RESTful API:**  
  Exposes endpoints that can be easily consumed by frontend applications.

---

## Technologies

- **FastAPI:** High-performance API framework for Python.
- **Uvicorn:** ASGI server to run the FastAPI application.
- **spaCy:** Industrial-strength NLP library.
- **OpenAI API:** For generating detailed, qualitative feedback.
- **python-dotenv:** Manage environment variables.
- **Pydantic:** Data validation and settings management.

---

## Setup and Installation

1. **Clone the repository:**

   ```bash
   git clone git@github.com:ptrcdev/restune-fastapi.git
   cd restune-fastapi
   ```
2. **Create and activate a virtual environment:**

   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows use: env\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download spaCy Language Model:**

   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Environment Variables:**

   Create a `.env` file in the root directory and add the following variables:

   ```bash
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage
1. **Running Locally:**

   ```bash
   uvicorn main:app --reload
   ```

   The API will be available at http://localhost:8000. Visit http://localhost:8000/docs to explore the automatically generated Swagger UI documentation.

2. **Endpoint Example:**

   ```http
   POST /analyze-resume
   ```

   **Request Body (JSON):**
   ```json
   {
    "resume_text": "Your resume text here...",
    "job_text": "Job description text here..."
   }
   ```

   **Response Example:**
   ```json
   {
    "fit_score": 89.9,
    "spacy_analysis": {
        "num_words": 764,
        "avg_sentence_length": 16.6,
        "flesch_score": 14.1,
        "keyword_count": 68,
        "keyword_ratio": 0.32,
        "common_keywords": ["feature", "technical", "..."],
        "scores": {
        "formatting": 10,
        "content_quality": 10,
        "structure": 6,
        "keyword_optimization": 3.18,
        "readability": 2,
        "achievements": 10,
        "professionalism": 10
        },
        "total_score": 51.2
    },
    "openai_feedback": "**Overall Fit Score: 85/100**\n\n**Strengths:**\n1. Strong match in terms of technical skills...\n\n**Areas for Improvement:**\n1. **Missing Keywords:** Include keywords such as OCR, RPA...\n\n**Actionable Suggestions:**\n- Incorporate specific examples...",
    "total_score": 51.2
    }
    ```

## Deployment

This API was deployed on Railway. Since we are containerizing, we need to create a Dockerfile.

**Dockerfile example:**
```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm
COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Future Improvements
- **Enhanced NLP Analysis:**
  - Incorporate additional NLP models or techniques for more nuanced resume analysis.

- **File Upload Support:**
  - Expand functionality to support resume file uploads and parsing.

- **Advanced Feedback Customization:**
  - Refine OpenAI prompts for more detailed and actionable feedback.

- **CI/CD Integration:**
  - Automate testing and deployment with GitHub Actions or Railway’s built-in pipelines.

- **User Authentication:**
  - Implement authentication to allow users to save and track resume analysis history.