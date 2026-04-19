"""
============================================================
  AI-Powered Resume Scanner Web App
  Built with Python Flask — Single File
  Uses: sklearn, nltk, PyPDF2, python-docx, matplotlib
  No paid APIs. Free & open-source only.
============================================================
"""

import os
import io
import re
import json
import base64
import string
import textwrap
from collections import Counter
from datetime import datetime

# Flask
from flask import Flask, request, render_template_string, redirect, url_for, session, jsonify, send_file

# File parsing
import PyPDF2
import docx

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

# ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─────────────────────────────────────────────
#  Download NLTK data (runs once)
# ─────────────────────────────────────────────
def download_nltk_data():
    packages = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'punkt_tab']
    for pkg in packages:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

download_nltk_data()

# ─────────────────────────────────────────────
#  App Setup
# ─────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "resume-scanner-secret-2024")
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB max upload

# ─────────────────────────────────────────────
#  Skill Database (Static Knowledge Base)
# ─────────────────────────────────────────────

SKILLS_DB = {
    # Programming Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust", "swift",
    "kotlin", "ruby", "php", "r", "matlab", "scala", "perl", "dart", "lua",

    # Web Dev
    "html", "css", "react", "angular", "vue", "node.js", "nodejs", "express", "django",
    "flask", "fastapi", "spring", "laravel", "rails", "next.js", "nextjs", "gatsby",
    "graphql", "rest", "restful", "api", "bootstrap", "tailwind", "sass", "webpack",

    # Data / ML / AI
    "machine learning", "deep learning", "artificial intelligence", "ai", "ml",
    "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn", "pandas", "numpy",
    "matplotlib", "seaborn", "scipy", "opencv", "nlp", "natural language processing",
    "computer vision", "data science", "data analysis", "data engineering", "etl",
    "spark", "hadoop", "kafka", "airflow", "dbt", "tableau", "power bi", "looker",

    # Cloud / DevOps
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "k8s", "terraform",
    "ansible", "jenkins", "ci/cd", "github actions", "devops", "cloud", "microservices",
    "linux", "bash", "shell scripting", "nginx", "apache",

    # Databases
    "sql", "mysql", "postgresql", "postgres", "mongodb", "redis", "elasticsearch",
    "cassandra", "dynamodb", "sqlite", "oracle", "nosql", "firebase",

    # Tools & Practices
    "git", "github", "gitlab", "jira", "agile", "scrum", "kanban", "tdd",
    "unit testing", "pytest", "jest", "selenium", "postman", "swagger",

    # Soft Skills / Business
    "communication", "leadership", "teamwork", "problem solving", "project management",
    "time management", "critical thinking", "collaboration", "mentoring",
    "stakeholder management", "presentation", "product management",

    # Domain
    "cybersecurity", "networking", "blockchain", "iot", "embedded systems",
    "mobile development", "android", "ios", "react native", "flutter",
    "game development", "unity", "unreal", "figma", "ui/ux", "ux design",
}

COURSE_RECOMMENDATIONS = {
    "machine learning": [
        {"title": "Machine Learning by Andrew Ng", "url": "https://www.coursera.org/learn/machine-learning", "platform": "Coursera (Audit Free)"},
        {"title": "ML Crash Course by Google", "url": "https://developers.google.com/machine-learning/crash-course", "platform": "Google"},
    ],
    "deep learning": [
        {"title": "Deep Learning Specialization", "url": "https://www.coursera.org/specializations/deep-learning", "platform": "Coursera (Audit Free)"},
        {"title": "Fast.ai Deep Learning", "url": "https://course.fast.ai/", "platform": "fast.ai (Free)"},
    ],
    "python": [
        {"title": "Python for Everybody", "url": "https://www.coursera.org/specializations/python", "platform": "Coursera (Audit Free)"},
        {"title": "Python Tutorial", "url": "https://www.youtube.com/watch?v=_uQrJ0TkZlc", "platform": "YouTube - Programming with Mosh"},
    ],
    "javascript": [
        {"title": "JavaScript Full Course", "url": "https://www.youtube.com/watch?v=PkZNo7MFNFg", "platform": "YouTube - freeCodeCamp"},
        {"title": "The Odin Project", "url": "https://www.theodinproject.com/", "platform": "The Odin Project (Free)"},
    ],
    "react": [
        {"title": "React Course", "url": "https://www.youtube.com/watch?v=bMknfKXIFA8", "platform": "YouTube - freeCodeCamp"},
        {"title": "React Official Docs", "url": "https://react.dev/learn", "platform": "React Docs (Free)"},
    ],
    "data science": [
        {"title": "IBM Data Science Professional", "url": "https://www.coursera.org/professional-certificates/ibm-data-science", "platform": "Coursera (Audit Free)"},
        {"title": "Data Science Full Course", "url": "https://www.youtube.com/watch?v=ua-CiDNNj30", "platform": "YouTube - freeCodeCamp"},
    ],
    "sql": [
        {"title": "SQL Tutorial", "url": "https://www.youtube.com/watch?v=HXV3zeQKqGY", "platform": "YouTube - freeCodeCamp"},
        {"title": "SQLZoo", "url": "https://sqlzoo.net/", "platform": "SQLZoo (Free)"},
    ],
    "aws": [
        {"title": "AWS Cloud Practitioner", "url": "https://www.youtube.com/watch?v=SOTamWNgDKc", "platform": "YouTube - freeCodeCamp"},
        {"title": "AWS Free Training", "url": "https://aws.amazon.com/training/digital/", "platform": "AWS (Free Tier)"},
    ],
    "docker": [
        {"title": "Docker Tutorial for Beginners", "url": "https://www.youtube.com/watch?v=3c-iBn73dDE", "platform": "YouTube - TechWorld with Nana"},
        {"title": "Docker Getting Started", "url": "https://docs.docker.com/get-started/", "platform": "Docker Docs (Free)"},
    ],
    "kubernetes": [
        {"title": "Kubernetes Tutorial for Beginners", "url": "https://www.youtube.com/watch?v=X48VuDVv0do", "platform": "YouTube - TechWorld with Nana"},
        {"title": "Kubernetes Basics", "url": "https://kubernetes.io/docs/tutorials/kubernetes-basics/", "platform": "Kubernetes Docs (Free)"},
    ],
    "nlp": [
        {"title": "NLP with Python", "url": "https://www.youtube.com/watch?v=05ONoGfmKvA", "platform": "YouTube - Sentdex"},
        {"title": "Hugging Face NLP Course", "url": "https://huggingface.co/learn/nlp-course/chapter1/1", "platform": "Hugging Face (Free)"},
    ],
    "git": [
        {"title": "Git & GitHub Crash Course", "url": "https://www.youtube.com/watch?v=RGOj5yH7evk", "platform": "YouTube - freeCodeCamp"},
        {"title": "Learn Git Branching", "url": "https://learngitbranching.js.org/", "platform": "Interactive (Free)"},
    ],
    "linux": [
        {"title": "Linux Command Line Full Course", "url": "https://www.youtube.com/watch?v=ZtqBQ68cfJc", "platform": "YouTube - freeCodeCamp"},
        {"title": "Linux Journey", "url": "https://linuxjourney.com/", "platform": "Linux Journey (Free)"},
    ],
    "ui/ux": [
        {"title": "Google UX Design Certificate", "url": "https://www.coursera.org/professional-certificates/google-ux-design", "platform": "Coursera (Audit Free)"},
        {"title": "Figma UI Design Tutorial", "url": "https://www.youtube.com/watch?v=FTFaQWZBqQ8", "platform": "YouTube"},
    ],
    "communication": [
        {"title": "Successful Presentation", "url": "https://www.coursera.org/learn/presentation-skills", "platform": "Coursera (Audit Free)"},
    ],
    "default": [
        {"title": "freeCodeCamp", "url": "https://www.freecodecamp.org/", "platform": "freeCodeCamp (Free)"},
        {"title": "MIT OpenCourseWare", "url": "https://ocw.mit.edu/", "platform": "MIT OCW (Free)"},
        {"title": "Khan Academy", "url": "https://www.khanacademy.org/", "platform": "Khan Academy (Free)"},
    ]
}

COMPANY_DB = {
    frozenset(["python", "machine learning"]): ["Google DeepMind", "OpenAI (public research)", "Hugging Face", "Nvidia", "Databricks", "DataRobot"],
    frozenset(["python", "data science"]): ["Airbnb", "Spotify", "Netflix", "Uber", "Lyft", "Palantir"],
    frozenset(["python", "django"]): ["Instagram (Meta)", "Disqus", "Mozilla", "Bitbucket", "Pinterest"],
    frozenset(["javascript", "react"]): ["Meta", "Airbnb", "Atlassian", "Shopify", "Twitter/X", "Vercel", "Netlify"],
    frozenset(["javascript", "node.js"]): ["LinkedIn", "PayPal", "Walmart Labs", "NASA (JPL)"],
    frozenset(["java", "spring"]): ["Amazon", "JPMorgan Chase", "Goldman Sachs", "SAP", "Salesforce"],
    frozenset(["aws", "docker"]): ["Amazon AWS", "Twilio", "HashiCorp", "CircleCI", "GitLab"],
    frozenset(["sql", "data analysis"]): ["Deloitte", "Accenture", "McKinsey Analytics", "KPMG", "IBM"],
    frozenset(["cybersecurity"]): ["CrowdStrike", "Palo Alto Networks", "Cisco", "Fortinet", "FireEye"],
    frozenset(["android", "kotlin"]): ["Samsung", "Google Android Team", "Grab", "Gojek"],
    frozenset(["ios", "swift"]): ["Apple", "Uber", "Airbnb iOS", "Robinhood"],
}

RESUME_SECTIONS = {
    "contact": ["email", "phone", "linkedin", "github", "address", "portfolio", "website"],
    "summary": ["summary", "objective", "profile", "about me", "professional summary", "career objective"],
    "skills": ["skills", "technical skills", "core competencies", "expertise", "technologies", "tools"],
    "experience": ["experience", "work experience", "employment", "professional experience", "career history", "work history"],
    "education": ["education", "academic", "qualification", "degree", "university", "college", "schooling"],
    "projects": ["projects", "personal projects", "academic projects", "portfolio"],
    "certifications": ["certifications", "certificates", "credentials", "licenses", "accreditation"],
    "achievements": ["achievements", "awards", "honors", "accomplishments", "recognition"],
}

ACTION_VERBS = [
    "achieved", "accelerated", "accomplished", "analyzed", "architected", "automated",
    "built", "collaborated", "created", "delivered", "designed", "developed", "drove",
    "engineered", "enhanced", "executed", "generated", "implemented", "improved",
    "increased", "launched", "led", "managed", "mentored", "optimized", "pioneered",
    "reduced", "resolved", "spearheaded", "streamlined", "transformed",
]

CHATBOT_KNOWLEDGE = {
    "improve resume": (
        "Here are key ways to improve your resume:\n"
        "1. Use strong action verbs (built, developed, achieved, led)\n"
        "2. Add quantifiable metrics (increased sales by 30%, reduced load time by 40%)\n"
        "3. Tailor your resume for each job — match keywords from the job description\n"
        "4. Keep it to 1-2 pages; use clean formatting with clear sections\n"
        "5. Include: Summary, Skills, Experience, Education, Projects, Certifications\n"
        "6. Use ATS-friendly fonts (Arial, Calibri) and avoid tables/graphics\n"
        "7. Put most recent experience first (reverse chronological order)"
    ),
    "skills": (
        "Popular skills employers look for in 2024-2025:\n"
        "• Tech: Python, JavaScript/TypeScript, React, Node.js, SQL, AWS/GCP, Docker, Kubernetes\n"
        "• Data: Machine Learning, Data Analysis, SQL, Tableau, Power BI, Pandas\n"
        "• Soft: Communication, Leadership, Problem-solving, Agile/Scrum\n"
        "• Trending: Generative AI, LLMs, MLOps, Cybersecurity, Cloud Native\n"
        "Upload your resume and a job description to get personalized skill gap analysis!"
    ),
    "companies": (
        "Top companies for tech professionals:\n"
        "• Big Tech: Google, Microsoft, Amazon, Meta, Apple, Netflix\n"
        "• Startups: Vercel, Hugging Face, Databricks, Figma, Notion\n"
        "• Finance/Fintech: JPMorgan, Goldman Sachs, Stripe, Plaid\n"
        "• Remote-first: GitLab, Automattic, Basecamp, InVision\n"
        "Tip: Upload your resume to get company suggestions based on YOUR skills!"
    ),
    "ats": (
        "ATS (Applicant Tracking System) Tips:\n"
        "1. Use standard section headings (Experience, Education, Skills)\n"
        "2. Avoid tables, columns, headers/footers — ATS can't parse them\n"
        "3. Use keywords exactly as written in the job description\n"
        "4. Save/send as .docx or .pdf (with text, not scanned images)\n"
        "5. Avoid fancy fonts, graphics, or special characters\n"
        "6. Aim for 60%+ keyword match with the job description"
    ),
    "job search": (
        "Job Search Strategy:\n"
        "1. LinkedIn — optimize your profile, turn on 'Open to Work'\n"
        "2. Indeed, Glassdoor, Naukri (India), AngelList (startups)\n"
        "3. Company career pages — apply directly\n"
        "4. GitHub/Portfolio — projects get you noticed\n"
        "5. Networking: 80 of jobs are filled through connections\n"
        "6. Apply to 5-10 targeted jobs/day rather than mass applying"
    ),
    "interview": (
        "Interview Preparation Tips:\n"
        "1. STAR method: Situation, Task, Action, Result for behavioral questions\n"
        "2. Research the company's products, culture, and recent news\n"
        "3. Practice coding problems: LeetCode, HackerRank, CodeSignal\n"
        "4. Prepare 5 questions to ask the interviewer\n"
        "5. Review your resume — they WILL ask about everything on it\n"
        "6. Mock interviews: Pramp, Interviewing.io (free options available)"
    ),
    "salary": (
        "Salary Negotiation Tips:\n"
        "1. Research market rates: Glassdoor, Levels.fyi, PayScale\n"
        "2. Never give a number first — ask about their budget\n"
        "3. Consider total compensation: base, bonus, equity, benefits\n"
        "4. Always negotiate — most companies expect it\n"
        "5. Get the offer in writing before resigning from current job"
    ),
}
# ─────────────────────────────────────────────
#  Text Extraction Functions
# ─────────────────────────────────────────────

def extract_text_from_pdf(file_bytes):
    """Extract text from a PDF file."""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text.strip()
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def extract_text_from_docx(file_bytes):
    """Extract text from a DOCX file."""
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        text = "\n".join([para.text for para in doc.paragraphs])
        return text.strip()
    except Exception as e:
        return f"Error reading DOCX: {str(e)}"

def extract_text_from_txt(file_bytes):
    """Extract text from a plain text file."""
    try:
        return file_bytes.decode('utf-8', errors='ignore').strip()
    except Exception as e:
        return f"Error reading TXT: {str(e)}"

def extract_resume_text(file):
    """Route file to appropriate parser based on extension."""
    filename = file.filename.lower()
    file_bytes = file.read()
    if filename.endswith('.pdf'):
        return extract_text_from_pdf(file_bytes)
    elif filename.endswith('.docx'):
        return extract_text_from_docx(file_bytes)
    elif filename.endswith('.txt'):
        return extract_text_from_txt(file_bytes)
    else:
        return "Unsupported file format. Please upload PDF, DOCX, or TXT."

# ─────────────────────────────────────────────
#  NLP Helper Functions
# ─────────────────────────────────────────────

def clean_text(text):
    """Lowercase, remove special chars, normalize whitespace."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tokenize_and_filter(text):
    """Tokenize and remove stopwords."""
    try:
        stop_words = set(stopwords.words('english'))
    except Exception:
        stop_words = set()
    tokens = word_tokenize(clean_text(text))
    return [t for t in tokens if t.isalpha() and t not in stop_words and len(t) > 2]

def extract_skills_from_text(text):
    """Extract known skills from text using substring and phrase matching."""
    text_lower = text.lower()
    found = set()
    for skill in SKILLS_DB:
        # Match as a whole word / phrase
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found.add(skill)
    return found

def extract_email(text):
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    matches = re.findall(pattern, text)
    return matches[0] if matches else None

def extract_phone(text):
    pattern = r'(?:\+?\d{1,3}[\s\-]?)?(?:\(?\d{2,4}\)?[\s\-]?)?\d{3,4}[\s\-]?\d{4}'
    matches = re.findall(pattern, text)
    return matches[0] if matches else None

def extract_name(text):
    """Heuristic: first non-empty line is often the candidate's name."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if lines:
        first_line = lines[0]
        # If it looks like a name (no @, no numbers, reasonable length)
        if len(first_line) < 50 and '@' not in first_line and not re.search(r'\d{5,}', first_line):
            return first_line
    return "Not detected"

# ─────────────────────────────────────────────
#  ATS Score Calculation
# ─────────────────────────────────────────────

def calculate_ats_score(resume_text):
    """
    Score resume from 0-100 based on:
    - Section presence (30 pts)
    - Keyword density (20 pts)
    - Contact info (15 pts)
    - Action verbs (15 pts)
    - Formatting signals (10 pts)
    - Length appropriateness (10 pts)
    """
    score = 0
    details = {}
    text_lower = resume_text.lower()

    # 1. Section Presence (30 pts)
    section_score = 0
    found_sections = {}
    for section, keywords in RESUME_SECTIONS.items():
        found = any(kw in text_lower for kw in keywords)
        found_sections[section] = found
        if found:
            section_score += 30 // len(RESUME_SECTIONS)
    section_score = min(30, section_score)
    score += section_score
    details['sections'] = {
        'score': section_score,
        'max': 30,
        'found': [s for s, f in found_sections.items() if f],
        'missing': [s for s, f in found_sections.items() if not f]
    }

    # 2. Technical Keywords (20 pts)
    found_skills = extract_skills_from_text(resume_text)
    skill_count = len(found_skills)
    keyword_score = min(20, skill_count * 2)
    score += keyword_score
    details['keywords'] = {
        'score': keyword_score,
        'max': 20,
        'count': skill_count,
        'found': list(found_skills)[:10]
    }

    # 3. Contact Info (15 pts)
    contact_score = 0
    email = extract_email(resume_text)
    phone = extract_phone(resume_text)
    has_linkedin = 'linkedin' in text_lower
    if email:
        contact_score += 6
    if phone:
        contact_score += 5
    if has_linkedin:
        contact_score += 4
    score += contact_score
    details['contact'] = {
        'score': contact_score,
        'max': 15,
        'email': email,
        'phone': phone,
        'linkedin': has_linkedin
    }
    tokens = text_lower.split()
    found_verbs = [v for v in ACTION_VERBS if v in tokens]
    verb_score = min(15, len(found_verbs) * 2)
    score += verb_score
    details['action_verbs'] = {
        'score': verb_score,
        'max': 15,
        'found': found_verbs[:8],
        'missing_examples': [v for v in ACTION_VERBS[:8] if v not in found_verbs][:5]
    }

    # 5. Formatting Signals (10 pts)
    format_score = 0
    word_count = len(resume_text.split())
    # Bullet points
    if re.search(r'[•·▪▸\-\*]', resume_text):
        format_score += 4
    # Dates (experience dating)
    if re.search(r'\b(19|20)\d{2}\b', resume_text):
        format_score += 3
    # Numbers/metrics
    if re.search(r'\d+[%\+x]', resume_text) or re.search(r'\$\d+', resume_text):
        format_score += 3
    score += format_score
    details['formatting'] = {
        'score': format_score,
        'max': 10,
        'has_bullets': bool(re.search(r'[•·▪▸\-\*]', resume_text)),
        'has_dates': bool(re.search(r'\b(19|20)\d{2}\b', resume_text)),
        'has_metrics': bool(re.search(r'\d+[%\+x]|\$\d+', resume_text))
    }

    # 6. Length (10 pts)
    length_score = 0
    word_count = len(resume_text.split())
    if 300 <= word_count <= 800:
        length_score = 10
    elif 200 <= word_count < 300 or 800 < word_count <= 1200:
        length_score = 7
    elif 100 <= word_count < 200 or 1200 < word_count <= 1500:
        length_score = 4
    else:
        length_score = 1
    score += length_score
    details['length'] = {
        'score': length_score,
        'max': 10,
        'word_count': word_count,
        'recommendation': (
            "Ideal (300-800 words)" if 300 <= word_count <= 800 else
            "Too short — add more detail" if word_count < 300 else
            "Too long — consider trimming"
        )
    }

    return min(100, score), details

# ─────────────────────────────────────────────
#  JD Matching
# 
def calculate_jd_match(resume_text, jd_text):
    """Use TF-IDF + cosine similarity to calculate job description match."""
    try:
        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=5000,
        )
        tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        match_percent = round(similarity * 100, 1)
    except Exception:
        match_percent = 0.0

    resume_skills = extract_skills_from_text(resume_text)
    jd_skills = extract_skills_from_text(jd_text)

    matched_skills = resume_skills & jd_skills
    missing_skills = jd_skills - resume_skills

    return {
        'match_percent': match_percent,
        'matched_skills': sorted(matched_skills),
        'missing_skills': sorted(missing_skills),
        'jd_skills_count': len(jd_skills),
        'resume_skills_count': len(resume_skills),
    }

# ─────────────────────────────────────────────
#  Course Recommendations
# ─────────────────────────────────────────────

def get_course_recommendations(missing_skills):
    """Map missing skills to free course recommendations."""
    recommendations = []
    added_skills = set()

    for skill in missing_skills:
        skill_lower = skill.lower()
        # Try exact match first
        if skill_lower in COURSE_RECOMMENDATIONS:
            for course in COURSE_RECOMMENDATIONS[skill_lower]:
                recommendations.append({'skill': skill, **course})
            added_skills.add(skill_lower)
        else:
            # Fuzzy match — find partial overlaps
            for key in COURSE_RECOMMENDATIONS:
                if key in skill_lower or skill_lower in key:
                    for course in COURSE_RECOMMENDATIONS[key]:
                        recommendations.append({'skill': skill, **course})
                    added_skills.add(skill_lower)
                    break

    # Add general recommendations if fewer than 3
    if len(recommendations) < 3:
        for course in COURSE_RECOMMENDATIONS['default']:
            recommendations.append({'skill': 'General Learning', **course})

    return recommendations[:12]  # Cap at 12

# ─────────────────────────────────────────────
#  Company Suggestions
# ─────────────────────────────────────────────

def suggest_companies(resume_skills):
    """Suggest companies based on detected skills."""
    suggestions = []
    skills_lower = {s.lower() for s in resume_skills}

    for skill_set, companies in COMPANY_DB.items():
        if skill_set & skills_lower:
            overlap = skill_set & skills_lower
            suggestions.append({
                'skills': list(overlap),
                'companies': companies
            })

    # Deduplicate company names
    seen = set()
    unique = []
    for entry in suggestions:
        for company in entry['companies']:
            if company not in seen:
                seen.add(company)
                unique.append({'company': company, 'skills': entry['skills']})

    if not unique:
        # Generic suggestions
        unique = [
            {'company': 'Tech startups (AngelList)', 'skills': list(resume_skills)[:3]},
            {'company': 'Consulting firms (Accenture, Infosys)', 'skills': list(resume_skills)[:3]},
            {'company': 'Remote-friendly companies (Remote.com)', 'skills': list(resume_skills)[:3]},
        ]

    return unique[:12]

# ─────────────────────────────────────────────
#  AI Resume Suggestions
# ─────────────────────────────────────────────

def generate_resume_suggestions(resume_text, ats_details, missing_skills=None):
    """Generate actionable resume improvement suggestions."""
    suggestions = []
    text_lower = resume_text.lower()

    # Action verbs
    missing_verbs = ats_details['action_verbs']['missing_examples']
    if missing_verbs:
        suggestions.append({
            'category': 'Action Verbs',
            'priority': 'High',
            'suggestion': f"Add strong action verbs to describe your experience. Try: {', '.join(missing_verbs)}",
            'example': 'Instead of "Responsible for managing team" → "Led a cross-functional team of 5 engineers"'
        })

    # Metrics
    if not ats_details['formatting']['has_metrics']:
        suggestions.append({
            'category': 'Quantifiable Impact',
            'priority': 'High',
            'suggestion': "Add numbers and metrics to demonstrate impact.",
            'example': '"Improved performance" → "Improved page load speed by 40%, reducing bounce rate by 25%"'
        })

    # Missing sections
    missing_secs = ats_details['sections']['missing']
    if 'summary' in missing_secs:
        suggestions.append({
            'category': 'Professional Summary',
            'priority': 'High',
            'suggestion': "Add a 2-3 sentence professional summary at the top of your resume.",
            'example': '"Results-driven Software Engineer with 3+ years experience in Python and cloud technologies, specializing in scalable backend systems."'
        })
    if 'projects' in missing_secs:
        suggestions.append({
            'category': 'Projects Section',
            'priority': 'Medium',
            'suggestion': "Add a Projects section to showcase hands-on work.",
            'example': 'Include project name, tech stack, your role, and key outcomes.'
        })
    if 'certifications' in missing_secs:
        suggestions.append({
            'category': 'Certifications',
            'priority': 'Low',
            'suggestion': "Add relevant certifications (AWS, Google Cloud, Coursera, etc.) to boost credibility.",
            'example': 'AWS Certified Solutions Architect | Coursera Machine Learning | Google Data Analytics'
        })

    # Contact info
    if not ats_details['contact']['email']:
        suggestions.append({
            'category': 'Contact Info',
            'priority': 'Critical',
            'suggestion': "Add your email address — this is essential for recruiters to reach you.",
            'example': 'yourname@email.com'
        })
    if not ats_details['contact']['linkedin']:
        suggestions.append({
            'category': 'LinkedIn Profile',
            'priority': 'Medium',
            'suggestion': "Add your LinkedIn profile URL.",
            'example': 'linkedin.com/in/yourname'
        })

    # Length
    word_count = ats_details['length']['word_count']
    if word_count < 300:
        suggestions.append({
            'category': 'Resume Length',
            'priority': 'High',
            'suggestion': f"Your resume is too short ({word_count} words). Add more detail to your experience and projects.",
            'example': 'Expand bullet points with context, impact, and technologies used.'
        })
    elif word_count > 1200:
        suggestions.append({
            'category': 'Resume Length',
            'priority': 'Medium',
            'suggestion': f"Your resume is long ({word_count} words). Consider trimming older or less relevant experience.",
            'example': 'Keep only the most recent 10 years of experience, and use concise bullet points.'
        })

    # Skills
    if missing_skills:
        top_missing = list(missing_skills)[:5]
        suggestions.append({
            'category': 'Missing Skills',
            'priority': 'High',
            'suggestion': f"Add these in-demand skills to your resume: {', '.join(top_missing)}",
            'example': 'List in a dedicated Skills section, and demonstrate usage in project/experience bullet points.'
        })

    # Formatting
    if not ats_details['formatting']['has_bullets']:
        suggestions.append({
            'category': 'Formatting',
            'priority': 'Medium',
            'suggestion': "Use bullet points in experience sections for better readability.",
            'example': '• Developed REST API using Python/FastAPI, reducing response time by 30%'
        })

    # Generic tip always included
    suggestions.append({
        'category': 'ATS Optimization',
        'priority': 'Medium',
        'suggestion': "Customize your resume for each job application using keywords from the job description.",
        'example': 'If the JD says "CI/CD pipelines", use that exact phrase in your resume.'
    })

    return suggestions

# ─────────────────────────────────────────────
#  Resume Builder
# ─────────────────────────────────────────────

def build_ats_resume(resume_text, ats_details, suggestions):
    """Generate an ATS-friendly plain text resume template."""
    name = extract_name(resume_text)
    email = ats_details['contact'].get('email') or 'your.email@example.com'
    phone = ats_details['contact'].get('phone') or '+1 (555) 000-0000'
    skills = ats_details['keywords'].get('found', [])
    word_count = ats_details['length']['word_count']

    # Extract experience and education blocks heuristically
    lines = resume_text.split('\n')
    exp_lines = []
    edu_lines = []
    current_section = None
    for line in lines:
        l = line.strip()
        if not l:
            continue
        ll = l.lower()
        if any(kw in ll for kw in ['experience', 'employment', 'work history']):
            current_section = 'exp'
        elif any(kw in ll for kw in ['education', 'academic', 'degree']):
            current_section = 'edu'
        elif any(kw in ll for kw in ['skills', 'summary', 'projects', 'certifications']):
            current_section = None
        elif current_section == 'exp' and len(exp_lines) < 8:
            exp_lines.append(l)
        elif current_section == 'edu' and len(edu_lines) < 5:
            edu_lines.append(l)

    resume = f"""
{'='*65}
{name.upper()}
{'='*65}
{email} | {phone} | LinkedIn: linkedin.com/in/{name.lower().replace(' ', '-')}

{'─'*65}
PROFESSIONAL SUMMARY
{'─'*65}
Results-driven professional with expertise in {', '.join(skills[:3]) if skills else 'technology and problem-solving'}.
Proven track record of delivering high-quality solutions. Passionate about continuous learning and innovation.

{'─'*65}
SKILLS
{'─'*65}
{' | '.join(s.title() for s in skills[:12]) if skills else 'Add your technical and soft skills here'}

{'─'*65}
PROFESSIONAL EXPERIENCE
{'─'*65}
"""

    if exp_lines:
        for l in exp_lines[:6]:
            if l and not l.lower() in ['experience', 'work experience', 'employment']:
                resume += f"• {l}\n"
    else:
        resume += """[Company Name] — [Job Title]                     [Start Date] – [End Date]
• Developed and maintained [product/feature] using [technology], improving [metric] by X%
• Collaborated with cross-functional teams to deliver [outcome] on time and within budget
• Led initiative to [action], resulting in [quantifiable result]

"""

    resume += f"""
{'─'*65}
EDUCATION
{'─'*65}
"""
    if edu_lines:
        for l in edu_lines[:4]:
            if l and not l.lower() in ['education', 'academic background']:
                resume += f"• {l}\n"
    else:
        resume += """[Degree Name] in [Field of Study]
[University Name], [City, Country]                          [Year of Graduation]
CGPA/GPA: X.X/10.0 or X.X/4.0
"""

    resume += f"""
{'─'*65}
PROJECTS
{'─'*65}
[Project Name] | [Tech Stack]                               [Year]
• [Brief description of what it does and why it matters]
• [Key achievement or metric]
• GitHub: github.com/yourusername/project-name

{'─'*65}
CERTIFICATIONS
{'─'*65}
• [Certification Name] — [Issuing Organization], [Year]
• [e.g., AWS Certified Solutions Architect | Google Data Analytics | Coursera ML]

{'─'*65}
  ATS Resume generated by AI Resume Scanner | {datetime.now().strftime('%B %Y')}
  Original word count: {word_count} words | ATS Score based on original resume
{'─'*65}
""".strip()

    return resume

# ─────────────────────────────────────────────
#  Chart Generation
# ─────────────────────────────────────────────

def generate_ats_chart(ats_score, ats_details):
    """Generate ATS score bar chart as a base64 image."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#0f172a')

    # --- Left: Overall Score Gauge ---
    ax1 = axes[0]
    ax1.set_facecolor('#1e293b')
    score = ats_score
    color = '#22c55e' if score >= 70 else '#f59e0b' if score >= 50 else '#ef4444'
    ax1.barh(['ATS Score'], [score], color=color, height=0.4)
    ax1.barh(['ATS Score'], [100 - score], left=[score], color='#334155', height=0.4)
    ax1.set_xlim(0, 100)
    ax1.set_xlabel('Score', color='#94a3b8', fontsize=10)
    ax1.set_title(f'Overall ATS Score: {score}/100', color='#f1f5f9', fontsize=13, fontweight='bold', pad=12)
    ax1.tick_params(colors='#94a3b8')
    ax1.spines[:].set_color('#334155')
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_color('#94a3b8')
    ax1.text(score / 2, 0, f'{score}', ha='center', va='center', color='white', fontsize=13, fontweight='bold')

    # --- Right: Category Breakdown ---
    ax2 = axes[1]
    ax2.set_facecolor('#1e293b')
    categories = ['Sections', 'Keywords', 'Contact', 'Action Verbs', 'Formatting', 'Length']
    scores = [
        ats_details['sections']['score'],
        ats_details['keywords']['score'],
        ats_details['contact']['score'],
        ats_details['action_verbs']['score'],
        ats_details['formatting']['score'],
        ats_details['length']['score'],
    ]
    maxes = [30, 20, 15, 15, 10, 10]
    pcts = [round(s / m * 100) for s, m in zip(scores, maxes)]
    bar_colors = ['#22c55e' if p >= 70 else '#f59e0b' if p >= 40 else '#ef4444' for p in pcts]
    bars = ax2.barh(categories, scores, color=bar_colors, height=0.5)
    bg_bars = ax2.barh(categories, [m - s for s, m in zip(scores, maxes)],
                       left=scores, color='#334155', height=0.5)
    ax2.set_xlim(0, 30)
    ax2.set_title('Category Breakdown', color='#f1f5f9', fontsize=13, fontweight='bold', pad=12)
    ax2.tick_params(colors='#94a3b8')
    ax2.spines[:].set_color('#334155')
    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
        label.set_color('#94a3b8')
    for bar, score, max_s in zip(bars, scores, maxes):
        ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                 f'{score}/{max_s}', va='center', color='#94a3b8', fontsize=8)

    plt.tight_layout(pad=1.5)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='#0f172a')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def generate_skill_chart(matched_skills, missing_skills, resume_skills):
    """Generate a skill match pie/bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#0f172a')

    # --- Left: Pie Chart ---
    ax1 = axes[0]
    ax1.set_facecolor('#0f172a')
    matched_count = len(matched_skills)
    missing_count = len(missing_skills)
    resume_only = max(0, len(resume_skills) - matched_count)
    sizes = [matched_count, missing_count, resume_only]
    labels = ['Matched Skills', 'Missing Skills', 'Resume-only Skills']
    colors = ['#22c55e', '#ef4444', '#3b82f6']
    non_zero = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
    if non_zero:
        s, l, c = zip(*non_zero)
        wedges, texts, autotexts = ax1.pie(s, labels=l, colors=c, autopct='%1.0f%%',
                                            textprops={'color': '#94a3b8', 'fontsize': 9},
                                            startangle=90)
        for at in autotexts:
            at.set_color('white')
            at.set_fontsize(9)
    ax1.set_title('Skill Distribution', color='#f1f5f9', fontsize=13, fontweight='bold')

    # --- Right: Top Matched vs Missing bar ---
    ax2 = axes[1]
    ax2.set_facecolor('#1e293b')
    top_matched = list(matched_skills)[:7]
    top_missing = list(missing_skills)[:7]
    all_skills = top_matched + top_missing
    bar_colors = ['#22c55e'] * len(top_matched) + ['#ef4444'] * len(top_missing)
    y = range(len(all_skills))
    ax2.barh(list(y), [1] * len(all_skills), color=bar_colors, height=0.6)
    ax2.set_yticks(list(y))
    ax2.set_yticklabels([s[:20] for s in all_skills], color='#94a3b8', fontsize=8)
    ax2.set_xticks([])
    ax2.spines[:].set_color('#334155')
    ax2.tick_params(colors='#94a3b8')
    ax2.set_title('Matched (green) vs Missing (red)', color='#f1f5f9', fontsize=11, fontweight='bold')

    plt.tight_layout(pad=1.5)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='#0f172a')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# ─────────────────────────────────────────────
#  TF-IDF Chatbot
# ─────────────────────────────────────────────
def chatbot_response(user_message, resume_text="", jd_text=""):
    """Rule-based + TF-IDF chatbot for resume Q&A."""
    user_lower = user_message.lower().strip()

    # Direct keyword matching first
    response = None
    if any(k in user_lower for k in ['improve', 'better', 'fix', 'enhance', 'tips']):
        response = CHATBOT_KNOWLEDGE['improve resume']
    elif any(k in user_lower for k in ['skill', 'learn', 'technology', 'tech', 'language']):
        response = CHATBOT_KNOWLEDGE['skills']
    elif any(k in user_lower for k in ['compan', 'employer', 'firm', 'apply', 'organization']):
        response = CHATBOT_KNOWLEDGE['companies']
    elif any(k in user_lower for k in ['ats', 'applicant tracking', 'keyword', 'parse']):
        response = CHATBOT_KNOWLEDGE['ats']
    elif any(k in user_lower for k in ['job search', 'find job', 'where to apply', 'job board', 'linkedin']):
        response = CHATBOT_KNOWLEDGE['job search']
    elif any(k in user_lower for k in ['interview', 'prepare', 'question', 'behavioral', 'coding interview']):
        response = CHATBOT_KNOWLEDGE['interview']
    elif any(k in user_lower for k in ['salary', 'pay', 'negotiate', 'compensation', 'ctc']):
        response = CHATBOT_KNOWLEDGE['salary']

    # Resume-aware responses
    if resume_text and not response:
        resume_skills = extract_skills_from_text(resume_text)
        if resume_skills:
            skill_list = ', '.join(list(resume_skills)[:6])
            response = (
                f"Based on your resume, you have skills in: {skill_list}.\n\n"
                "Try asking me:\n"
                "• 'How to improve my resume?'\n"
                "• 'What skills should I learn?'\n"
                "• 'Which companies should I apply to?'\n"
                "• 'How to prepare for interviews?'"
            )

    if not response:
        # TF-IDF similarity against knowledge base
        try:
            knowledge_items = list(CHATBOT_KNOWLEDGE.items())
            knowledge_texts = [v for k, v in knowledge_items]
            all_texts = knowledge_texts + [user_message]
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf = vectorizer.fit_transform(all_texts)
            sims = cosine_similarity(tfidf[-1], tfidf[:-1])[0]
            best_idx = int(np.argmax(sims))
            if sims[best_idx] > 0.05:
                response = knowledge_items[best_idx][1]
        except Exception:
            pass

    if not response:
        response = (
            "I'm here to help with your job search! Try asking:\n"
            "• 'How to improve my resume?'\n"
            "• 'What skills to learn for data science?'\n"
            "• 'Which companies should I apply to?'\n"
            "• 'How to prepare for interviews?'\n"
            "• 'What is ATS and how to optimize?'\n\n"
            "Upload your resume for personalized advice!"
        )

    return response

# ─────────────────────────────────────────────
#  HTML Templates
# ─────────────────────────────────────────────

BASE_STYLE = """
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', Arial, sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; }
  .navbar { background: #1e293b; border-bottom: 1px solid #334155; padding: 14px 24px; display: flex; align-items: center; justify-content: space-between; }
  .navbar .brand { font-size: 1.2rem; font-weight: 700; color: #6366f1; display: flex; align-items: center; gap: 8px; }
  .navbar nav a { color: #94a3b8; text-decoration: none; margin-left: 20px; font-size: 0.9rem; transition: color 0.2s; }
  .navbar nav a:hover { color: #e2e8f0; }
  .container { max-width: 1100px; margin: 0 auto; padding: 32px 20px; }
  .card { background: #1e293b; border-radius: 12px; padding: 28px; margin-bottom: 24px; border: 1px solid #334155; }
  .card h2 { color: #f1f5f9; font-size: 1.2rem; margin-bottom: 16px; display: flex; align-items: center; gap: 8px; }
  .badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; margin: 3px; }
  .badge-green { background: #166534; color: #4ade80; }
  .badge-red { background: #7f1d1d; color: #fca5a5; }
  .badge-blue { background: #1e3a5f; color: #93c5fd; }
  .badge-yellow { background: #78350f; color: #fcd34d; }
  .badge-purple { background: #3b1f6e; color: #c4b5fd; }
  .score-big { font-size: 3rem; font-weight: 800; text-align: center; padding: 12px; }
  .score-green { color: #22c55e; }
  .score-yellow { color: #f59e0b; }
  .score-red { color: #ef4444; }
  .progress-bar { background: #334155; border-radius: 8px; height: 10px; overflow: hidden; margin: 8px 0; }
  .progress-fill { height: 100%; border-radius: 8px; transition: width 0.5s; }
  .btn { display: inline-block; padding: 10px 24px; border-radius: 8px; border: none; cursor: pointer; font-size: 0.95rem; font-weight: 600; text-decoration: none; transition: all 0.2s; }
  .btn-primary { background: #6366f1; color: white; }
  .btn-primary:hover { background: #4f46e5; }
  .btn-secondary { background: #334155; color: #e2e8f0; }
  .btn-secondary:hover { background: #475569; }
  .btn-success { background: #16a34a; color: white; }
  .btn-success:hover { background: #15803d; }
  .form-group { margin-bottom: 18px; }
  .form-group label { display: block; color: #94a3b8; font-size: 0.9rem; margin-bottom: 6px; font-weight: 500; }
  input[type=file], textarea, input[type=text] { width: 100%; background: #0f172a; border: 1px solid #334155; border-radius: 8px; color: #e2e8f0; padding: 10px 14px; font-size: 0.95rem; outline: none; transition: border 0.2s; }
  input[type=file]:focus, textarea:focus, input[type=text]:focus { border-color: #6366f1; }
  textarea { resize: vertical; min-height: 130px; }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
  .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; }
  .alert { padding: 12px 16px; border-radius: 8px; margin-bottom: 16px; }
  .alert-info { background: #1e3a5f; border: 1px solid #3b82f6; color: #93c5fd; }
  .alert-success { background: #14532d; border: 1px solid #22c55e; color: #4ade80; }
  .alert-warning { background: #431407; border: 1px solid #f97316; color: #fdba74; }
  .suggestion { padding: 14px 16px; border-radius: 8px; border-left: 4px solid; margin-bottom: 12px; background: #1e293b; }
  .suggestion.critical { border-color: #ef4444; }
  .suggestion.high { border-color: #f97316; }
  .suggestion.medium { border-color: #f59e0b; }
  .suggestion.low { border-color: #22c55e; }
  .suggestion .category { font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; color: #94a3b8; }
  .suggestion .text { color: #e2e8f0; font-size: 0.95rem; margin-bottom: 4px; }
  .suggestion .example { color: #64748b; font-size: 0.85rem; font-style: italic; }
  .course-card { background: #0f172a; border-radius: 8px; padding: 14px; border: 1px solid #334155; }
  .course-card .skill-tag { color: #6366f1; font-size: 0.8rem; font-weight: 600; margin-bottom: 4px; }
  .course-card .title { color: #e2e8f0; font-size: 0.95rem; font-weight: 600; margin-bottom: 4px; }
  .course-card .platform { color: #64748b; font-size: 0.8rem; }
  .course-card a { color: #6366f1; text-decoration: none; font-size: 0.85rem; }
  .course-card a:hover { text-decoration: underline; }
  .company-item { padding: 10px 14px; background: #0f172a; border-radius: 8px; border: 1px solid #334155; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; }
  .chat-messages { height: 340px; overflow-y: auto; padding: 12px; background: #0f172a; border-radius: 8px; border: 1px solid #334155; margin-bottom: 12px; }
  .msg { margin-bottom: 12px; }
  .msg-user { text-align: right; }
  .msg-bubble { display: inline-block; padding: 10px 14px; border-radius: 12px; max-width: 80%; font-size: 0.9rem; line-height: 1.5; white-space: pre-wrap; text-align: left; }
  .msg-user .msg-bubble { background: #6366f1; color: white; border-radius: 12px 12px 2px 12px; }
  .msg-bot .msg-bubble { background: #1e293b; color: #e2e8f0; border-radius: 12px 12px 12px 2px; border: 1px solid #334155; }
  .chat-input-row { display: flex; gap: 8px; }
  .chat-input-row input { flex: 1; }
  .section-header { color: #6366f1; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px; padding-bottom: 6px; border-bottom: 1px solid #334155; }
  .tab-nav { display: flex; gap: 4px; margin-bottom: 24px; background: #1e293b; padding: 6px; border-radius: 10px; }
  .tab-nav a { flex: 1; text-align: center; padding: 9px; border-radius: 7px; color: #94a3b8; text-decoration: none; font-size: 0.85rem; font-weight: 500; transition: all 0.2s; }
  .tab-nav a.active, .tab-nav a:hover { background: #6366f1; color: white; }
  .hero { text-align: center; padding: 60px 20px 40px; }
  .hero h1 { font-size: 2.5rem; font-weight: 800; color: #f1f5f9; line-height: 1.2; margin-bottom: 12px; }
  .hero h1 span { color: #6366f1; }
  .hero p { color: #94a3b8; font-size: 1.05rem; max-width: 560px; margin: 0 auto 28px; }
  .features-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 40px; }
  .feature-item { background: #1e293b; border-radius: 10px; padding: 20px; text-align: center; border: 1px solid #334155; }
  .feature-item .icon { font-size: 2rem; margin-bottom: 8px; }
  .feature-item h3 { color: #f1f5f9; font-size: 0.95rem; margin-bottom: 6px; }
  .feature-item p { color: #64748b; font-size: 0.82rem; }
  .upload-zone { border: 2px dashed #334155; border-radius: 12px; padding: 40px 24px; text-align: center; transition: all 0.2s; cursor: pointer; }
  .upload-zone:hover { border-color: #6366f1; background: rgba(99,102,241,0.05); }
  .upload-icon { font-size: 2.5rem; margin-bottom: 12px; }
  .stat-box { background: #0f172a; border-radius: 8px; padding: 16px; text-align: center; border: 1px solid #334155; }
  .stat-box .num { font-size: 2rem; font-weight: 800; color: #6366f1; }
  .stat-box .label { color: #64748b; font-size: 0.8rem; margin-top: 4px; }
  @media (max-width: 700px) { .grid-2, .grid-3 { grid-template-columns: 1fr; } .hero h1 { font-size: 1.8rem; } }
  .spinner { display: inline-block; width: 18px; height: 18px; border: 3px solid #334155; border-top-color: #6366f1; border-radius: 50%; animation: spin 0.7s linear infinite; vertical-align: middle; margin-right: 6px; }
  @keyframes spin { to { transform: rotate(360deg); } }
  pre { background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 16px; overflow-x: auto; font-size: 0.85rem; color: #e2e8f0; white-space: pre-wrap; }
</style>
"""

NAVBAR = """
<nav class="navbar">
  <div class="brand">&#129302; talent decode</div>
  <nav>
    <a href="/">Home</a>
    <a href="/analyze">Analyze</a>
    <a href="/chatbot">Chatbot</a>
  </nav>
</nav>
"""

INDEX_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Resume Scanner</title>
""" + BASE_STYLE + """
</head>
<body>
""" + NAVBAR + """
<div class="container">
  <div class="hero">
    <h1>AI-Powered <span>Resume Scanner</span><br>& Career Coach</h1>
    <p>Upload your resume for instant ATS scoring, job match analysis, skill gap detection, and personalized career recommendations — all free, no API keys.</p>
    <a href="/analyze" class="btn btn-primary" style="font-size:1.05rem;padding:14px 36px;">Scan My Resume &#8594;</a>
  </div>

  <div class="features-grid">
    <div class="feature-item">
      <div class="icon">&#128200;</div>
      <h3>ATS Score</h3>
      <p>Get a 0-100 ATS score with section-by-section breakdown</p>
    </div>
    <div class="feature-item">
      <div class="icon">&#127919;</div>
      <h3>JD Matching</h3>
      <p>TF-IDF cosine similarity matching with any job description</p>
    </div>
    <div class="feature-item">
      <div class="icon">&#128214;</div>
      <h3>Skill Gap Analysis</h3>
      <p>Identify missing skills and get free course recommendations</p>
    </div>
    <div class="feature-item">
      <div class="icon">&#129302;</div>
      <h3>AI Suggestions</h3>
      <p>Actionable tips to improve your resume instantly</p>
    </div>
    <div class="feature-item">
      <div class="icon">&#127970;</div>
      <h3>Company Match</h3>
      <p>Discover companies that match your skill profile</p>
    </div>
    <div class="feature-item">
      <div class="icon">&#128172;</div>
      <h3>Career Chatbot</h3>
      <p>Ask career questions and get personalized answers</p>
    </div>
    <div class="feature-item">
      <div class="icon">&#128196;</div>
      <h3>Resume Builder</h3>
      <p>Auto-generate an ATS-friendly resume template</p>
    </div>
    <div class="feature-item">
      <div class="icon">&#128202;</div>
      <h3>Visualizations</h3>
      <p>Charts for ATS score and skill match breakdown</p>
    </div>
  </div>

  <div class="card">
    <h2>&#128640; Get Started in 3 Steps</h2>
    <div class="grid-3" style="text-align:center;gap:24px;">
      <div>
        <div style="font-size:2rem;margin-bottom:8px;">&#128194;</div>
        <h3 style="color:#f1f5f9;margin-bottom:6px;">1. Upload Resume</h3>
        <p style="color:#64748b;font-size:0.9rem;">PDF, DOCX, or TXT — up to 5MB</p>
      </div>
      <div>
        <div style="font-size:2rem;margin-bottom:8px;">&#128203;</div>
        <h3 style="color:#f1f5f9;margin-bottom:6px;">2. Paste Job Description</h3>
        <p style="color:#64748b;font-size:0.9rem;">Optional — for JD match & skill gap analysis</p>
      </div>
      <div>
        <div style="font-size:2rem;margin-bottom:8px;">&#129351;</div>
        <h3 style="color:#f1f5f9;margin-bottom:6px;">3. Get AI Insights</h3>
        <p style="color:#64748b;font-size:0.9rem;">Full report with score, suggestions & more</p>
      </div>
    </div>
  </div>
</div>
</body></html>
"""

ANALYZE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Analyze Resume — AI Resume Scanner</title>
""" + BASE_STYLE + """
</head>
<body>
""" + NAVBAR + """
<div class="container">
  {% if error %}
  <div class="alert alert-warning">{{ error }}</div>
  {% endif %}

  <div class="card">
    <h2>&#128194; Upload Your Resume</h2>
    <form method="POST" action="/analyze" enctype="multipart/form-data" id="analyzeForm">
      <div class="grid-2">
        <div>
          <div class="form-group">
            <label>Resume File (PDF, DOCX, TXT)</label>
            <div class="upload-zone" onclick="document.getElementById('resumeFile').click()">
              <div class="upload-icon">&#128196;</div>
              <div style="color:#94a3b8;font-size:0.95rem;">Click to upload or drag & drop</div>
              <div style="color:#64748b;font-size:0.8rem;margin-top:4px;">Max 5MB · PDF, DOCX, TXT</div>
              <input type="file" id="resumeFile" name="resume" accept=".pdf,.docx,.txt" required style="display:none" onchange="this.parentNode.querySelector('div:nth-child(2)').textContent=this.files[0]?this.files[0].name:'Click to upload'">
            </div>
          </div>
        </div>
        <div>
          <div class="form-group">
            <label>Job Description (Optional — for JD matching)</label>
            <textarea name="job_description" placeholder="Paste the full job description here for skill gap analysis and match scoring...">{{ jd or '' }}</textarea>
          </div>
        </div>
      </div>
      <button type="submit" class="btn btn-primary" onclick="this.innerHTML='<span class=spinner></span>Analyzing...'">
        &#128202; Analyze Resume
      </button>
    </form>
  </div>
</div>
</body></html>
"""

RESULTS_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Results —talent decode</title>
""" + BASE_STYLE + """
</head>
<body>
""" + NAVBAR + """
<div class="container">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:24px;">
    <div>
      <h1 style="font-size:1.5rem;color:#f1f5f9;font-weight:700;">Resume Analysis Report</h1>
      <div style="color:#64748b;font-size:0.85rem;">{{ name }} &bull; Analyzed {{ timestamp }}</div>
    </div>
    <div>
      <a href="/analyze" class="btn btn-secondary">&#8592; Analyze Another</a>
      <a href="/download-resume" class="btn btn-success" style="margin-left:8px;">&#8659; Download ATS Resume</a>
    </div>
  </div>

  <!-- Tab Navigation -->
  <div class="tab-nav">
    <a href="#ats" class="active" onclick="showTab('ats',this)">&#128200; ATS Score</a>
    <a href="#jd" onclick="showTab('jd',this)">&#127919; JD Match</a>
    <a href="#skills" onclick="showTab('skills',this)">&#128214; Skills</a>
    <a href="#suggestions" onclick="showTab('suggestions',this)">&#129302; Suggestions</a>
    <a href="#courses" onclick="showTab('courses',this)">&#127891; Courses</a>
    <a href="#companies" onclick="showTab('companies',this)">&#127970; Companies</a>
    <a href="#charts" onclick="showTab('charts',this)">&#128202; Charts</a>
    <a href="#builder" onclick="showTab('builder',this)">&#128196; Resume</a>
  </div>

  <!-- ATS Score Tab -->
  <div id="tab-ats">
    <div class="grid-2">
      <div class="card" style="text-align:center;">
        <h2>&#128200; ATS Score</h2>
        <div class="score-big {{ score_class }}">{{ ats_score }}<span style="font-size:1.5rem;color:#64748b;">/100</span></div>
        <div style="color:#94a3b8;margin-bottom:16px;">
          {% if ats_score >= 70 %}&#9989; Strong Resume
          {% elif ats_score >= 50 %}&#9888;&#65039; Needs Improvement
          {% else %}&#10060; Significant Work Needed
          {% endif %}
        </div>
        <div class="progress-bar"><div class="progress-fill" style="width:{{ ats_score }}%;background:{{ score_color }};"></div></div>
        <div style="color:#64748b;font-size:0.8rem;margin-top:8px;">Score: {{ ats_score }}/100</div>
      </div>
      <div class="card">
        <h2>&#128203; Quick Stats</h2>
        <div class="grid-2" style="gap:12px;">
          <div class="stat-box">
            <div class="num">{{ ats_details.length.word_count }}</div>
            <div class="label">Words</div>
          </div>
          <div class="stat-box">
            <div class="num">{{ ats_details.keywords.count }}</div>
            <div class="label">Skills Found</div>
          </div>
          <div class="stat-box">
            <div class="num">{{ ats_details.sections.found|length }}</div>
            <div class="label">Sections</div>
          </div>
          <div class="stat-box">
            <div class="num">{{ ats_details.action_verbs.found|length }}</div>
            <div class="label">Action Verbs</div>
          </div>
        </div>
      </div>
    </div>

    <div class="grid-3">
      <div class="card">
        <div class="section-header">Sections ({{ ats_details.sections.score }}/30)</div>
        {% for s in ats_details.sections.found %}<span class="badge badge-green">&#10003; {{ s }}</span>{% endfor %}
        {% for s in ats_details.sections.missing %}<span class="badge badge-red">&#10007; {{ s }}</span>{% endfor %}
      </div>
      <div class="card">
        <div class="section-header">Contact Info ({{ ats_details.contact.score }}/15)</div>
        <div style="margin-bottom:6px;color:{% if ats_details.contact.email %}#4ade80{% else %}#fca5a5{% endif %};">
          {% if ats_details.contact.email %}&#10003; Email: {{ ats_details.contact.email }}{% else %}&#10007; No email found{% endif %}
        </div>
        <div style="margin-bottom:6px;color:{% if ats_details.contact.phone %}#4ade80{% else %}#fca5a5{% endif %};">
          {% if ats_details.contact.phone %}&#10003; Phone detected{% else %}&#10007; No phone found{% endif %}
        </div>
        <div style="color:{% if ats_details.contact.linkedin %}#4ade80{% else %}#fca5a5{% endif %};">
          {% if ats_details.contact.linkedin %}&#10003; LinkedIn detected{% else %}&#10007; No LinkedIn{% endif %}
        </div>
      </div>
      <div class="card">
        <div class="section-header">Formatting ({{ ats_details.formatting.score }}/10)</div>
        <div style="margin-bottom:6px;color:{% if ats_details.formatting.has_bullets %}#4ade80{% else %}#fca5a5{% endif %};">
          {% if ats_details.formatting.has_bullets %}&#10003; Bullet points{% else %}&#10007; No bullets detected{% endif %}
        </div>
        <div style="margin-bottom:6px;color:{% if ats_details.formatting.has_dates %}#4ade80{% else %}#fca5a5{% endif %};">
          {% if ats_details.formatting.has_dates %}&#10003; Dates included{% else %}&#10007; No dates found{% endif %}
        </div>
        <div style="color:{% if ats_details.formatting.has_metrics %}#4ade80{% else %}#fca5a5{% endif %};">
          {% if ats_details.formatting.has_metrics %}&#10003; Metrics/numbers{% else %}&#10007; No quantified metrics{% endif %}
        </div>
      </div>
    </div>

    <div class="card">
      <div class="section-header">Skills Detected ({{ ats_details.keywords.count }} skills)</div>
      {% for skill in ats_details.keywords.found %}<span class="badge badge-blue">{{ skill }}</span>{% endfor %}
      {% if not ats_details.keywords.found %}<span style="color:#64748b;">No recognized skills found — add a Skills section</span>{% endif %}
    </div>
  </div>

  <!-- JD Match Tab -->
  <div id="tab-jd" style="display:none;">
    {% if jd_match %}
    <div class="grid-2">
      <div class="card" style="text-align:center;">
        <h2>&#127919; JD Match Score</h2>
        <div class="score-big {% if jd_match.match_percent >= 60 %}score-green{% elif jd_match.match_percent >= 35 %}score-yellow{% else %}score-red{% endif %}">
          {{ jd_match.match_percent }}<span style="font-size:1.5rem;color:#64748b;">%</span>
        </div>
        <div class="progress-bar"><div class="progress-fill" style="width:{{ jd_match.match_percent }}%;background:{% if jd_match.match_percent >= 60 %}#22c55e{% elif jd_match.match_percent >= 35 %}#f59e0b{% else %}#ef4444{% endif %};"></div></div>
        <div style="color:#94a3b8;margin-top:10px;font-size:0.9rem;">
          {% if jd_match.match_percent >= 60 %}Strong match — good fit for this role
          {% elif jd_match.match_percent >= 35 %}Moderate match — bridge the skill gaps
          {% else %}Low match — significant upskilling needed
          {% endif %}
        </div>
      </div>
      <div class="card">
        <h2>&#128202; Match Summary</h2>
        <div class="grid-2" style="gap:12px;">
          <div class="stat-box"><div class="num" style="color:#22c55e;">{{ jd_match.matched_skills|length }}</div><div class="label">Matched Skills</div></div>
          <div class="stat-box"><div class="num" style="color:#ef4444;">{{ jd_match.missing_skills|length }}</div><div class="label">Missing Skills</div></div>
          <div class="stat-box"><div class="num">{{ jd_match.jd_skills_count }}</div><div class="label">JD Skills Total</div></div>
          <div class="stat-box"><div class="num">{{ jd_match.resume_skills_count }}</div><div class="label">Resume Skills</div></div>
        </div>
      </div>
    </div>
    <div class="grid-2">
      <div class="card">
        <div class="section-header">&#10003; Matched Skills</div>
        {% for skill in jd_match.matched_skills %}<span class="badge badge-green">{{ skill }}</span>{% endfor %}
        {% if not jd_match.matched_skills %}<span style="color:#64748b;">No matching skills detected</span>{% endif %}
      </div>
      <div class="card">
        <div class="section-header">&#10007; Missing Skills (from JD)</div>
        {% for skill in jd_match.missing_skills %}<span class="badge badge-red">{{ skill }}</span>{% endfor %}
        {% if not jd_match.missing_skills %}<span style="color:#4ade80;">&#10003; You have all the JD skills!</span>{% endif %}
      </div>
    </div>
    {% else %}
    <div class="card">
      <div class="alert alert-info">&#128161; No job description was provided. <a href="/analyze" style="color:#93c5fd;">Go back</a> and paste a job description to see match analysis.</div>
    </div>
    {% endif %}
  </div>

  <!-- Skills Tab -->
  <div id="tab-skills" style="display:none;">
    <div class="card">
      <h2>&#128214; All Skills Detected in Resume</h2>
      {% for skill in all_resume_skills %}
        <span class="badge badge-blue">{{ skill }}</span>
      {% endfor %}
      {% if not all_resume_skills %}<span style="color:#64748b;">No recognized skills found. Add a Skills section with specific technologies.</span>{% endif %}
    </div>
    {% if jd_match %}
    <div class="card">
      <h2>&#128270; Skill Gap Analysis</h2>
      <div class="alert alert-info">Skills in the job description that are NOT in your resume. Add these to your Skills section after gaining experience.</div>
      <div style="margin-top:16px;">
        {% for skill in jd_match.missing_skills %}
        <div style="padding:10px 14px;background:#0f172a;border-radius:8px;border:1px solid #334155;margin-bottom:8px;display:flex;justify-content:space-between;align-items:center;">
          <span style="color:#e2e8f0;">{{ skill }}</span>
          <span class="badge badge-red">Missing</span>
        </div>
        {% endfor %}
        {% if not jd_match.missing_skills %}
        <div class="alert alert-success">&#127775; Your resume covers all skills in the job description!</div>
        {% endif %}
      </div>
    </div>
    {% endif %}
  </div>

  <!-- Suggestions Tab -->
  <div id="tab-suggestions" style="display:none;">
    <div class="card">
      <h2>&#129302; AI-Powered Resume Suggestions</h2>
      <p style="color:#64748b;margin-bottom:20px;font-size:0.9rem;">Personalized recommendations to boost your ATS score and land more interviews.</p>
      {% for s in suggestions %}
      <div class="suggestion {{ s.priority.lower() }}">
        <div class="category">
          {% if s.priority == 'Critical' %}&#128308;{% elif s.priority == 'High' %}&#128992;{% elif s.priority == 'Medium' %}&#128993;{% else %}&#128994;{% endif %}
          {{ s.category }} &bull; {{ s.priority }} Priority
        </div>
        <div class="text">{{ s.suggestion }}</div>
        <div class="example">&#128161; Example: {{ s.example }}</div>
      </div>
      {% endfor %}
    </div>
  </div>

  <!-- Courses Tab -->
  <div id="tab-courses" style="display:none;">
    <div class="card">
      <h2>&#127891; Free Course Recommendations</h2>
      <p style="color:#64748b;margin-bottom:20px;font-size:0.9rem;">Curated free resources to help you bridge skill gaps. All courses are free or audit-free.</p>
      <div class="grid-3">
        {% for course in courses %}
        <div class="course-card">
          <div class="skill-tag">{{ course.skill }}</div>
          <div class="title">{{ course.title }}</div>
          <div class="platform">{{ course.platform }}</div>
          <div style="margin-top:8px;"><a href="{{ course.url }}" target="_blank" rel="noopener">Visit Course &#8599;</a></div>
        </div>
        {% endfor %}
        {% if not courses %}<div style="color:#64748b;">Add a job description to get targeted course recommendations.</div>{% endif %}
      </div>
    </div>
  </div>

  <!-- Companies Tab -->
  <div id="tab-companies" style="display:none;">
    <div class="card">
      <h2>&#127970; Company Suggestions Based on Your Skills</h2>
      <p style="color:#64748b;margin-bottom:20px;font-size:0.9rem;">Companies that commonly hire for your skill profile.</p>
      {% for c in companies %}
      <div class="company-item">
        <div>
          <div style="color:#e2e8f0;font-weight:600;">{{ c.company }}</div>
          <div style="font-size:0.8rem;color:#64748b;">Relevant skills: {{ c.skills | join(', ') }}</div>
        </div>
        <span class="badge badge-purple">Match</span>
      </div>
      {% endfor %}
    </div>
  </div>

  <!-- Charts Tab -->
  <div id="tab-charts" style="display:none;">
    <div class="card">
      <h2>&#128202; ATS Score Breakdown</h2>
      <img src="data:image/png;base64,{{ ats_chart }}" alt="ATS Score Chart" style="width:100%;border-radius:8px;">
    </div>
    {% if skill_chart %}
    <div class="card">
      <h2>&#128202; Skill Match Visualization</h2>
      <img src="data:image/png;base64,{{ skill_chart }}" alt="Skill Match Chart" style="width:100%;border-radius:8px;">
    </div>
    {% endif %}
  </div>

  <!-- Resume Builder Tab -->
  <div id="tab-builder" style="display:none;">
    <div class="card">
      <h2>&#128196; Auto-Generated ATS-Friendly Resume</h2>
      <div class="alert alert-info" style="margin-bottom:16px;">&#128161; This is a structured template based on your resume data. Edit it to add your actual content, then download.</div>
      <pre>{{ built_resume }}</pre>
      <div style="margin-top:16px;">
        <a href="/download-resume" class="btn btn-success">&#8659; Download as Text File</a>
      </div>
    </div>
  </div>

</div>

<script>
function showTab(id, el) {
  document.querySelectorAll('[id^="tab-"]').forEach(t => t.style.display='none');
  document.getElementById('tab-'+id).style.display='block';
  document.querySelectorAll('.tab-nav a').forEach(a => a.classList.remove('active'));
  el.classList.add('active');
  return false;
}
</script>
</body></html>
"""

CHATBOT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Career Chatbot — AI Resume Scanner</title>
""" + BASE_STYLE + """
</head>
<body>
""" + NAVBAR + """
<div class="container" style="max-width:720px;">
  <div class="card">
    <h2>&#128172; Career Chatbot</h2>
    <p style="color:#64748b;font-size:0.9rem;margin-bottom:16px;">Ask about resume tips, skills to learn, interview prep, companies to apply, and more!</p>

    <div class="chat-messages" id="chatMessages">
      <div class="msg msg-bot">
        <div class="msg-bubble">Hello! I'm your AI career assistant &#129302;<br><br>I can help you with:
• Resume improvement tips
• Skills to learn for your field
• Company suggestions
• Interview preparation
• ATS optimization
• Salary negotiation

What would you like to know?</div>
      </div>
      {% for msg in chat_history %}
      <div class="msg {{ 'msg-user' if msg.role == 'user' else 'msg-bot' }}">
        <div class="msg-bubble">{{ msg.content }}</div>
      </div>
      {% endfor %}
    </div>

    <form method="POST" action="/chatbot" id="chatForm">
      <div class="chat-input-row">
        <input type="text" name="message" id="chatInput" placeholder="Ask me anything about your career..." autocomplete="off" required>
        <button type="submit" class="btn btn-primary">Send</button>
      </div>
    </form>

    <div style="margin-top:16px;">
      <div class="section-header">Quick Questions</div>
      <div style="display:flex;flex-wrap:wrap;gap:8px;">
        {% for q in quick_questions %}
        <button class="btn btn-secondary" style="font-size:0.8rem;padding:6px 12px;" onclick="document.getElementById('chatInput').value=this.textContent;document.getElementById('chatForm').submit()">{{ q }}</button>
        {% endfor %}
      </div>
    </div>
  </div>

  <div class="card">
    <div class="section-header">Upload Resume for Personalized Answers</div>
    {% if has_resume %}
    <div class="alert alert-success">&#10003; Resume loaded — answers will be personalized to your skill profile!</div>
    {% else %}
    <div class="alert alert-info">&#128161; <a href="/analyze" style="color:#93c5fd;">Analyze your resume first</a> for personalized career advice based on YOUR skills.</div>
    {% endif %}
  </div>
</div>

<script>
var el = document.getElementById('chatMessages');
el.scrollTop = el.scrollHeight;
document.getElementById('chatForm').addEventListener('submit', function() {
  document.querySelector('.chat-input-row button').textContent = '...';
});
</script>
</body></html>
"""

# ─────────────────────────────────────────────
#  Flask Routes
# ─────────────────────────────────────────────

@app.route('/')
def index():
    """Landing page."""
    return render_template_string(INDEX_TEMPLATE)

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    """Resume upload and analysis."""
    if request.method == 'GET':
        return render_template_string(ANALYZE_TEMPLATE, error=None, jd='')

    # --- Handle file upload ---
    if 'resume' not in request.files or request.files['resume'].filename == '':
        return render_template_string(ANALYZE_TEMPLATE, error="Please upload a resume file.", jd='')

    resume_file = request.files['resume']
    jd_text = request.form.get('job_description', '').strip()

    # Validate file extension
    allowed_ext = {'.pdf', '.docx', '.txt'}
    file_ext = os.path.splitext(resume_file.filename.lower())[1]
    if file_ext not in allowed_ext:
        return render_template_string(ANALYZE_TEMPLATE,
                                      error="Unsupported format. Upload a PDF, DOCX, or TXT file.", jd=jd_text)

    # Extract text
    resume_text = extract_resume_text(resume_file)
    if not resume_text or resume_text.startswith("Error"):
        return render_template_string(ANALYZE_TEMPLATE,
                                      error=f"Could not read the file: {resume_text}", jd=jd_text)

    # Store in session for chatbot
    session['resume_text'] = resume_text[:4000]  # Limit session size
    session['jd_text'] = jd_text[:2000]

    # --- Core Analysis ---
    ats_score, ats_details = calculate_ats_score(resume_text)
    all_resume_skills = sorted(extract_skills_from_text(resume_text))

    jd_match = None
    missing_skills = set()
    if jd_text:
        jd_match = calculate_jd_match(resume_text, jd_text)
        missing_skills = set(jd_match['missing_skills'])

    suggestions = generate_resume_suggestions(resume_text, ats_details, missing_skills)
    courses = get_course_recommendations(list(missing_skills) or all_resume_skills[:3])
    companies = suggest_companies(all_resume_skills)
    built_resume = build_ats_resume(resume_text, ats_details, suggestions)

    # Store for download
    session['built_resume'] = built_resume

    # --- Charts ---
    ats_chart = generate_ats_chart(ats_score, ats_details)
    skill_chart = None
    if jd_match:
        skill_chart = generate_skill_chart(
            set(jd_match['matched_skills']),
            set(jd_match['missing_skills']),
            set(all_resume_skills)
        )

    # Determine score class
    if ats_score >= 70:
        score_class = 'score-green'
        score_color = '#22c55e'
    elif ats_score >= 50:
        score_class = 'score-yellow'
        score_color = '#f59e0b'
    else:
        score_class = 'score-red'
        score_color = '#ef4444'

    return render_template_string(
        RESULTS_TEMPLATE,
        name=extract_name(resume_text),
        timestamp=datetime.now().strftime('%B %d, %Y %I:%M %p'),
        ats_score=ats_score,
        ats_details=ats_details,
        all_resume_skills=all_resume_skills,
        jd_match=jd_match,
        suggestions=suggestions,
        courses=courses,
        companies=companies,
        built_resume=built_resume,
        ats_chart=ats_chart,
        skill_chart=skill_chart,
        score_class=score_class,
        score_color=score_color,
    )

@app.route('/download-resume')
def download_resume():
    """Download the generated ATS resume as a text file."""
    built_resume = session.get('built_resume', 'No resume generated yet. Please analyze a resume first.')
    buf = io.BytesIO(built_resume.encode('utf-8'))
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name='ats_resume.txt', mimetype='text/plain')

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    """Career chatbot page."""
    if 'chat_history' not in session:
        session['chat_history'] = []

    quick_questions = [
        "How to improve my resume?",
        "What skills should I learn?",
        "Which companies should I apply to?",
        "How to prepare for interviews?",
        "What is ATS and how to optimize?",
        "How to negotiate salary?",
    ]

    resume_text = session.get('resume_text', '')
    has_resume = bool(resume_text)

    if request.method == 'POST':
        user_msg = request.form.get('message', '').strip()
        if user_msg:
            jd_text = session.get('jd_text', '')
            response = chatbot_response(user_msg, resume_text, jd_text)

            history = session.get('chat_history', [])
            history.append({'role': 'user', 'content': user_msg})
            history.append({'role': 'bot', 'content': response})
            # Keep last 20 messages
            session['chat_history'] = history[-20:]

    return render_template_string(
        CHATBOT_TEMPLATE,
        chat_history=session.get('chat_history', []),
        quick_questions=quick_questions,
        has_resume=has_resume,
    )

@app.route('/clear-chat', methods=['POST'])
def clear_chat():
    session.pop('chat_history', None)
    return redirect(url_for('chatbot'))

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """JSON API endpoint for chatbot (AJAX)."""
    data = request.get_json(force=True)
    message = data.get('message', '')
    resume_text = session.get('resume_text', '')
    jd_text = session.get('jd_text', '')
    response = chatbot_response(message, resume_text, jd_text)
    return jsonify({'response': response})

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'app': 'AI Resume Scanner'})
if __name__ == '__main__':
    print("""
╔══════════════════════════════════════════════════════════╗
║         AI-Powered Resume Scanner — Starting Up         ║
║                                                          ║
║  Open: http://localhost:5000                             ║
║                                                          ║
║  Features:                                               ║
║  • ATS Score (0-100) with detailed breakdown             ║
║  • Job Description Matching (TF-IDF)                     ║
║  • Skill Gap Detection + Course Recommendations          ║
║  • AI Resume Suggestions                                 ║
║  • ATS Resume Builder (downloadable)                     ║
║  • Company Suggestions                                   ║
║  • Career Chatbot (rule-based + TF-IDF)                  ║
║  • Visualizations (matplotlib charts)                    ║
╚══════════════════════════════════════════════════════════╝
    """)
    port = int(os.environ.get('PORT', 8000))
    app.run(debug=True, host='0.0.0.0', port=port)
print("bimla")