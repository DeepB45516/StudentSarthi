"""
agents.py — Multi-API Agent System
  - Groq Llama3 (groq_api_1, groq_api_2 for load balancing)
  - Gemini (gemini_api_1, gemini_api_2 for different tasks)
  - Google/DuckDuckGo search
"""

import os
import json
import requests
import random
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════
# MULTI-KEY API CONFIGURATION
# Each model uses its own dedicated API key pool
# ══════════════════════════════════════════════════════

# Groq API Keys — used for: interview agent, research agent, report agent, skill verification
GROQ_KEYS = [
    os.getenv("GROQ_API_1") or os.getenv("GROQ_API_KEY", ""),
    os.getenv("GROQ_API_2") or os.getenv("GROQ_API_KEY", ""),
]
GROQ_KEYS = [k for k in GROQ_KEYS if k]  # remove empty

# Gemini API Keys — used for: AI Daily newsletter, DocuGenius document analysis
GEMINI_KEYS = [
    os.getenv("GEMINI_API_1") or os.getenv("GEMINI_API_KEY", ""),
    os.getenv("GEMINI_API_2") or os.getenv("GEMINI_API_KEY", ""),
]
GEMINI_KEYS = [k for k in GEMINI_KEYS if k]  # remove empty

GOOGLE_SEARCH_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
GOOGLE_SEARCH_CX  = os.getenv("GOOGLE_SEARCH_CX")

GROQ_MODEL   = "llama-3.1-8b-instant"
GEMINI_MODEL = "gemini-2.0-flash"


def _get_groq_client() -> Groq:
    """Return a Groq client using a random key for load balancing."""
    if not GROQ_KEYS:
        raise ValueError("No GROQ API keys configured. Set GROQ_API_1 or GROQ_API_KEY in .env")
    key = random.choice(GROQ_KEYS)
    return Groq(api_key=key)


def _get_gemini_key() -> str:
    """Return a Gemini key using a random key for load balancing."""
    if not GEMINI_KEYS:
        raise ValueError("No GEMINI API keys configured. Set GEMINI_API_1 or GEMINI_API_KEY in .env")
    return random.choice(GEMINI_KEYS)


# ══════════════════════════════════════════════════════
# CORE LLM CALLS
# ══════════════════════════════════════════════════════

def llm(prompt: str, json_mode: bool = False) -> str:
    """Call Groq Llama3 using groq_api_1 / groq_api_2."""
    if json_mode:
        prompt += "\n\nIMPORTANT: Respond ONLY with valid JSON. No markdown fences, no extra text, no explanation."
    client = _get_groq_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    text = response.choices[0].message.content.strip()
    if json_mode:
        text = text.replace("```json", "").replace("```", "").strip()
    return text


def gemini(prompt: str) -> str:
    """Call Gemini using gemini_api_1 / gemini_api_2. Falls back to Groq on error."""
    key = _get_gemini_key()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={key}"
    try:
        resp = requests.post(
            url,
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=25,
        )
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        print(f"❌ GEMINI ERROR (falling back to Groq): {e}")
        return llm(prompt)


def gemini_analyze_document(text: str) -> dict:
    """
    DocuGenius: Analyze a document using Gemini (gemini_api_1 / gemini_api_2).
    Returns structured DocumentAnalysis dict.
    """
    clean_text = " ".join(text.replace("\n", " ").split())[:8000]
    prompt = f"""Analyze the following document and return a JSON object with this EXACT structure (no markdown, no fences):

{{
  "shortSummary": "A 5-line concise summary",
  "detailedSummary": "A detailed explanation with bullet points",
  "keyPoints": ["Key point 1", "Key point 2", "Key point 3", "Key point 4", "Key point 5"],
  "importantConcepts": ["Concept 1", "Concept 2", "Concept 3"],
  "mcqs": [
    {{"question": "Q1?", "options": ["A", "B", "C", "D"], "correctAnswer": "A"}},
    {{"question": "Q2?", "options": ["A", "B", "C", "D"], "correctAnswer": "B"}},
    {{"question": "Q3?", "options": ["A", "B", "C", "D"], "correctAnswer": "C"}},
    {{"question": "Q4?", "options": ["A", "B", "C", "D"], "correctAnswer": "D"}},
    {{"question": "Q5?", "options": ["A", "B", "C", "D"], "correctAnswer": "A"}}
  ],
  "shortQuestions": ["Short Q1?", "Short Q2?", "Short Q3?", "Short Q4?", "Short Q5?"],
  "longQuestions": ["Long Q1?", "Long Q2?", "Long Q3?"],
  "answers": [
    {{"question": "Q1?", "answer": "Answer 1"}},
    {{"question": "Q2?", "answer": "Answer 2"}}
  ],
  "insights": {{
    "difficulty": "Easy",
    "readingTime": "5 mins",
    "focusAreas": ["Area 1", "Area 2", "Area 3"]
  }}
}}

Document Text:
{clean_text}

Return ONLY the JSON object. No markdown fences. No extra text."""

    key = _get_gemini_key()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={key}"
    try:
        resp = requests.post(
            url,
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=40,
        )
        resp.raise_for_status()
        raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        print(f"❌ DocuGenius Gemini error: {e}")
        # Fallback to Groq
        try:
            raw = llm(prompt, json_mode=True)
            return json.loads(raw)
        except Exception as e2:
            raise Exception(f"Both Gemini and Groq failed for document analysis: {e2}")


# ══════════════════════════════════════════════════════
# GOOGLE / DUCKDUCKGO SEARCH
# ══════════════════════════════════════════════════════

def google_search(query: str, num: int = 8) -> list:
    if GOOGLE_SEARCH_KEY and GOOGLE_SEARCH_CX:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {"key": GOOGLE_SEARCH_KEY, "cx": GOOGLE_SEARCH_CX, "q": query, "num": min(num, 10)}
        try:
            r = requests.get(url, params=params, timeout=8)
            data = r.json()
            results = [{"title": item.get("title",""), "link": item.get("link",""), "snippet": item.get("snippet","")} for item in data.get("items", [])]
            if results:
                return results
        except Exception as e:
            print(f"Google Search error: {e}")
    return duckduckgo_search(query, num)


def duckduckgo_search(query: str, num: int = 8) -> list:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        r = requests.get("https://html.duckduckgo.com/html/", params={"q": query}, headers=headers, timeout=8)
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(r.text, "html.parser")
        results = []
        for res in soup.select(".result")[:num]:
            title_el   = res.select_one(".result__title")
            link_el    = res.select_one(".result__url")
            snippet_el = res.select_one(".result__snippet")
            if title_el:
                results.append({
                    "title":   title_el.get_text(strip=True),
                    "link":    "https://" + link_el.get_text(strip=True) if link_el else "#",
                    "snippet": snippet_el.get_text(strip=True) if snippet_el else "",
                })
        return results
    except Exception as e:
        print(f"DuckDuckGo fallback error: {e}")
        return []


# ══════════════════════════════════════════════════════
# AGENT 1 — INTERVIEW AGENT (uses Groq groq_api_1/2)
# ══════════════════════════════════════════════════════

def interview_agent(user_input: str) -> dict:
    prompt = f"""You are a friendly student counselor AI. A student said:
"{user_input}"

Generate exactly 5 short clarifying questions to understand their profile before searching for opportunities.
Cover: field/domain, year of study, specific skills, location preference, goal (job/prize/experience/certificate).

Return ONLY this JSON structure:
{{
  "summary": "one sentence of what student wants",
  "questions": [
    {{"id": 1, "question": "What is your field of study?", "type": "select", "options": ["Computer Science", "Electronics", "Mechanical", "Civil", "Other"]}},
    {{"id": 2, "question": "Which year are you in?", "type": "select", "options": ["1st Year", "2nd Year", "3rd Year", "4th Year", "Postgraduate"]}},
    {{"id": 3, "question": "List your top 3 technical skills", "type": "text", "placeholder": "e.g. Python, Machine Learning, Web Dev"}},
    {{"id": 4, "question": "Your location / state?", "type": "text", "placeholder": "e.g. Maharashtra, Remote preferred"}},
    {{"id": 5, "question": "What matters most to you?", "type": "select", "options": ["Cash Prize", "Stipend", "Certificate", "Experience", "All"]}}
  ]
}}

Customize the questions based on what the student said. Keep them short and friendly."""

    text = llm(prompt, json_mode=True)
    try:
        return json.loads(text)
    except Exception:
        return {
            "summary": "Student looking for opportunities",
            "questions": [
                {"id": 1, "question": "What is your field of study?", "type": "text", "placeholder": "e.g. Computer Science"},
                {"id": 2, "question": "Which year are you in?", "type": "select", "options": ["1st Year","2nd Year","3rd Year","4th Year","Postgraduate"]},
                {"id": 3, "question": "Your top skills?", "type": "text", "placeholder": "e.g. Python, ML, Web Dev"},
                {"id": 4, "question": "Your location?", "type": "text", "placeholder": "e.g. Pune, Maharashtra"},
                {"id": 5, "question": "What matters most?", "type": "select", "options": ["Cash Prize","Stipend","Certificate","Experience"]},
            ]
        }


# ══════════════════════════════════════════════════════
# AGENT 2 — RESEARCH AGENT (uses Groq groq_api_1/2)
# ══════════════════════════════════════════════════════

def research_agent(profile: dict, categories: list) -> dict:
    logs = []
    all_results = {}

    field    = profile.get("What is your field of study?", profile.get("field", "engineering"))
    location = profile.get("Your location / state?", profile.get("location", "India"))
    skills   = profile.get("List your top 3 technical skills", profile.get("skills", ""))
    year     = profile.get("Which year are you in?", "")

    for category in categories:
        logs.append(f"[🔎 Research] Searching for {category} opportunities...")
        queries = [
            f"{category} for {field} students India 2025 2026 apply",
            f"best {category} {skills} students open registration",
            f"{category} {field} {location} stipend prize deadline 2025",
        ]
        raw_results = []
        for q in queries:
            results = google_search(q, num=5)
            raw_results.extend(results)
            logs.append(f"  → Searched: '{q}' → {len(results)} results")

        seen = set()
        unique = []
        for r in raw_results:
            if r["link"] not in seen:
                seen.add(r["link"])
                unique.append(r)

        indexed = unique[:15]
        search_text = "\n\n".join([
            f"[{i}] Title: {r['title']}\nURL: {r['link']}\nDescription: {r['snippet']}"
            for i, r in enumerate(indexed)
        ])
        url_map = {str(i): r['link'] for i, r in enumerate(indexed)}

        extract_prompt = f"""You are a research analyst. Based on these search results, extract real {category} opportunities for this student.

STUDENT PROFILE:
- Field: {field}
- Year: {year}
- Skills: {skills}
- Location: {location}

SEARCH RESULTS (each has an index number in brackets):
{search_text}

Extract 4-6 real opportunities from these results. Return ONLY a JSON array.
For "apply_link" use the index number of the result (e.g. "0", "3") — NOT the URL itself.

[
  {{
    "title": "exact opportunity name",
    "organizer": "company or org",
    "deadline": "date or Check website",
    "stipend_prize": "amount or N/A",
    "eligibility": "who can apply",
    "why_suitable": "one sentence why this fits the student",
    "apply_link": "0",
    "difficulty": "Beginner or Intermediate or Advanced",
    "tags": ["tag1", "tag2"]
  }}
]

IMPORTANT: apply_link must be the index number of the search result (0-{len(indexed)-1}). No invented URLs."""

        extracted_raw = llm(extract_prompt, json_mode=True)

        try:
            opps = json.loads(extracted_raw) if isinstance(extracted_raw, str) else extracted_raw
            if isinstance(opps, list):
                for opp in opps:
                    idx = str(opp.get("apply_link", "")).strip()
                    if idx in url_map:
                        opp["apply_link"] = url_map[idx]
                    elif idx.isdigit() and int(idx) < len(indexed):
                        opp["apply_link"] = indexed[int(idx)]["link"]
                    else:
                        opp["apply_link"] = indexed[0]["link"] if indexed else "#"
                extracted_raw = json.dumps(opps)
        except Exception as e:
            logs.append(f"  ⚠ URL fix error: {e}")

        all_results[category] = extracted_raw
        logs.append(f"[🔎 Research] {category} done ✅")

    return {"results": all_results, "logs": logs}


# ══════════════════════════════════════════════════════
# AGENT 3 — REPORT AGENT (uses Groq groq_api_1/2)
# ══════════════════════════════════════════════════════

def report_agent(profile: dict, research_results: dict) -> dict:
    profile_text    = "\n".join([f"- {k}: {v}" for k, v in profile.items()])
    categories_data = []
    all_opps        = []

    for cat_name, raw in research_results.items():
        try:
            opps = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(opps, list):
                all_opps.extend(opps)
                categories_data.append({"name": cat_name, "opportunities": opps})
        except Exception:
            categories_data.append({"name": cat_name, "opportunities": []})

    cats_text = json.dumps(categories_data, indent=2)

    prompt = f"""You are a career counselor compiling a final report for a student.

STUDENT PROFILE:
{profile_text}

ALL OPPORTUNITIES FOUND:
{cats_text}

Create a comprehensive report in this EXACT JSON format:
{{
  "student_summary": "2-3 sentence profile and what they are looking for",
  "total_opportunities": <total count as number>,
  "categories": [
    {{
      "name": "Category name",
      "emoji": "🏆",
      "count": <number>,
      "opportunities": [
        {{
          "title": "...",
          "organizer": "...",
          "type": "...",
          "deadline": "...",
          "stipend_prize": "...",
          "eligibility": "...",
          "why_suitable": "...",
          "apply_link": "...",
          "difficulty": "...",
          "tags": ["...", "..."]
        }}
      ]
    }}
  ],
  "top_picks": [
    {{"rank": 1, "title": "...", "reason": "...", "apply_link": "..."}},
    {{"rank": 2, "title": "...", "reason": "...", "apply_link": "..."}},
    {{"rank": 3, "title": "...", "reason": "...", "apply_link": "..."}}
  ],
  "action_plan": [
    "Step 1: ...",
    "Step 2: ...",
    "Step 3: ...",
    "Step 4: ...",
    "Step 5: ..."
  ]
}}

Use emojis: 💻 Hackathon, 🏢 Internship, 🎓 Scholarship, 🏆 Competition, 🔬 Research, 🌟 Fellowship.
Top picks must be the 3 best matches for this student specifically.
Action plan must be concrete and specific."""

    text = llm(prompt, json_mode=True)
    try:
        return json.loads(text)
    except Exception:
        return {
            "student_summary": "Report generated successfully.",
            "total_opportunities": len(all_opps),
            "categories": categories_data,
            "top_picks": [],
            "action_plan": ["Review all opportunities above and apply to the most relevant ones."]
        }


# ══════════════════════════════════════════════════════
# SKILL VERIFICATION AGENT (uses Groq groq_api_1/2)
# ══════════════════════════════════════════════════════

def verify_skill_agent(skills: str, field: str) -> dict:
    prompt = f"""You are a technical assessment engine. A student claims to have the following skills:
Skills: {skills}
Field: {field}

Generate exactly 5 multiple-choice questions to verify these skills. Questions should be moderate difficulty.

Return ONLY this JSON array (no other text):
[
  {{
    "question": "Question text here?",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "answer": 0
  }}
]

Rules:
- "answer" is the 0-based index of the correct option
- Cover different aspects/subtopics across the 5 questions
- Mix conceptual, practical, and scenario-based questions
- Keep questions clear and unambiguous
- Each question must have exactly 4 options"""

    text = llm(prompt, json_mode=True)
    try:
        questions = json.loads(text)
        if isinstance(questions, list) and len(questions) >= 3:
            return {"success": True, "questions": questions[:5]}
    except Exception:
        pass

    return {
        "success": True,
        "questions": [
            {"question": "What does 'debugging' mean in programming?", "options": ["Writing new features", "Finding and fixing errors in code", "Deleting old files", "Compiling code"], "answer": 1},
            {"question": "Which data structure uses LIFO (Last In, First Out)?", "options": ["Queue", "Array", "Stack", "Linked List"], "answer": 2},
            {"question": "What is the time complexity of binary search?", "options": ["O(n)", "O(n²)", "O(log n)", "O(1)"], "answer": 2},
            {"question": "What does HTML stand for?", "options": ["HyperText Markup Language", "High Transfer Markup Language", "HyperText Machine Language", "None of these"], "answer": 0},
            {"question": "Which keyword is used to define a function in Python?", "options": ["func", "function", "define", "def"], "answer": 3},
        ]
    }


# ══════════════════════════════════════════════════════
# PROFICIENCY REPORT AGENT (uses Groq groq_api_1/2)
# ══════════════════════════════════════════════════════

def proficiency_report_agent(skills: str, field: str, score: int, answered: list) -> dict:
    wrong = [a["question"] for a in answered if not a.get("correct")]
    right = [a["question"] for a in answered if a.get("correct")]

    prompt = f"""You are a professional domain competency analyst. Evaluate this professional/student's performance.

DOMAIN / FIELD: {field}
CLAIMED SKILLS: {skills}
QUIZ SCORE: {score}%
QUESTIONS ANSWERED CORRECTLY: {json.dumps(right)}
QUESTIONS ANSWERED INCORRECTLY: {json.dumps(wrong)}

Generate a detailed proficiency report. Return ONLY this JSON (no markdown, no extra text):
{{
  "domain_title": "Short domain title e.g. 'Python & Machine Learning'",
  "level": "Expert | Proficient | Intermediate | Beginner",
  "level_desc": "One sentence describing what this level means for this person",
  "efficiency_summary": "2-3 sentence executive summary of the person's efficiency in their domain",
  "strengths": [
    "Specific strength 1 based on what they got right",
    "Specific strength 2",
    "Specific strength 3"
  ],
  "gaps": [
    "Specific gap 1 based on what they got wrong",
    "Specific gap 2",
    "Specific gap 3"
  ],
  "skill_scores": [
    {{"skill": "Core {field} Knowledge", "score": 0}},
    {{"skill": "Practical Application",  "score": 0}},
    {{"skill": "Problem Solving",         "score": 0}},
    {{"skill": "Industry Awareness",      "score": 0}}
  ],
  "recommendations": [
    "Specific, actionable recommendation 1",
    "Specific, actionable recommendation 2",
    "Specific, actionable recommendation 3",
    "Specific, actionable recommendation 4"
  ],
  "career_readiness": "One sentence verdict on how ready this person is for a job/role in their domain"
}}

Set skill_scores based on quiz performance honestly. Do not inflate. Be constructive."""

    text = llm(prompt, json_mode=True)
    try:
        report = json.loads(text)
        return {"success": True, "report": report}
    except Exception:
        level = "Expert" if score >= 80 else "Proficient" if score >= 60 else "Intermediate" if score >= 40 else "Beginner"
        return {
            "success": True,
            "report": {
                "domain_title": field,
                "level": level,
                "level_desc": f"Scored {score}% on the skill verification test.",
                "efficiency_summary": f"The candidate scored {score}% demonstrating {level.lower()} knowledge in {field}.",
                "strengths": right[:3] if right else ["Attempted the assessment"],
                "gaps":      wrong[:3] if wrong else ["Review core concepts"],
                "skill_scores": [
                    {"skill": "Core Knowledge",        "score": score},
                    {"skill": "Practical Application", "score": max(0, score - 10)},
                    {"skill": "Problem Solving",        "score": max(0, score - 5)},
                    {"skill": "Industry Awareness",     "score": max(0, score - 15)},
                ],
                "recommendations": [
                    "Review the topics you answered incorrectly.",
                    "Build projects to solidify practical skills.",
                    "Take an advanced course in your domain.",
                    "Follow industry news and trends regularly.",
                ],
                "career_readiness": "Ready for junior-to-mid roles." if score >= 60 else "Needs more preparation before job applications."
            }
        }
