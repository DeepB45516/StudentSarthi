"""
app.py — Unified Student Sarthi
Features:
  1. Beautiful login page → dashboard with all features
  2. Student Opportunity Finder (Groq-powered, 3-agent pipeline)
  3. DocuGenius AI (Gemini-powered document analysis)
  4. AI Daily Newsletter (Gemini-powered)
  5. Resume Builder, Skill Verifier, Proficiency Reports
  6. Separate API keys per model (GROQ_API_1/2, GEMINI_API_1/2)
  7. One-click deploy on Render (Gunicorn ready)
"""

import os
import json
import io
import hashlib
import secrets
import feedparser
import requests
from datetime import datetime, timedelta
from functools import wraps
from urllib.parse import urlencode
from flask import (Flask, render_template, request, jsonify,
                   send_file, send_from_directory, session, redirect, url_for)
from flask_cors import CORS
from dotenv import load_dotenv

try:
    import psycopg2
    from psycopg2.extras import Json, RealDictCursor
except ImportError:
    psycopg2 = None
    Json = None
    RealDictCursor = None

load_dotenv()

app = Flask(__name__)

# Secret key for sessions — use env var on Render, fallback for dev
app.secret_key = os.getenv("SECRET_KEY", secrets.token_hex(32))
app.permanent_session_lifetime = timedelta(days=int(os.getenv("SESSION_DAYS", "30")))
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)

CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

from agents import (interview_agent, research_agent, report_agent,
                    verify_skill_agent, proficiency_report_agent,
                    gemini_analyze_document, gemini, GROQ_KEYS, GEMINI_KEYS)

# ══════════════════════════════════════════════════════
# SIMPLE USER STORE  (flat-file, perfect for Render)
# For production scale → swap with PostgreSQL / Redis
# ══════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
USERS_FILE = os.path.join(OUTPUT_DIR, "users.json")
PORTFOLIOS_FILE = os.path.join(OUTPUT_DIR, "portfolios.json")
DATABASE_URL = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL") or os.getenv("NEON_DATABASE_URL")

_db_initialized = False
_db_error = None

def _load_json(path: str, default):
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return default

def _save_json(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def _load_users() -> dict:
    return _load_json(USERS_FILE, {})

def _save_users(users: dict):
    _save_json(USERS_FILE, users)

def _hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def _check_password(pw: str, hashed: str) -> bool:
    return _hash_password(pw) == hashed

def _db_configured() -> bool:
    return bool(DATABASE_URL and psycopg2)

def _db_connect():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor, connect_timeout=8)

def _ensure_db() -> bool:
    global _db_initialized, _db_error
    if _db_initialized:
        return True
    if not _db_configured():
        _db_error = "PostgreSQL driver or DATABASE_URL is missing."
        return False
    try:
        with _db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY,
                        name TEXT NOT NULL,
                        email TEXT UNIQUE NOT NULL,
                        password_hash TEXT,
                        auth_provider TEXT NOT NULL DEFAULT 'password',
                        google_sub TEXT UNIQUE,
                        avatar_url TEXT,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        last_login_at TIMESTAMPTZ
                    );
                    CREATE TABLE IF NOT EXISTS portfolios (
                        id SERIAL PRIMARY KEY,
                        user_email TEXT UNIQUE NOT NULL REFERENCES users(email) ON DELETE CASCADE,
                        title TEXT NOT NULL DEFAULT 'My Portfolio',
                        theme TEXT NOT NULL DEFAULT 'glow',
                        data JSONB NOT NULL DEFAULT '{}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    CREATE INDEX IF NOT EXISTS idx_portfolios_user_email ON portfolios(user_email);
                """)
                for email, user in _load_users().items():
                    cur.execute("""
                        INSERT INTO users (name, email, password_hash, auth_provider, created_at)
                        VALUES (%s, %s, %s, 'password', %s)
                        ON CONFLICT (email) DO NOTHING
                    """, (
                        user.get("name") or email.split("@")[0],
                        email.lower(),
                        user.get("password") or user.get("password_hash"),
                        user.get("createdAt") or datetime.now().isoformat(),
                    ))
                for email, portfolio in _load_json(PORTFOLIOS_FILE, {}).items():
                    if not isinstance(portfolio, dict):
                        continue
                    email = email.lower()
                    cur.execute("SELECT 1 FROM users WHERE email = %s", (email,))
                    if not cur.fetchone():
                        continue
                    portfolio_data = portfolio.get("data") if isinstance(portfolio.get("data"), dict) else portfolio
                    cur.execute("""
                        INSERT INTO portfolios (user_email, title, theme, data, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, NOW(), NOW())
                        ON CONFLICT (user_email) DO NOTHING
                    """, (
                        email,
                        portfolio.get("title") or portfolio_data.get("name") or "My Portfolio",
                        portfolio.get("theme") or portfolio_data.get("theme") or "glow",
                        Json(portfolio_data),
                    ))
            conn.commit()
        _db_initialized = True
        _db_error = None
        return True
    except Exception as exc:
        _db_error = str(exc)
        return False

def _get_user(email: str):
    email = (email or "").strip().lower()
    if not email:
        return None
    if _ensure_db():
        try:
            with _db_connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT name, email, password_hash, auth_provider, google_sub, avatar_url
                        FROM users
                        WHERE email = %s
                    """, (email,))
                    row = cur.fetchone()
                    if row:
                        return dict(row)
        except Exception:
            pass
    user = _load_users().get(email)
    if user:
        return {
            "name": user.get("name") or email.split("@")[0],
            "email": email,
            "password_hash": user.get("password") or user.get("password_hash"),
            "auth_provider": user.get("auth_provider", "password"),
            "avatar_url": user.get("avatar_url"),
        }
    return None

def _create_password_user(name: str, email: str, password: str):
    password_hash = _hash_password(password)
    if _ensure_db():
        try:
            with _db_connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO users (name, email, password_hash, auth_provider, created_at, last_login_at)
                        VALUES (%s, %s, %s, 'password', NOW(), NOW())
                        RETURNING name, email
                    """, (name, email, password_hash))
                    user = dict(cur.fetchone())
                conn.commit()
            return True, user, None
        except Exception as exc:
            if "duplicate" in str(exc).lower() or "unique" in str(exc).lower():
                return False, None, "An account with this email already exists."

    users = _load_users()
    if email in users:
        return False, None, "An account with this email already exists."
    users[email] = {
        "name": name,
        "email": email,
        "password": password_hash,
        "createdAt": datetime.now().isoformat(),
    }
    _save_users(users)
    return True, {"name": name, "email": email}, None

def _touch_login(email: str):
    if not _ensure_db():
        return
    try:
        with _db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE users SET last_login_at = NOW() WHERE email = %s", (email,))
            conn.commit()
    except Exception:
        pass

def _upsert_google_user(profile: dict):
    email = (profile.get("email") or "").strip().lower()
    name = (profile.get("name") or email.split("@")[0]).strip()
    google_sub = profile.get("sub")
    avatar_url = profile.get("picture")
    if not email:
        return None

    if _ensure_db():
        try:
            with _db_connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO users (name, email, auth_provider, google_sub, avatar_url, created_at, last_login_at)
                        VALUES (%s, %s, 'google', %s, %s, NOW(), NOW())
                        ON CONFLICT (email) DO UPDATE SET
                            name = EXCLUDED.name,
                            google_sub = COALESCE(users.google_sub, EXCLUDED.google_sub),
                            avatar_url = EXCLUDED.avatar_url,
                            last_login_at = NOW()
                        RETURNING name, email, avatar_url
                    """, (name, email, google_sub, avatar_url))
                    user = dict(cur.fetchone())
                conn.commit()
            return user
        except Exception:
            pass

    users = _load_users()
    users[email] = {
        **users.get(email, {}),
        "name": name,
        "email": email,
        "auth_provider": "google",
        "google_sub": google_sub,
        "avatar_url": avatar_url,
        "createdAt": users.get(email, {}).get("createdAt", datetime.now().isoformat()),
    }
    _save_users(users)
    return {"name": name, "email": email, "avatar_url": avatar_url}

def _set_session_user(user: dict):
    session.permanent = True
    session["user"] = {
        "name": user.get("name") or user.get("email", "").split("@")[0],
        "email": user.get("email", "").lower(),
    }

def _get_portfolio(email: str):
    email = (email or "").strip().lower()
    if _ensure_db():
        try:
            with _db_connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT title, theme, data, updated_at
                        FROM portfolios
                        WHERE user_email = %s
                    """, (email,))
                    row = cur.fetchone()
                    if row:
                        result = dict(row)
                        if result.get("updated_at"):
                            result["updated_at"] = result["updated_at"].isoformat()
                        return result
        except Exception:
            pass
    return _load_json(PORTFOLIOS_FILE, {}).get(email)

def _save_portfolio(email: str, title: str, theme: str, data: dict):
    email = (email or "").strip().lower()
    title = (title or "My Portfolio").strip()[:160]
    theme = (theme or "glow").strip()[:40]
    if theme not in {"glow", "mono", "classic", "purple", "bold"}:
        theme = "glow"
    if _ensure_db():
        try:
            with _db_connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO portfolios (user_email, title, theme, data, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, NOW(), NOW())
                        ON CONFLICT (user_email) DO UPDATE SET
                            title = EXCLUDED.title,
                            theme = EXCLUDED.theme,
                            data = EXCLUDED.data,
                            updated_at = NOW()
                        RETURNING title, theme, data, updated_at
                    """, (email, title, theme, Json(data)))
                    row = dict(cur.fetchone())
                conn.commit()
            if row.get("updated_at"):
                row["updated_at"] = row["updated_at"].isoformat()
            return row
        except Exception as exc:
            print(f"[Portfolio] DB save failed, using JSON fallback: {exc}")

    portfolios = _load_json(PORTFOLIOS_FILE, {})
    portfolios[email] = {
        "title": title,
        "theme": theme,
        "data": data,
        "updated_at": datetime.now().isoformat(),
    }
    _save_json(PORTFOLIOS_FILE, portfolios)
    return portfolios[email]

# ── Auth decorator ────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            if request.path.startswith("/api/"):
                return jsonify({"success": False, "error": "Not authenticated"}), 401
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated


# ══════════════════════════════════════════════════════
# AUTH ROUTES
# ══════════════════════════════════════════════════════

@app.route("/")
def root():
    if "user" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login_page"))


@app.route("/assets/<path:filename>")
def asset_file(filename):
    return send_from_directory(os.path.join(BASE_DIR, "assets"), filename)

@app.route("/login")
def login_page():
    if "user" in session:
        return redirect(url_for("dashboard"))
    google_enabled = bool(os.getenv("GOOGLE_CLIENT_ID") and os.getenv("GOOGLE_CLIENT_SECRET"))
    return render_template("login.html", google_enabled=google_enabled)

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", user=session["user"])

@app.route("/app")
@login_required
def main_app():
    return render_template("index.html", user=session["user"])

@app.route("/docugenius")
@login_required
def docugenius_page():
    return render_template("docugenius.html", user=session["user"])

@app.route("/skill")
@login_required
def skill_page():
    return render_template("skill.html", user=session["user"])

@app.route("/resume")
@login_required
def resume_page():
    return render_template("resume.html", user=session["user"])

@app.route("/newsletter")
@login_required
def newsletter_page():
    return render_template("newsletter.html", user=session["user"])

@app.route("/domainhire")
@login_required
def domainhire_page():
    return render_template("domainhire.html", user=session["user"])

@app.route("/studyhub")
@login_required
def studyhub_page():
    return render_template("studyhub.html", user=session["user"])

@app.route("/portfolio")
@login_required
def portfolio_page():
    return render_template("portfolio.html", user=session["user"])

@app.route("/api/auth/register", methods=["POST"])
def api_register():
    body = request.get_json(force=True)
    name     = (body.get("name") or "").strip()
    email    = (body.get("email") or "").strip().lower()
    password = (body.get("password") or "").strip()

    if not name or not email or not password:
        return jsonify({"success": False, "error": "Name, email and password are required."}), 400
    if len(password) < 6:
        return jsonify({"success": False, "error": "Password must be at least 6 characters."}), 400
    if "@" not in email:
        return jsonify({"success": False, "error": "Invalid email address."}), 400

    if _get_user(email):
        return jsonify({"success": False, "error": "An account with this email already exists."}), 400

    ok, user, err = _create_password_user(name, email, password)
    if not ok:
        return jsonify({"success": False, "error": err or "Could not create account."}), 400

    _set_session_user(user)
    return jsonify({"success": True, "name": name})

@app.route("/api/auth/login", methods=["POST"])
def api_login():
    body = request.get_json(force=True)
    email    = (body.get("email") or "").strip().lower()
    password = (body.get("password") or "").strip()

    if not email or not password:
        return jsonify({"success": False, "error": "Email and password are required."}), 400

    user = _get_user(email)
    if not user or not user.get("password_hash") or not _check_password(password, user["password_hash"]):
        return jsonify({"success": False, "error": "Invalid email or password."}), 401

    _touch_login(email)
    _set_session_user(user)
    return jsonify({"success": True, "name": user["name"]})

@app.route("/auth/google")
def google_login():
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
    if not client_id or not client_secret:
        return redirect(url_for("login_page", auth_error="Google sign-in needs GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET."))

    state = secrets.token_urlsafe(24)
    session.permanent = True
    session["google_oauth_state"] = state
    params = {
        "client_id": client_id,
        "redirect_uri": url_for("google_callback", _external=True),
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "prompt": "select_account",
    }
    return redirect("https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params))

@app.route("/auth/google/callback")
def google_callback():
    if request.args.get("error"):
        return redirect(url_for("login_page", auth_error=request.args.get("error_description") or request.args["error"]))
    if request.args.get("state") != session.pop("google_oauth_state", None):
        return redirect(url_for("login_page", auth_error="Google sign-in could not be verified. Please try again."))

    code = request.args.get("code")
    if not code:
        return redirect(url_for("login_page", auth_error="Google did not return an authorization code."))

    try:
        token_res = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": os.getenv("GOOGLE_CLIENT_ID"),
                "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
                "redirect_uri": url_for("google_callback", _external=True),
                "grant_type": "authorization_code",
            },
            timeout=12,
        )
        token_res.raise_for_status()
        access_token = token_res.json().get("access_token")
        user_res = requests.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=12,
        )
        user_res.raise_for_status()
        profile = user_res.json()
        if not profile.get("email_verified", True):
            return redirect(url_for("login_page", auth_error="Google email is not verified."))
        user = _upsert_google_user(profile)
        if not user:
            return redirect(url_for("login_page", auth_error="Could not create your Google account."))
        _set_session_user(user)
        return redirect(url_for("dashboard"))
    except Exception as exc:
        return redirect(url_for("login_page", auth_error=f"Google sign-in failed: {exc}"))

@app.route("/api/auth/logout", methods=["POST"])
def api_logout():
    session.clear()
    return jsonify({"success": True})

@app.route("/api/auth/me")
def api_me():
    if "user" in session:
        return jsonify({"authenticated": True, "user": session["user"]})
    return jsonify({"authenticated": False})

@app.route("/api/portfolio", methods=["GET"])
@login_required
def api_get_portfolio():
    portfolio = _get_portfolio(session["user"]["email"])
    return jsonify({"success": True, "portfolio": portfolio})

@app.route("/api/portfolio", methods=["POST"])
@login_required
def api_save_portfolio():
    body = request.get_json(force=True)
    portfolio = body.get("portfolio") or {}
    if not isinstance(portfolio, dict):
        return jsonify({"success": False, "error": "Portfolio data must be an object."}), 400
    title = portfolio.get("name") or portfolio.get("headline") or "My Portfolio"
    theme = (body.get("theme") or portfolio.get("theme") or "glow").strip()
    saved = _save_portfolio(session["user"]["email"], title, theme, portfolio)
    return jsonify({"success": True, "portfolio": saved})


# ══════════════════════════════════════════════════════
# DOCUGENIUS AI — Document Analysis (uses Gemini)
# ══════════════════════════════════════════════════════

@app.route("/api/docugenius/analyze", methods=["POST"])
@login_required
def api_docugenius_analyze():
    """Analyze document text using Gemini (gemini_api_1 / gemini_api_2)."""
    body = request.get_json(force=True)
    text = (body.get("text") or "").strip()
    if not text:
        return jsonify({"success": False, "error": "Document text is required."}), 400
    if len(text) < 50:
        return jsonify({"success": False, "error": "Document is too short to analyze."}), 400
    try:
        analysis = gemini_analyze_document(text)
        return jsonify({"success": True, "analysis": analysis})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ══════════════════════════════════════════════════════
# STUDENT OPPORTUNITY FINDER ROUTES (uses Groq)
# ══════════════════════════════════════════════════════

@app.route("/api/interview", methods=["POST"])
@login_required
def api_interview():
    body = request.get_json(force=True)
    user_input = (body.get("input") or "").strip()
    if not user_input:
        return jsonify({"success": False, "error": "Please describe what you are looking for."}), 400
    try:
        result = interview_agent(user_input)
        return jsonify({"success": True, **result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/verify-skill", methods=["POST"])
@login_required
def api_verify_skill():
    body   = request.get_json(force=True)
    skills = (body.get("skills") or "general programming").strip()
    field  = (body.get("field")  or "engineering").strip()
    try:
        result = verify_skill_agent(skills, field)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/proficiency-report", methods=["POST"])
@login_required
def api_proficiency_report():
    body     = request.get_json(force=True)
    skills   = (body.get("skills")   or "general programming").strip()
    field    = (body.get("field")    or "engineering").strip()
    score    = int(body.get("score", 0))
    answered = body.get("answered", [])
    try:
        result = proficiency_report_agent(skills, field, score, answered)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/research", methods=["POST"])
@login_required
def api_research():
    body       = request.get_json(force=True)
    profile    = body.get("profile", {})
    categories = body.get("categories", ["Hackathon", "Internship", "Scholarship", "Competition"])
    if not profile:
        return jsonify({"success": False, "error": "Profile is required."}), 400
    try:
        research = research_agent(profile, categories)
        report   = report_agent(profile, research["results"])
        os.makedirs("output", exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"output/report_{ts}.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        return jsonify({"success": True, "report": report, "logs": research["logs"], "file": path})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ══════════════════════════════════════════════════════
# PDF GENERATORS
# ══════════════════════════════════════════════════════

def generate_pdf(report: dict) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from reportlab.lib.enums import TA_LEFT, TA_CENTER

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()

    style_title    = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=22, textColor=colors.HexColor("#0a0a0f"), spaceAfter=4, leading=26)
    style_subtitle = ParagraphStyle("subtitle", fontName="Helvetica", fontSize=10, textColor=colors.HexColor("#78716c"), spaceAfter=2)
    style_summary  = ParagraphStyle("summary", fontName="Helvetica", fontSize=10, textColor=colors.HexColor("#1c1917"), backColor=colors.HexColor("#f5f3ee"), borderPad=8, leading=16, spaceAfter=6)
    style_section  = ParagraphStyle("section", fontName="Helvetica-Bold", fontSize=13, textColor=colors.HexColor("#0a0a0f"), spaceBefore=14, spaceAfter=6)
    style_opp_title= ParagraphStyle("opp_title", fontName="Helvetica-Bold", fontSize=10, textColor=colors.HexColor("#0a0a0f"), spaceAfter=2)
    style_opp_detail=ParagraphStyle("opp_detail", fontName="Helvetica", fontSize=9, textColor=colors.HexColor("#44403c"), leading=14, spaceAfter=2)
    style_why      = ParagraphStyle("why", fontName="Helvetica-Oblique", fontSize=9, textColor=colors.HexColor("#92400e"), backColor=colors.HexColor("#fffbeb"), borderPad=5, leading=13, spaceAfter=4)
    style_link     = ParagraphStyle("link", fontName="Helvetica", fontSize=9, textColor=colors.HexColor("#1a56db"), spaceAfter=6)
    style_pick     = ParagraphStyle("pick", fontName="Helvetica-Bold", fontSize=10, textColor=colors.HexColor("#ff5c00"), spaceAfter=2)
    style_action   = ParagraphStyle("action", fontName="Helvetica", fontSize=9, textColor=colors.HexColor("#1c1917"), leading=15, spaceAfter=3)

    story = []
    now = datetime.now().strftime("%d %B %Y, %I:%M %p")
    story.append(Paragraph("🎓 Student Opportunity Report", style_title))
    story.append(Paragraph(f"Generated on {now}  ·  Powered by Agentic AI", style_subtitle))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#0a0a0f"), spaceAfter=10))

    summary = report.get("student_summary", "")
    if summary:
        story.append(Paragraph(f"<b>Profile:</b> {summary}", style_summary))
        story.append(Spacer(1, 6))

    total = report.get("total_opportunities", 0)
    cats  = len(report.get("categories", []))
    picks = len(report.get("top_picks", []))
    stats_data = [[f"{total}\nOpportunities Found", f"{cats}\nCategories", f"{picks}\nTop Picks"]]
    stats_table = Table(stats_data, colWidths=[5.5*cm, 5.5*cm, 5.5*cm])
    stats_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,-1), colors.HexColor("#ff5c00")),
        ("TEXTCOLOR",  (0,0),(-1,-1), colors.white),
        ("FONTNAME",   (0,0),(-1,-1), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0),(-1,-1), 11),
        ("ALIGN",      (0,0),(-1,-1), "CENTER"),
        ("VALIGN",     (0,0),(-1,-1), "MIDDLE"),
        ("ROWHEIGHT",  (0,0),(-1,-1), 36),
        ("GRID",       (0,0),(-1,-1), 1, colors.white),
    ]))
    story.append(stats_table)
    story.append(Spacer(1, 14))

    top_picks = report.get("top_picks", [])
    if top_picks:
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#d6d0c4"), spaceAfter=6))
        story.append(Paragraph("⭐  Top Picks For You", style_section))
        for p in top_picks:
            story.append(Paragraph(f"#{p.get('rank','')}  {p.get('title','')}", style_pick))
            story.append(Paragraph(p.get("reason",""), style_opp_detail))
            link = p.get("apply_link","#")
            story.append(Paragraph(f'Apply → <a href="{link}" color="#1a56db">{link}</a>', style_link))
            story.append(Spacer(1, 4))

    for cat in report.get("categories", []):
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#d6d0c4"), spaceAfter=6))
        emoji = cat.get("emoji","📌")
        name  = cat.get("name","")
        count = cat.get("count", len(cat.get("opportunities",[])))
        story.append(Paragraph(f"{emoji}  {name}  ({count} found)", style_section))
        for opp in cat.get("opportunities", []):
            story.append(Paragraph(opp.get("title","Opportunity"), style_opp_title))
            meta = [
                ["Organizer", opp.get("organizer","—"), "Deadline", opp.get("deadline","Check website")],
                ["Prize/Stipend", opp.get("stipend_prize","N/A"), "Difficulty", opp.get("difficulty","—")],
                ["Eligibility", opp.get("eligibility","—"), "", ""],
            ]
            meta_table = Table(meta, colWidths=[2.8*cm, 5*cm, 2.8*cm, 5*cm])
            meta_table.setStyle(TableStyle([
                ("FONTNAME",(0,0),(-1,-1),"Helvetica"), ("FONTSIZE",(0,0),(-1,-1),8),
                ("TEXTCOLOR",(0,0),(0,-1),colors.HexColor("#78716c")),
                ("TEXTCOLOR",(2,0),(2,-1),colors.HexColor("#78716c")),
                ("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),
                ("FONTNAME",(2,0),(2,-1),"Helvetica-Bold"),
                ("VALIGN",(0,0),(-1,-1),"TOP"),
                ("TOPPADDING",(0,0),(-1,-1),2), ("BOTTOMPADDING",(0,0),(-1,-1),2),
            ]))
            story.append(meta_table)
            why = opp.get("why_suitable","")
            if why:
                story.append(Paragraph(f"💡 {why}", style_why))
            link = opp.get("apply_link","#")
            story.append(Paragraph(f'🔗 Apply: <a href="{link}" color="#1a56db">{link}</a>', style_link))
            story.append(Spacer(1, 6))

    action_plan = report.get("action_plan", [])
    if action_plan:
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#0a0a0f"), spaceAfter=8))
        story.append(Paragraph("🗺  Your Personal Action Plan", style_section))
        for step in action_plan:
            story.append(Paragraph(f"  ▸  {step}", style_action))

    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#d6d0c4"), spaceAfter=4))
    story.append(Paragraph("Generated by Student Opportunity Finder · Agentic AI System · 3-Agent Pipeline", style_subtitle))
    doc.build(story)
    buffer.seek(0)
    return buffer.read()


def generate_prof_pdf(report: dict, score: int, profile: dict) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)

    s_title   = ParagraphStyle("pt", fontName="Helvetica-Bold", fontSize=20, textColor=colors.HexColor("#0a0a0f"), spaceAfter=4, leading=24)
    s_sub     = ParagraphStyle("ps", fontName="Helvetica", fontSize=9, textColor=colors.HexColor("#78716c"), spaceAfter=2)
    s_section = ParagraphStyle("pse", fontName="Helvetica-Bold", fontSize=12, textColor=colors.HexColor("#0a0a0f"), spaceBefore=14, spaceAfter=6)
    s_body    = ParagraphStyle("pb", fontName="Helvetica", fontSize=9, textColor=colors.HexColor("#1c1917"), leading=15, spaceAfter=4)

    story = []
    now   = datetime.now().strftime("%d %B %Y, %I:%M %p")
    field = profile.get("What is your field of study?", report.get("domain_title", "Domain"))

    story.append(Paragraph("📊 Domain Proficiency Report", s_title))
    story.append(Paragraph(f"Generated on {now}  ·  Powered by CareerAI Skill Verification", s_sub))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#0a0a0f"), spaceAfter=10))

    level  = report.get("level", "Intermediate")
    lv_col = {"Expert":"#059669","Proficient":"#d97706","Intermediate":"#ff5c00","Beginner":"#dc2626"}.get(level,"#ff5c00")
    banner = [[f"Score: {score}%", level, report.get("domain_title", field)]]
    bt = Table(banner, colWidths=[5.5*cm, 5.5*cm, 5.5*cm])
    bt.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(0,0), colors.HexColor("#0a0a0f")),
        ("BACKGROUND",(1,0),(1,0), colors.HexColor(lv_col)),
        ("BACKGROUND",(2,0),(2,0), colors.HexColor("#f5f3ee")),
        ("TEXTCOLOR",(0,0),(1,0), colors.white),
        ("TEXTCOLOR",(2,0),(2,0), colors.HexColor("#0a0a0f")),
        ("FONTNAME",(0,0),(-1,-1),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1),11),
        ("ALIGN",(0,0),(-1,-1),"CENTER"),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("ROWHEIGHT",(0,0),(-1,-1),34),
        ("GRID",(0,0),(-1,-1),1,colors.white),
    ]))
    story.append(bt)
    story.append(Spacer(1, 10))

    summary = report.get("efficiency_summary", "")
    if summary:
        story.append(Paragraph(f"<b>Summary:</b> {summary}", s_body))
        story.append(Paragraph(f"<b>Career Readiness:</b> {report.get('career_readiness','')}", s_body))
        story.append(Spacer(1, 6))

    skill_scores = report.get("skill_scores", [])
    if skill_scores:
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#d6d0c4"), spaceAfter=6))
        story.append(Paragraph("Skill Breakdown", s_section))
        ss_data = [["Skill Area", "Score", "Rating"]] + [
            [s["skill"], f"{s['score']}%", "Strong" if s["score"] >= 70 else "Moderate" if s["score"] >= 40 else "Needs Work"]
            for s in skill_scores
        ]
        ss_table = Table(ss_data, colWidths=[8*cm, 3*cm, 5.5*cm])
        ss_table.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0), colors.HexColor("#0a0a0f")),
            ("TEXTCOLOR",(0,0),(-1,0), colors.white),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("FONTNAME",(0,1),(-1,-1),"Helvetica"),
            ("FONTSIZE",(0,0),(-1,-1),9),
            ("ROWHEIGHT",(0,0),(-1,-1),22),
            ("ALIGN",(1,0),(-1,-1),"CENTER"),
            ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#d6d0c4")),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, colors.HexColor("#f5f3ee")]),
        ]))
        story.append(ss_table)
        story.append(Spacer(1, 8))

    strengths = report.get("strengths", [])
    gaps      = report.get("gaps", [])
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#d6d0c4"), spaceAfter=6))
    str_items = "\n".join([f"  ✔  {s}" for s in strengths])
    gap_items = "\n".join([f"  ✘  {g}" for g in gaps])
    sg_data = [["✅  Strengths", "⚠️  Gaps to Address"], [str_items or "—", gap_items or "—"]]
    sg_table = Table(sg_data, colWidths=[8.25*cm, 8.25*cm])
    sg_table.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(0,0), colors.HexColor("#f0fdf4")),
        ("BACKGROUND",(1,0),(1,0), colors.HexColor("#fff5f5")),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1),9),
        ("TEXTCOLOR",(0,0),(0,0), colors.HexColor("#059669")),
        ("TEXTCOLOR",(1,0),(1,0), colors.HexColor("#dc2626")),
        ("VALIGN",(0,0),(-1,-1),"TOP"),
        ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#d6d0c4")),
        ("TOPPADDING",(0,0),(-1,-1),8), ("BOTTOMPADDING",(0,0),(-1,-1),8),
        ("LEFTPADDING",(0,0),(-1,-1),10),
    ]))
    story.append(sg_table)
    story.append(Spacer(1, 10))

    recs = report.get("recommendations", [])
    if recs:
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#d6d0c4"), spaceAfter=6))
        story.append(Paragraph("🎯  Recommendations", s_section))
        for i, rec in enumerate(recs, 1):
            story.append(Paragraph(f"  {i}.  {rec}", s_body))

    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#d6d0c4"), spaceAfter=4))
    story.append(Paragraph("Generated by CareerAI · Skill Verification System", s_sub))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


@app.route("/api/download-pdf", methods=["POST"])
@login_required
def api_download_pdf():
    body   = request.get_json(force=True)
    report = body.get("report")
    if not report:
        return jsonify({"error": "No report data provided"}), 400
    try:
        pdf_bytes = generate_pdf(report)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return send_file(io.BytesIO(pdf_bytes), mimetype="application/pdf", as_attachment=True, download_name=f"opportunity_report_{ts}.pdf")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/download-prof-pdf", methods=["POST"])
@login_required
def api_download_prof_pdf():
    body    = request.get_json(force=True)
    report  = body.get("report", {})
    score   = int(body.get("score", 0))
    profile = body.get("profile", {})
    if not report:
        return jsonify({"error": "No report data provided"}), 400
    try:
        pdf_bytes = generate_prof_pdf(report, score, profile)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return send_file(io.BytesIO(pdf_bytes), mimetype="application/pdf", as_attachment=True, download_name=f"proficiency_report_{ts}.pdf")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/download/<path:filename>")
@login_required
def api_download(filename):
    path = os.path.join(os.path.dirname(__file__), filename)
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    return send_file(path, as_attachment=True)


# ══════════════════════════════════════════════════════
# RESUME PDF GENERATOR
# ══════════════════════════════════════════════════════

def generate_resume_pdf(resume: dict, template: str = "classic") -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    W = A4[0] - 4*cm

    if template == "modern":
        hdr_bg = colors.HexColor("#0a0a0f"); acc = colors.HexColor("#059669"); acc2 = colors.HexColor("#34d399"); hdr_fg = colors.white; sec_fg = colors.HexColor("#0a0a0f")
    elif template == "minimal":
        hdr_bg = colors.HexColor("#f9f7f4"); acc = colors.HexColor("#1a1a1a"); acc2 = colors.HexColor("#888888"); hdr_fg = colors.HexColor("#1a1a1a"); sec_fg = colors.HexColor("#999999")
    else:
        hdr_bg = None; acc = colors.HexColor("#0a0a0f"); acc2 = colors.HexColor("#1a56db"); hdr_fg = colors.HexColor("#0a0a0f"); sec_fg = colors.HexColor("#888888")

    p          = resume.get("personal", {})
    education  = resume.get("education", [])
    experience = resume.get("experience", [])
    skills     = resume.get("skills", {})
    projects   = resume.get("projects", [])
    certs      = resume.get("certs", [])
    awards     = resume.get("awards", [])
    languages  = resume.get("languages", [])
    volunteer  = resume.get("volunteer", [])

    s_name   = ParagraphStyle("rn", fontName="Helvetica-Bold", fontSize=22, leading=26, textColor=hdr_fg if template != "classic" else acc, spaceAfter=3)
    s_title_p= ParagraphStyle("rt", fontName="Helvetica", fontSize=10, textColor=colors.HexColor("#666666"), spaceAfter=4)
    s_contact= ParagraphStyle("rc", fontName="Helvetica", fontSize=8, textColor=colors.HexColor("#555555"), spaceAfter=2, leading=12)
    s_sec    = ParagraphStyle("rs", fontName="Helvetica-Bold", fontSize=8, textColor=sec_fg, spaceBefore=10, spaceAfter=5, leading=11, letterSpacing=1.5)
    s_body   = ParagraphStyle("rb", fontName="Helvetica", fontSize=9, textColor=colors.HexColor("#333333"), leading=14, spaceAfter=2)
    s_bold   = ParagraphStyle("rbd", fontName="Helvetica-Bold", fontSize=9.5, textColor=colors.HexColor("#0a0a0f"), leading=13, spaceAfter=1)
    s_muted  = ParagraphStyle("rm", fontName="Helvetica", fontSize=8, textColor=colors.HexColor("#777777"), leading=11, spaceAfter=2)
    s_blue   = ParagraphStyle("rbl", fontName="Helvetica", fontSize=8.5, textColor=acc2, leading=12, spaceAfter=2)
    s_bullet = ParagraphStyle("rbul", fontName="Helvetica", fontSize=8.5, textColor=colors.HexColor("#333333"), leading=13, leftIndent=12, spaceAfter=1)

    story = []
    def hr(color=colors.HexColor("#d6d0c4"), thick=1):
        return HRFlowable(width="100%", thickness=thick, color=color, spaceAfter=6)
    def sec_title(txt):
        story.append(hr()); story.append(Paragraph(txt.upper(), s_sec))

    name = p.get("name") or "Your Name"
    story.append(Paragraph(name, s_name))
    if p.get("title"): story.append(Paragraph(p["title"], s_title_p))
    contact_parts = [x for x in [p.get("email"), p.get("phone"), p.get("location"), p.get("linkedin"), p.get("github")] if x]
    if contact_parts: story.append(Paragraph(" · ".join(contact_parts), s_contact))
    story.append(hr(acc, 2))

    if p.get("summary"): sec_title("Summary"); story.append(Paragraph(p["summary"], s_body))

    if experience:
        sec_title("Experience")
        for exp in experience:
            title_txt = exp.get("title",""); company = exp.get("company",""); duration = exp.get("duration","")
            if title_txt or company:
                row = [[Paragraph(title_txt, s_bold), Paragraph(duration, ParagraphStyle("rd", fontName="Helvetica", fontSize=8, textColor=colors.HexColor("#777777"), leading=11, alignment=TA_RIGHT))]]
                t = Table(row, colWidths=[W*0.7, W*0.3])
                t.setStyle(TableStyle([("TOPPADDING",(0,0),(-1,-1),0),("BOTTOMPADDING",(0,0),(-1,-1),1)]))
                story.append(t)
            if company: story.append(Paragraph(company, s_blue))
            for b in exp.get("bullets", []):
                if b: story.append(Paragraph("• " + b, s_bullet))
            story.append(Spacer(1, 5))

    if education:
        sec_title("Education")
        for edu in education:
            degree = edu.get("degree",""); inst = edu.get("institution",""); year = edu.get("year",""); gpa = edu.get("gpa","")
            if degree:
                row = [[Paragraph(degree, s_bold), Paragraph(year, ParagraphStyle("ry", fontName="Helvetica", fontSize=8, textColor=colors.HexColor("#777777"), leading=11, alignment=TA_RIGHT))]]
                t = Table(row, colWidths=[W*0.7, W*0.3])
                t.setStyle(TableStyle([("TOPPADDING",(0,0),(-1,-1),0),("BOTTOMPADDING",(0,0),(-1,-1),1)]))
                story.append(t)
            if inst: story.append(Paragraph(inst, s_blue))
            extras = []
            if gpa: extras.append("GPA: " + gpa)
            if edu.get("courses"): extras.append("Courses: " + edu["courses"])
            if extras: story.append(Paragraph(" · ".join(extras), s_muted))
            story.append(Spacer(1, 4))

    tech = skills.get("tech", []); soft = skills.get("soft", []); all_skills = tech + soft
    if all_skills or projects:
        sec_title("Skills & Projects")
        left_items = []
        if tech: left_items.append(Paragraph("Technical: " + ", ".join(tech), s_body))
        if soft: left_items.append(Paragraph("Soft Skills: " + ", ".join(soft), s_muted))
        right_items = []
        for proj in projects:
            if proj.get("title"): right_items.append(Paragraph(proj["title"], s_bold))
            if proj.get("tech"): right_items.append(Paragraph(proj["tech"], s_muted))
            if proj.get("description"): right_items.append(Paragraph(proj["description"], ParagraphStyle("pd",fontName="Helvetica",fontSize=8,textColor=colors.HexColor("#444444"),leading=12,spaceAfter=6)))
        if left_items or right_items:
            max_len = max(len(left_items), len(right_items))
            while len(left_items) < max_len: left_items.append(Spacer(1,1))
            while len(right_items) < max_len: right_items.append(Spacer(1,1))
            tdata = [[left_items, right_items]]
            t = Table(tdata, colWidths=[W*0.48, W*0.48])
            t.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),("TOPPADDING",(0,0),(-1,-1),0),("BOTTOMPADDING",(0,0),(-1,-1),0),("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),8)]))
            story.append(t)

    if certs:
        sec_title("Certifications")
        for cert in certs:
            name_txt = cert.get("name",""); issuer = cert.get("issuer",""); year = cert.get("year","")
            if name_txt:
                row = [[Paragraph(name_txt + (" — " + issuer if issuer else ""), s_body), Paragraph(year, s_muted)]]
                t = Table(row, colWidths=[W*0.8, W*0.2])
                t.setStyle(TableStyle([("TOPPADDING",(0,0),(-1,-1),0),("BOTTOMPADDING",(0,0),(-1,-1),2),("ALIGN",(1,0),(1,-1),"RIGHT")]))
                story.append(t)

    doc.build(story)
    buffer.seek(0)
    return buffer.read()

@app.route("/api/resume-pdf", methods=["POST"])
@login_required
def api_resume_pdf():
    body     = request.get_json(force=True)
    resume   = body.get("resume", {})
    template = body.get("template", "classic")
    if not resume:
        return jsonify({"error": "No resume data provided"}), 400
    if not isinstance(resume.get("skills"), dict):
        resume["skills"] = {"tech": [], "soft": []}
    resume["skills"].setdefault("tech", [])
    resume["skills"].setdefault("soft", [])
    for key in ["education", "experience", "projects", "certs", "awards", "languages", "volunteer"]:
        if not isinstance(resume.get(key), list):
            resume[key] = []
    if not isinstance(resume.get("personal"), dict):
        resume["personal"] = {}
    try:
        pdf_bytes = generate_resume_pdf(resume, template)
        name = (resume.get("personal", {}).get("name") or "resume").replace(" ", "_")
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        return send_file(io.BytesIO(pdf_bytes), mimetype="application/pdf", as_attachment=True, download_name=f"resume_{name}_{ts}.pdf")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════
# AI DAILY — News Engine (uses Gemini)
# ══════════════════════════════════════════════════════

import hashlib

AI_DAILY_DIR = os.path.join(os.path.dirname(__file__), "output", "ai_daily")
os.makedirs(AI_DAILY_DIR, exist_ok=True)

RSS_FEEDS = [
    {"name": "TechCrunch AI",   "url": "https://techcrunch.com/category/artificial-intelligence/feed/"},
    {"name": "MIT Tech Review", "url": "https://www.technologyreview.com/topic/artificial-intelligence/feed/"},
    {"name": "The Verge AI",    "url": "https://www.theverge.com/ai-artificial-intelligence/rss/index.xml"},
    {"name": "VentureBeat AI",  "url": "https://venturebeat.com/category/ai/feed/"},
]

@app.route("/api/raw-news")
@login_required
def api_raw_news():
    import re as _re
    import urllib.request
    UA = "Mozilla/5.0 (compatible; StudentPlatform/1.0; +https://github.com)"
    articles = []
    for feed in RSS_FEEDS:
        try:
            req = urllib.request.Request(feed["url"], headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw_xml = resp.read()
            parsed = feedparser.parse(raw_xml)
            for item in parsed.entries[:5]:
                content = (item.get("summary", "") or (item.get("content") or [{}])[0].get("value", "") or "")
                content = _re.sub(r"<[^>]+>", " ", content).strip()
                content = " ".join(content.split())[:400]
                articles.append({"title": item.get("title",""), "content": content, "link": item.get("link","#"), "pubDate": item.get("published",""), "source": feed["name"]})
        except Exception as e:
            print(f"RSS error [{feed['name']}]: {e}")
    return jsonify(articles)

@app.route("/api/ai-daily/newsletters")
@login_required
def api_list_newsletters():
    files = sorted([f for f in os.listdir(AI_DAILY_DIR) if f.endswith(".json") and f.startswith("nl_")], reverse=True)
    newsletters = []
    for f in files[:20]:
        try:
            with open(os.path.join(AI_DAILY_DIR, f)) as fh:
                nl = json.load(fh)
                newsletters.append({"id": nl.get("id"), "date": nl.get("date"), "insight": nl.get("insight",""), "count": len(nl.get("articles",[]))})
        except Exception:
            pass
    return jsonify(newsletters)

@app.route("/api/ai-daily/newsletter/<nl_id>")
@login_required
def api_get_newsletter(nl_id):
    nl_id = nl_id.replace("..", "").replace("/", "")
    path = os.path.join(AI_DAILY_DIR, f"nl_{nl_id}.json")
    if not os.path.exists(path):
        return jsonify({"error": "Not found"}), 404
    with open(path) as f:
        return jsonify(json.load(f))

@app.route("/api/ai-daily/generate", methods=["POST"])
@login_required
def api_generate_newsletter():
    """Generate today's AI Daily newsletter using Gemini (gemini_api_1 / gemini_api_2)."""
    if not GEMINI_KEYS:
        return jsonify({"success": False, "error": "GEMINI_API_1 or GEMINI_API_KEY not set in environment"}), 500

    import re as _re
    import urllib.request as _ur
    import requests as _req

    UA = "Mozilla/5.0 (compatible; StudentPlatform/1.0)"
    try:
        raw_articles = []
        for feed in RSS_FEEDS:
            try:
                req = _ur.Request(feed["url"], headers={"User-Agent": UA})
                with _ur.urlopen(req, timeout=10) as resp:
                    raw_xml = resp.read()
                parsed = feedparser.parse(raw_xml)
                for item in parsed.entries[:4]:
                    content = _re.sub(r"<[^>]+>", " ", (item.get("summary","") or "")).strip()
                    content = " ".join(content.split())[:300]
                    raw_articles.append({"title": item.get("title",""), "content": content, "link": item.get("link","#"), "pubDate": item.get("published",""), "source": feed["name"]})
            except Exception as fe:
                print(f"Feed error [{feed['name']}]: {fe}")
    except Exception as e:
        return jsonify({"success": False, "error": f"RSS fetch failed: {e}"}), 500

    if not raw_articles:
        return jsonify({"success": False, "error": "No articles fetched from RSS feeds"}), 500

    summarized = []
    for art in raw_articles[:6]:
        prompt = f"""Summarize this AI news article in JSON with keys:
"headline" (max 10 words), "bulletPoints" (array of 3 strings), "whyItMatters" (1-2 sentences), "category" (one of: LLMs, Robotics, Startups, Ethics, Research, Tools, Other).
Return ONLY the JSON object, no markdown fences, no extra text.

Title: {art['title']}
Content: {art['content']}"""
        try:
            text = gemini(prompt)
            text = text.replace("```json","").replace("```","").strip()
            summary = json.loads(text)
            summarized.append({**summary, "sourceUrl": art["link"], "sourceName": art["source"], "publishedAt": art["pubDate"]})
        except Exception as e:
            print(f"Summarize error: {e}")

    if not summarized:
        return jsonify({"success": False, "error": "Failed to summarize articles"}), 500

    headlines = "\n".join([a.get("headline","") for a in summarized])
    insight_prompt = f"""Based on these AI headlines, write one short insightful trend observation in exactly 2 sentences. No quotes. Just plain text.\n\nHeadlines:\n{headlines}"""
    try:
        insight = gemini(insight_prompt)
    except Exception:
        insight = "AI continues to advance rapidly across multiple domains today."

    nl_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    nl_date = datetime.now().strftime("%Y-%m-%d")
    newsletter = {"id": nl_id, "date": nl_date, "insight": insight, "articles": summarized, "createdAt": datetime.now().isoformat()}
    path = os.path.join(AI_DAILY_DIR, f"nl_{nl_id}.json")
    with open(path, "w") as f:
        json.dump(newsletter, f, indent=2)

    # Email broadcast (unchanged logic)
    broadcast_count = 0
    broadcast_errors = []
    try:
        import smtplib, ssl
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        email_user = os.getenv("EMAIL_USER") or os.getenv("SMTP_USER","")
        email_pass = os.getenv("EMAIL_PASS") or os.getenv("SMTP_PASS","")
        if email_user and email_pass:
            subs_path = os.path.join(AI_DAILY_DIR, "subscriptions.json")
            subs = []
            if os.path.exists(subs_path):
                with open(subs_path) as f:
                    try: subs = json.load(f)
                    except: subs = []
            if subs:
                ssl_ctx = ssl.create_default_context()
                with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl_ctx) as server:
                    server.login(email_user, email_pass)
                    for sub in subs:
                        sub_email = sub.get("email","")
                        if not sub_email: continue
                        try:
                            msg = MIMEMultipart("alternative")
                            msg["Subject"] = f"🤖 AI Daily — {nl_date} Edition ({len(summarized)} stories)"
                            msg["From"] = f"AI Daily <{email_user}>"
                            msg["To"] = sub_email
                            msg.attach(MIMEText(f"<h2>AI Daily — {nl_date}</h2><p>{insight}</p>", "html"))
                            server.sendmail(email_user, sub_email, msg.as_string())
                            broadcast_count += 1
                        except Exception as sub_err:
                            broadcast_errors.append(sub_email)
    except Exception as broadcast_err:
        print(f"[Newsletter] Broadcast error: {broadcast_err}")

    return jsonify({"success": True, "newsletter": newsletter, "broadcast": {"sent": broadcast_count, "failed": len(broadcast_errors)}})

@app.route("/api/ai-daily/subscribe", methods=["POST"])
@login_required
def api_ai_daily_subscribe():
    body   = request.get_json(force=True)
    email  = (body.get("email") or "").strip()
    topics = body.get("topics", [])
    if not email or "@" not in email:
        return jsonify({"success": False, "error": "Invalid email"}), 400
    subs_path = os.path.join(AI_DAILY_DIR, "subscriptions.json")
    subs = []
    if os.path.exists(subs_path):
        with open(subs_path) as f:
            try: subs = json.load(f)
            except: subs = []
    is_new = not any(s.get("email") == email for s in subs)
    if is_new:
        subs.append({"email": email, "topics": topics, "subscribedAt": datetime.now().isoformat()})
        with open(subs_path, "w") as f:
            json.dump(subs, f, indent=2)
    return jsonify({"success": True, "is_new": is_new})


# ══════════════════════════════════════════════════════
# HEALTH CHECK (for Render)
# ══════════════════════════════════════════════════════
@app.route("/health")
def health():
    db_ready = _ensure_db()
    return jsonify({
        "status": "ok",
        "groq_keys": len(GROQ_KEYS),
        "gemini_keys": len(GEMINI_KEYS),
        "database": "connected" if db_ready else "fallback",
        "database_error": None if db_ready else _db_error,
        "timestamp": datetime.now().isoformat()
    })


if __name__ == "__main__":
    if not GROQ_KEYS:
        print("\n⚠️  No GROQ API keys set! Set GROQ_API_1 or GROQ_API_KEY in .env")
    else:
        print(f"\n✅ {len(GROQ_KEYS)} Groq key(s) loaded")
    if not GEMINI_KEYS:
        print("⚠️  No GEMINI API keys set! Set GEMINI_API_1 or GEMINI_API_KEY in .env")
    else:
        print(f"✅ {len(GEMINI_KEYS)} Gemini key(s) loaded")

    os.makedirs("output", exist_ok=True)
    print("Student Sarthi -> http://localhost:5000\n")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
