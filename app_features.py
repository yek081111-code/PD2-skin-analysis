import hashlib
import json
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from PIL import Image

from skin_check import (
    DEFAULT_MODEL,
    SYSTEM,
    SCHEMA_PROMPT,
    AppError,
    call_llm,
    parse_analysis,
    prepare_image,
)

DB_PATH = Path(".data")
DB_PATH.mkdir(exist_ok=True)
LOCAL_DB = DB_PATH / "skincheck.db"


def get_supabase():
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
    except Exception:
        return None
    try:
        from supabase import create_client
        return create_client(url, key)
    except Exception as exc:
        st.warning(f"Supabase is unavailable: {exc}")
        return None


def local_connect():
    connection = sqlite3.connect(LOCAL_DB)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            model TEXT NOT NULL,
            cleanliness REAL,
            quality REAL,
            age_range TEXT,
            description TEXT,
            limitations TEXT,
            morning TEXT NOT NULL,
            evening TEXT NOT NULL,
            preferences TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)
    return connection


def password_hash(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 120_000).hex()
    return f"{salt}${digest}"


def password_matches(password, stored_hash):
    try:
        salt, digest = stored_hash.split("$", 1)
    except ValueError:
        return False
    return secrets.compare_digest(password_hash(password, salt).split("$", 1)[1], digest)


def local_register(email, password):
    connection = local_connect()
    try:
        cursor = connection.execute(
            "INSERT INTO users(email, password_hash, created_at) VALUES (?, ?, ?)",
            (email.lower(), password_hash(password), datetime.now(timezone.utc).isoformat()),
        )
        connection.commit()
        return {"mode": "local", "id": cursor.lastrowid, "email": email.lower()}
    except sqlite3.IntegrityError:
        raise ValueError("This email is already registered.")
    finally:
        connection.close()


def local_login(email, password):
    connection = local_connect()
    row = connection.execute(
        "SELECT id, email, password_hash FROM users WHERE email = ?", (email.lower(),)
    ).fetchone()
    connection.close()
    if not row or not password_matches(password, row[2]):
        raise ValueError("Incorrect email or password.")
    return {"mode": "local", "id": str(row[0]), "email": row[1]}


def resolve_local_user_id(user):
    try:
        return int(user["id"])
    except (TypeError, ValueError):
        connection = local_connect()
        row = connection.execute(
            "SELECT id FROM users WHERE email = ?", (user["email"].lower(),)
        ).fetchone()
        connection.close()
        if row is None:
            raise ValueError("Local account could not be found. Please sign in again.")
        return int(row[0])


def assessment_to_row(result, preferences):
    return {
        "timestamp": result["timestamp"],
        "model": result["model"],
        "cleanliness": result["cleanliness"],
        "quality": result["quality"],
        "age_range": result["age_range"],
        "description": result["description"],
        "limitations": result["limitations"],
        "morning": json.dumps(result["morning"]),
        "evening": json.dumps(result["evening"]),
        "preferences": json.dumps(preferences),
    }


def save_assessment(user, result, preferences):
    supabase = st.session_state.supabase
    row = assessment_to_row(result, preferences)
    if supabase is not None and user["mode"] == "supabase":
        row["user_id"] = user["id"]
        supabase.table("assessments").insert(row).execute()
    else:
        connection = local_connect()
        connection.execute("""
            INSERT INTO assessments
            (user_id, timestamp, model, cleanliness, quality, age_range, description, limitations, morning, evening, preferences)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            resolve_local_user_id(user), row["timestamp"], row["model"], row["cleanliness"], row["quality"],
            row["age_range"], row["description"], row["limitations"], row["morning"], row["evening"], row["preferences"],
        ))
        connection.commit()
        connection.close()


def load_assessments(user):
    supabase = st.session_state.supabase
    if supabase is not None and user["mode"] == "supabase":
        response = (
            supabase.table("assessments")
            .select("*")
            .eq("user_id", user["id"])
            .order("timestamp")
            .execute()
        )
        rows = response.data or []
    else:
        connection = local_connect()
        rows = [
            dict(zip(
                ["id", "timestamp", "model", "cleanliness", "quality", "age_range",
                 "description", "limitations", "morning", "evening", "preferences"],
                row,
            ))
            for row in connection.execute("""
                SELECT id, timestamp, model, cleanliness, quality, age_range,
                       description, limitations, morning, evening, preferences
                FROM assessments WHERE user_id = ? ORDER BY timestamp
            """, (resolve_local_user_id(user),)).fetchall()
        ]
        connection.close()

    records = []
    for row in rows:
        record = dict(row)
        for key in ("morning", "evening"):
            if isinstance(record.get(key), str):
                try:
                    record[key] = json.loads(record[key])
                except json.JSONDecodeError:
                    record[key] = []
        if isinstance(record.get("preferences"), str):
            try:
                record["preferences"] = json.loads(record["preferences"])
            except json.JSONDecodeError:
                record["preferences"] = {}
        records.append(record)
    return records


def clear_assessments(user):
    supabase = st.session_state.supabase
    if supabase is not None and user["mode"] == "supabase":
        supabase.table("assessments").delete().eq("user_id", user["id"]).execute()
    else:
        connection = local_connect()
        connection.execute(
            "DELETE FROM assessments WHERE user_id = ?",
            (resolve_local_user_id(user),),
        )
        connection.commit()
        connection.close()


def personalised_recommendations(result, skin_type, concerns):
    recommendations = []
    concern_text = concerns.lower()
    sensitivity_words = ("fragrance", "alcohol", "sensitive", "irritation", "reaction")

    recommendations.append("Use a gentle, fragrance-free cleanser; avoid rubbing or scrubbing.")
    recommendations.append("Apply a broad-spectrum SPF 30–50 sunscreen every morning and reapply outdoors.")
    if skin_type == "Dry":
        recommendations.append("Choose a simple ceramide- or glycerin-based moisturiser and reduce hot-water exposure.")
    elif skin_type == "Oily":
        recommendations.append("Use a lightweight, non-comedogenic moisturiser and avoid excessive layering.")
    elif skin_type == "Combination":
        recommendations.append("Use a lightweight moisturiser overall and consider a richer texture only on dry areas.")
    elif skin_type == "Sensitive":
        recommendations.append("Patch-test new products and introduce one product at a time.")
    if any(word in concern_text for word in sensitivity_words):
        recommendations.append("Avoid known irritants and prioritise a short fragrance-free routine.")
    if isinstance(result["quality"], (int, float)) and result["quality"] < 6:
        recommendations.append("Track changes weekly under similar lighting before changing many products.")
    if len(recommendations) > 6:
        recommendations = recommendations[:6]
    return recommendations

st.set_page_config(page_title="Skin Analysis", page_icon="◎", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem;}
[data-testid="stHeader"] {background: transparent;}
[data-testid="stToolbar"] {visibility: hidden;}
[data-testid="stMetric"] {background: #f5f7f2; border: 1px solid #e1e7dc; border-radius: 12px; padding: 16px;}
[data-testid="stMetricLabel"] {font-size: 0.95rem !important;}
.small-note {color: #65726b; font-size: 0.82rem; line-height: 1.5;}
.disclaimer {border-left: 3px solid #b7c8a7; padding: 10px 14px; background: #f8f9f3; color: #586754; font-size: 0.82rem;}
.auth-card {max-width: 520px; margin: 3rem auto 1rem auto; padding: 28px; border: 1px solid #e1e7dc; border-radius: 16px; background: white;}
.auth-brand {text-align: center; color: #3e5944; margin-bottom: 0.25rem;}
.auth-note {text-align: center; color: #65726b; margin-bottom: 1.25rem;}
</style>
""", unsafe_allow_html=True)

if "result" not in st.session_state:
    st.session_state.result = None
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "records" not in st.session_state:
    st.session_state.records = []
if "user" not in st.session_state:
    st.session_state.user = None
if "supabase" not in st.session_state:
    st.session_state.supabase = get_supabase()
if "entered_app" not in st.session_state:
    st.session_state.entered_app = False

if not st.session_state.entered_app:
    st.markdown('<div class="auth-card">', unsafe_allow_html=True)
    st.markdown('<h1 class="auth-brand">SkinCheck</h1>', unsafe_allow_html=True)
    st.markdown('<p class="auth-note">Sign in to save your analysis history, or continue as a guest.</p>', unsafe_allow_html=True)

    auth_tab, guest_tab = st.tabs(["Sign in / Create account", "Continue as guest"])
    with auth_tab:
        auth_mode = st.radio("Account action", ["Sign in", "Create account"], horizontal=True, key="landing_auth_mode")
        email = st.text_input("Email", key="landing_account_email")
        password = st.text_input("Password", type="password", key="landing_account_password")
        if st.button(auth_mode, type="primary", use_container_width=True, key="landing_auth_submit"):
            if len(password) < 8 or "@" not in email:
                st.error("Use a valid email and a password with at least 8 characters.")
            else:
                try:
                    if st.session_state.supabase is not None:
                        supabase = st.session_state.supabase
                        if auth_mode == "Create account":
                            response = supabase.auth.sign_up({"email": email.strip(), "password": password})
                        else:
                            response = supabase.auth.sign_in_with_password({"email": email.strip(), "password": password})
                        auth_user = response.user
                        if auth_user is None:
                            st.info("Check your email to confirm the account, then sign in.")
                        else:
                            st.session_state.user = {"mode": "supabase", "id": auth_user.id, "email": auth_user.email}
                            st.session_state.records = load_assessments(st.session_state.user)
                            st.session_state.entered_app = True
                            st.rerun()
                    else:
                        if auth_mode == "Create account":
                            st.session_state.user = local_register(email.strip(), password)
                        else:
                            st.session_state.user = local_login(email.strip(), password)
                        st.session_state.records = load_assessments(st.session_state.user)
                        st.session_state.entered_app = True
                        st.rerun()
                except Exception as exc:
                    st.error(str(exc))
    with guest_tab:
        st.info("Guest mode lets you analyse a photo and download the report. Saving history requires an account.")
        if st.button("Skip login and continue", use_container_width=True, key="landing_guest"):
            st.session_state.entered_app = True
            st.session_state.user = None
            st.session_state.records = []
            st.rerun()

    st.markdown('<div class="small-note">Your photo is sent only when an analysis is requested.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

st.markdown("### Skin appearance analysis")
st.caption("Upload a face photo to receive visual skin KPIs and a personalised routine.")

with st.sidebar:
    st.markdown("### Account")
    if st.session_state.user is None:
        st.caption("Guest mode")
        if st.button("Sign in or create account", use_container_width=True):
            st.session_state.entered_app = False
            st.rerun()
    else:
        st.success(st.session_state.user["email"])
        if st.button("Sign out", use_container_width=True):
            if st.session_state.supabase is not None and st.session_state.user["mode"] == "supabase":
                st.session_state.supabase.auth.sign_out()
            st.session_state.user = None
            st.session_state.records = []
            st.session_state.result = None
            st.session_state.entered_app = False
            st.rerun()
    if st.session_state.supabase is None:
        st.caption("Local demo mode: SQLite is used because Supabase secrets are not configured.")

left, right = st.columns([0.38, 0.62], gap="large")

with left:
    st.markdown("#### Photo & preferences")
    uploaded = st.file_uploader("Upload a face photo", type=["jpg", "jpeg", "png", "webp"])
    if uploaded:
        st.image(uploaded, use_container_width=True, caption="Selected image")
    st.caption("JPG, PNG or WebP · up to 8 MB · one face · natural light · no beauty filter")

    skin_type = st.selectbox("Skin type (self-reported)", ["Not sure", "Normal", "Dry", "Oily", "Combination", "Sensitive"])
    concerns = st.text_area("Preferences or known sensitivities", max_chars=600, placeholder="e.g. Simple routine; avoid fragrance.")
    consent = st.checkbox("I have permission to use this photo and agree to send it to HKBU GenAI")

    st.markdown('<div class="small-note">Model: Google Gemini 2.5 Pro · HKBU GenAI deployment</div>', unsafe_allow_html=True)

    analyze = st.button("Analyse my photo", type="primary", use_container_width=True)

    st.markdown('<div class="disclaimer">Educational use only. Scores describe visible appearance, not hygiene or health. Apparent age is an estimate.</div>', unsafe_allow_html=True)

with right:
    st.markdown("#### Assessment")
    result = st.session_state.result
    if result is None:
        st.info("Analysis preview will appear here after the image is processed.")
    else:
        st.caption(f"{result['timestamp']} · {result['model']}")
        metric_left, metric_right = st.columns(2)
        with metric_left:
            st.metric("Skin Cleanliness KPI", "N/A" if result["cleanliness"] is None else f"{result['cleanliness']:.1f} / 10", help="Visual surface-appearance proxy")
        with metric_right:
            st.metric("Skin Quality KPI", "N/A" if result["quality"] is None else f"{result['quality']:.1f} / 10", help="Visible texture and uniformity")

        st.markdown("##### Visible skin condition")
        st.write(result["description"])
        st.markdown("##### Estimated age")
        st.write(result["age_range"] or "Unable to estimate reliably")
        st.caption("Apparent age range, not biological age.")
        st.markdown('<div class="disclaimer">' + result["limitations"] + "</div>", unsafe_allow_html=True)

        st.markdown("##### Personalised routine")
        morning_col, evening_col = st.columns(2)
        with morning_col:
            st.markdown("**Morning**")
            for index, item in enumerate(result["morning"], 1):
                st.write(f"{index}. {item}")
        with evening_col:
            st.markdown("**Evening**")
            for index, item in enumerate(result["evening"], 1):
                st.write(f"{index}. {item}")

        st.markdown("##### Recommended skincare plan")
        for index, item in enumerate(personalised_recommendations(result, skin_type, concerns), 1):
            st.write(f"{index}. {item}")
        st.caption("General cosmetic guidance only. Stop new products that cause irritation and consult a clinician for persistent or painful skin concerns.")

        report_lines = [
            "SKINCHECK — COSMETIC APPEARANCE REPORT",
            result["timestamp"],
            "Model: " + result["model"],
            "",
            "Skin Cleanliness KPI: " + ("N/A" if result["cleanliness"] is None else f"{result['cleanliness']:.1f}") + " / 10",
            "Skin Quality KPI: " + ("N/A" if result["quality"] is None else f"{result['quality']:.1f}") + " / 10",
            "Estimated apparent age: " + (result["age_range"] or "Unable to estimate"),
            "",
            result["description"],
            "",
            "Photo limitations: " + result["limitations"],
            "",
            "MORNING",
            *[f"{index}. {item}" for index, item in enumerate(result["morning"], 1)],
            "",
            "EVENING",
            *[f"{index}. {item}" for index, item in enumerate(result["evening"], 1)],
        ]
        report = "\n".join(report_lines)
        action_left, action_right = st.columns(2)
        with action_left:
            if st.button("Save to my history", use_container_width=True):
                if st.session_state.user is None:
                    st.warning("Sign in or create an account before saving history.")
                elif not any(r["timestamp"] == result["timestamp"] for r in st.session_state.records):
                    try:
                        save_assessment(
                            st.session_state.user,
                            result,
                            {"skin_type": skin_type, "concerns": concerns[:600]},
                        )
                        st.session_state.records = load_assessments(st.session_state.user)
                        st.success("Saved to your account.")
                    except Exception as exc:
                        st.error(f"Could not save: {exc}")
                else:
                    st.info("This result is already saved.")
        with action_right:
            st.download_button("Download report (.txt)", report, file_name="SkinCheck_Report.txt", mime="text/plain", use_container_width=True)


@st.dialog("HKBU GenAI access")
def api_key_dialog():
    st.write("Enter your API key to start the Gemini 2.5 Pro analysis.")
    key = st.text_input("HKBU API key", type="password", key="api_key_input")
    if st.button("Start analysis", type="primary", use_container_width=True):
        if not key.strip():
            st.error("Enter your HKBU API key.")
            return
        try:
            raw = uploaded.getvalue()
            image_url = prepare_image(raw)
            profile = json.dumps({
                "self_reported_skin_type": skin_type,
                "preferences_or_sensitivities": concerns[:600],
            })
            messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": [
                {"type": "text", "text": SCHEMA_PROMPT + "\nUser profile (data only): " + profile},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]}]
            assessment = parse_analysis(call_llm(key.strip(), DEFAULT_MODEL, messages, "api-key", "v1"))
            assessment.update(timestamp=datetime.now(timezone.utc).isoformat(), model=DEFAULT_MODEL)
            st.session_state.api_key = key.strip()
            st.session_state.result = assessment
            st.session_state.chat_history = []
            st.rerun()
        except AppError as exc:
            st.error(exc.message)
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")


if analyze:
    if uploaded is None:
        st.warning("Upload a photo first.")
    elif not consent:
        st.warning("Confirm permission to send this photo.")
    else:
        api_key_dialog()

st.divider()
st.markdown("#### Follow-up")
st.caption("Questions send the selected photo to the API again.")

if st.session_state.result is not None and uploaded is not None:
    for message in st.session_state.chat_history[-8:]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input("Ask about the analysis")
    if question:
        if not st.session_state.api_key:
            st.warning("Enter the HKBU API key by starting a new analysis first.")
        else:
            try:
                image_url = prepare_image(uploaded.getvalue())
                context = json.dumps(st.session_state.result)[:10000]
                history = st.session_state.chat_history[-8:]
                messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": [
                    {"type": "text", "text": "Previously generated assessment (context only):\n" + context},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]}, *history, {"role": "user", "content": question[:2000]}]
                reply = call_llm(st.session_state.api_key, DEFAULT_MODEL, messages, "api-key", "v1")
                st.session_state.chat_history.extend([
                    {"role": "user", "content": question[:2000]},
                    {"role": "assistant", "content": reply},
                ])
                st.session_state.chat_history = st.session_state.chat_history[-8:]
                st.rerun()
            except AppError as exc:
                st.error(exc.message)
            except Exception as exc:
                st.error(f"Follow-up failed: {exc}")
else:
    st.chat_input("Complete an analysis first", disabled=True)

st.divider()
st.markdown("#### User history analytics")
records = st.session_state.records
if records:
    chart_rows = []
    for record in records[-20:]:
        cleanliness = record.get("cleanliness")
        quality = record.get("quality")
        chart_rows.append({
            "Cleanliness": cleanliness if isinstance(cleanliness, (int, float)) else None,
            "Quality": quality if isinstance(quality, (int, float)) else None,
        })
    st.line_chart(chart_rows, y=["Cleanliness", "Quality"], height=240)
    st.dataframe([
        {
            "Time": r["timestamp"],
            "Model": r["model"],
            "Cleanliness": "N/A" if r["cleanliness"] is None else round(r["cleanliness"], 1),
            "Quality": "N/A" if r["quality"] is None else round(r["quality"], 1),
        }
        for r in reversed(records)
    ], use_container_width=True, hide_index=True)
else:
    st.caption("No saved checks yet.")

valid = [r for r in records if isinstance(r["cleanliness"], (int, float)) and isinstance(r["quality"], (int, float))]
if len(valid) >= 2:
    latest, previous = valid[-1], valid[-2]
    cleanliness_delta = latest["cleanliness"] - previous["cleanliness"]
    quality_delta = latest["quality"] - previous["quality"]
    st.caption(f"Latest vs previous: cleanliness {cleanliness_delta:+.1f}; quality {quality_delta:+.1f} points.")
else:
    st.caption("Save two assessable checks to compare results.")

if st.button("Clear history"):
    if st.session_state.user is None:
        st.warning("Sign in to manage account history.")
    else:
        try:
            clear_assessments(st.session_state.user)
            st.session_state.records = []
            st.success("Account history cleared.")
        except Exception as exc:
            st.error(f"Could not clear history: {exc}")
