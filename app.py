import json
from datetime import datetime, timezone

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

st.markdown("### Skin appearance analysis")
st.caption("Upload a face photo to receive visual skin KPIs and a personalised routine.")

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
            if st.button("Save to session history", use_container_width=True):
                if not any(r["timestamp"] == result["timestamp"] for r in st.session_state.records):
                    st.session_state.records.append(dict(result))
                    st.session_state.records = st.session_state.records[-20:]
                    st.success("Saved to this session.")
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
    st.chat_input("Ask about the analysis", disabled=True, placeholder="Complete an analysis first")

st.divider()
st.markdown("#### Session history & comparison")
records = st.session_state.records
if records:
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
    st.session_state.records = []
    st.success("Session history cleared.")
