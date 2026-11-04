# PD2 Skin Analysis

A Flask and Streamlit skincare appearance prototype that accepts a face image and uses the HKBU GenAI Gemini 2.5 Pro deployment to provide educational skin-quality observations, apparent-age estimates, and personalised cosmetic routine suggestions.

## Features

- Face image upload and preview
- API key entered through a session-only modal
- Skin cleanliness and quality KPIs
- Visible skin-condition description and apparent-age range
- Personalised morning and evening routines
- Personalised recommendations based on skin type and sensitivities
- Follow-up chat
- Account-based persistent history and trend analytics
- Email/password user access
- Supabase/Postgres storage with local SQLite fallback
- Downloadable text report

## Local setup

```powershell
python -m pip install -r requirements_features.txt
```

Run the Streamlit version:

```powershell
python -m streamlit run app_features.py
```

Run the Flask version locally:

```powershell
python skin_check.py
```

Streamlit Cloud entry point: `app_features.py`. This preserves the original `app.py` deployment and keeps the feature UI in a separate named file. Local Flask address: <http://127.0.0.1:5000>.

The application uses the HKBU GenAI deployment endpoint configured in `skin_check.py`. Enter your own API key in the page; the key is not stored in the project or repository.

## Supabase setup for persistent accounts

1. Create a Supabase project.
2. Run `supabase.sql` in the Supabase SQL editor.
3. Add the project URL and publishable/anon key to Streamlit secrets:

```toml
[supabase]
url = "https://your-project.supabase.co"
key = "your-anon-key"
```

In Streamlit Community Cloud, add the same values under **App settings → Secrets**. Row Level Security limits each user to their own assessment records. Without Supabase configuration, the app falls back to local SQLite for demonstration.

## Important

This prototype provides cosmetic visual estimates for educational purposes only. It does not diagnose medical conditions, measure actual hygiene, or determine biological age.
