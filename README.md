# PD2 Skin Analysis

A Flask-based skincare appearance prototype that accepts a face image and uses the HKBU GenAI Gemini 2.5 Pro deployment to provide educational skin-quality observations, apparent-age estimates, and personalised cosmetic routine suggestions.

## Features

- Face image upload and preview
- API key entered through a session-only modal
- Skin cleanliness and quality KPIs
- Visible skin-condition description and apparent-age range
- Personalised morning and evening routines
- Follow-up chat
- Session-only history and comparison
- Downloadable text report

## Local setup

```powershell
python -m pip install -r requirements.txt
python skin_check.py
```

Open <http://127.0.0.1:5000>.

The application uses the HKBU GenAI deployment endpoint configured in `skin_check.py`. Enter your own API key in the page; the key is not stored in the project or repository.

## Important

This prototype provides cosmetic visual estimates for educational purposes only. It does not diagnose medical conditions, measure actual hygiene, or determine biological age.
