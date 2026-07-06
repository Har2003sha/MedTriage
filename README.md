# MedTriage AI — Multilingual Medical Triage Assistant (Flask)

A working Flask web app that accepts patient symptoms in **English, Hindi,
Hinglish (Roman-script Hindi, e.g. "chest me dard ho raha hai"), Marathi,
Tamil, and Telugu**, classifies urgency, recommends a hospital department,
and generates a clinical summary for staff.

> ⚠️ **This is a decision-support prototype, not a certified medical device.**
> It must not be used as the sole basis for real clinical triage without
> validation by qualified medical professionals and regulatory review.

## Stack
- **Backend/Frontend:** Flask (server-rendered, Jinja2 + Bootstrap 5)
- **Auth:** Flask-Login + Flask-Bcrypt (hashed passwords, sessions)
- **Database:** SQLAlchemy — SQLite by default, PostgreSQL via `DATABASE_URL`
- **NLP classifier:** multilingual lexicon-based urgency/department engine
  (`models_ml/classifier.py`) — swappable for a real fine-tuned model
- **Summary generation:** fully offline, rule-based (`llm_service.py`) —
  **no API key, no external calls, no internet dependency required**

## 1. Setup

```bash
cd medtriage
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# edit .env: set SECRET_KEY (optionally set DATABASE_URL for Postgres)
python app.py
```

No API key of any kind is required — the app runs 100% offline out of the box.

Visit `http://localhost:5000`, register an account, and submit a symptom
description in any of the 5 supported languages.

## 2. Using PostgreSQL instead of SQLite

1. Create a database: `createdb medtriage`
2. In `.env`: `DATABASE_URL=postgresql://user:password@localhost:5432/medtriage`
3. Re-run `python app.py` (tables are auto-created on startup). For schema
   migrations going forward, use Flask-Migrate (`flask db init/migrate/upgrade`)
   which is already wired in via `Migrate(app, db)` in `app.py`.

## 3. Clinical summaries (no API key needed)

`llm_service.py` builds each triage note entirely offline: it takes the
classifier's output (urgency, department, matched symptom categories) and
assembles a readable clinical sentence from templates — no external API,
no internet call, no key to manage. This is intentional so the app has
zero external dependencies and zero recurring cost.

If you ever want to upgrade this to a true LLM-generated summary later,
`generate_summary()` in `llm_service.py` is the single function to replace
— just keep the same return shape `(summary_text, source_label)`.

## 4. How the multilingual classification works today

`models_ml/classifier.py` contains a hand-built lexicon of symptom phrases
in all 5 languages (chest pain, breathlessness, stroke signs, fever, etc.),
each tagged with an urgency weight and department. Free text is matched
against these phrases to produce:
- `urgency`: Emergency / Urgent / Non-Urgent
- `score`: 1–10 severity score
- `department`: recommended hospital department
- `matched_terms`: which symptom categories fired

This gives a fully functional, fast, offline baseline across all 5
languages without needing GPU infrastructure or model hosting.

## 5. Upgrading to a real fine-tuned multilingual LLM (recommended path)

To move from lexicon-matching to an actual fine-tuned model:

1. **Pick a base model suited to Indic languages**, e.g.:
   - `google/muril-base-cased` (Hindi, Marathi, Tamil, Telugu + English)
   - `ai4bharat/indic-bert`
   - A multilingual instruction-tuned LLM (e.g. via API) prompted for
     classification instead of fine-tuned
2. **Collect/label training data**: symptom text → (urgency label,
   department label) pairs, ideally sourced from real (de-identified,
   consented) triage logs or clinician-authored examples in each language.
3. **Fine-tune** a classification head on top of the encoder (standard
   HuggingFace `Trainer` workflow), or fine-tune/prompt-engineer an LLM for
   structured JSON output.
4. **Swap the implementation**: replace the body of `score_symptoms()` in
   `models_ml/classifier.py` with a call to your model (local
   `transformers` pipeline or a hosted inference endpoint), keeping the same
   return shape:
   ```python
   {"urgency": ..., "score": ..., "department": ..., "matched_terms": [...]}
   ```
   No other file needs to change — routes, DB schema, and templates are
   already built against this contract.
5. **Validate clinically** before any real-world use — have licensed
   medical professionals review classification accuracy per language and
   symptom category, including false-negative (under-triage) rates
   specifically, since those carry the highest patient risk.

## 6. Project structure

```
medtriage/
├── app.py                 # Flask app factory, entry point
├── config.py               # Env-based configuration
├── extensions.py            # bcrypt, login_manager (avoids circular imports)
├── models.py                # SQLAlchemy models: User, TriageRecord
├── auth.py                  # Register/login/logout blueprint
├── triage.py                 # Dashboard, symptom form, result view blueprint
├── llm_service.py            # Claude API summary + offline fallback
├── models_ml/
│   └── classifier.py         # Multilingual symptom lexicon + scoring logic
├── templates/                # Jinja2 + Bootstrap templates
├── static/style.css
├── requirements.txt
└── .env.example
```

## 7. Security notes for production

- Set a strong random `SECRET_KEY`.
- Run behind HTTPS; set `SESSION_COOKIE_SECURE = True`.
- Add rate limiting (e.g. Flask-Limiter) on `/auth/login` and `/triage/new`.
- Treat symptom text and triage records as sensitive health data (PHI):
  encrypt at rest, restrict DB access, add audit logging, and review
  applicable regulations (e.g. India's DPDP Act, HIPAA if serving US
  patients) before production deployment.
