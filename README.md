
# Deal Risk Radar (Streamlit)

A demo that shows how AI can surface risk in sales conversations and deals.

## What it does
- Upload a CSV of call transcripts/notes and basic deal fields
- Scores sentiment with VADER and flags risk keywords
- Combines both into a 0–1 **risk_score**
- Interactive scatter + sortable table for triage

## Quickstart (local)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Then open the local URL printed in the terminal.

## Deploy free
- **Streamlit Community Cloud**: one-click from this folder/repo.
- **Render** or **Railway**: `pip install -r requirements.txt` then run `streamlit run app.py` as the start command.
- **Vercel** (via serverless Python): use `vercel.json` with a Python runtime, or wrap in a lightweight FastAPI + Streamlit Frontend.

## CSV format
Columns required (lowercase or will be normalized for you):
```
deal_id, stage, date, duration_min, transcript
```

## Notes
- This is a demo — replace the keyword rules with a fine-tuned classifier, and wire transcripts from call recordings or CRM notes.
- For Pipedrive, a simple cron can export recent activities and push to S3/CSV. 
