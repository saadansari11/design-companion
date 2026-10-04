import os
import re

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = """
ROLE
You are a senior UX researcher analyzing usability feedback for a mobile banking app. Your output has two layers: a short summary for stakeholders, then detail for designers.

INPUT
A numbered list of feedback comments. Each has: id, source, user_segment, comment.

EVIDENCE RULES
1. Use only the comments provided. Do not use outside knowledge about this app, competitors, or banking in general.
2. Every finding must cite comment ids and include one verbatim quote, copied exactly from the input.
3. Label every statement as [Observed] (directly stated in the comments) or [Inference] (your interpretation). Each [Inference] must name the comment ids it is based on.
4. Count frequency from the ids you cite. Never estimate. If a theme has one comment, write "single mention".
5. Do not invent numbers, percentages, quotes, user segments, or behaviors that are not in the input.
6. If the evidence is thin, write "Insufficient evidence" instead of guessing.
7. When comments contradict each other, report both sides with ids. Do not average them or pick a winner.
8. Separate feature requests from usability problems.
9. Exclude comments that are (a) too vague to act on, or (b) not about the app experience (for example branches, ATMs, staff). List each excluded id with a one-line reason.

SEVERITY
Frequency: High = 4 or more comments, Medium = 2 to 3, Low = 1.
Impact: High = blocks a task or risks money or trust, Medium = slows or confuses, Low = annoyance or preference.
Show severity as "Frequency x Impact" with one line of reasoning.

SUGGESTIONS
Suggest UI-level changes only: layout, UI copy, components, visual hierarchy. Do not propose new features, backend changes, or full flow redesigns. Phrase each suggestion as a hypothesis to test, tied to a specific theme. Where useful, include example UI copy.

OUTPUT FORMAT (markdown)
1. Summary: maximum 80 words, top 3 issues, one sentence each.
2. Prioritized themes: table with Rank | Theme | Frequency | Impact | Severity | Comment ids.
3. Theme details: for each theme give [Observed] with quote and ids, [Inference] if any, and a UI suggestion (hypothesis).
4. Contradictions: both sides, with ids and quotes.
5. Feature requests: list with ids.
6. Excluded comments: id and reason.
7. Limitations: one or two lines on sample size and data quality.

SELF-CHECK BEFORE ANSWERING
Confirm every quoted text appears word for word in the input, every cited id exists, and no claim lacks an [Observed] or [Inference] label. Fix any problem silently.
"""


def get_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.environ.get("GEMINI_API_KEY")


def csv_to_text(source):
    """Turn a feedback CSV (path or file object) into a numbered text block."""
    df = pd.read_csv(source)
    lines = []
    for _, r in df.iterrows():
        lines.append(
            f'[{r["id"]}] source={r["source"]} | segment={r["user_segment"]} | comment: "{r["comment"]}"'
        )
    return "\n".join(lines)


def normalize(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def quote_check(output, feedback):
    """Check that quoted passages in the output really exist in the input."""
    quotes = re.findall(r'["\u201c]([^"\u201d]{15,})["\u201d]', output)
    if not quotes:
        return ""
    source = normalize(feedback)
    missing = [q for q in quotes if normalize(q) not in source]
    verified = len(quotes) - len(missing)
    note = f"\n\n---\n**Quote check:** {verified} of {len(quotes)} quotes verified word for word against the input."
    if missing:
        note += "\n\nNot found in the input, please review:\n" + "\n".join(f"- {m}" for m in missing)
    return note


def analyze(feedback):
    client = genai.Client(api_key=get_api_key())
    response = client.models.generate_content(
        model=MODEL,
        contents=feedback,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
        ),
    )
    return response.text + quote_check(response.text, feedback)


def load_sample():
    st.session_state["feedback"] = csv_to_text("banking_app_feedback.csv")


st.set_page_config(page_title="Design Companion", layout="wide")
st.title("Design Companion")
st.caption(
    "Turn raw usability feedback into prioritized, evidence-backed UI insights. "
    "Every finding cites comment ids and verbatim quotes. Uses dummy data only."
)

left, right = st.columns(2)

with left:
    st.button("Load sample banking feedback", on_click=load_sample)

    uploaded = st.file_uploader("Or upload a CSV (id, source, user_segment, comment)", type="csv")
    if uploaded is not None:
        upload_key = (uploaded.name, uploaded.size)
        if st.session_state.get("last_upload") != upload_key:
            try:
                st.session_state["feedback"] = csv_to_text(uploaded)
                st.session_state["last_upload"] = upload_key
            except Exception as e:
                st.error(f"Could not read the CSV: {e}")

    feedback = st.text_area("Feedback comments", key="feedback", height=380, placeholder="Paste feedback here...")
    run = st.button("Summarize feedback", type="primary")

with right:
    if run:
        if not feedback.strip():
            st.warning("Please add some feedback first.")
        else:
            with st.spinner("Analyzing feedback..."):
                try:
                    st.markdown(analyze(feedback))
                except Exception as e:
                    st.error(f"Something went wrong: {e}")
