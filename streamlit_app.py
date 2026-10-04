import html
import os
import re
import time

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

DEFAULT_MODEL = "gemini-3.8-flash"
SAMPLE_PATH = "banking_app_feedback.csv"

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

BREVITY
Be concise. Maximum 7 themes. Keep each theme's detail under 70 words. One quote per finding. No introduction and no closing remarks.

OUTPUT FORMAT (markdown)
Use exactly these level-2 headings, in this order, with no numbering:
## Summary
Maximum 80 words, top 3 issues, one sentence each.
## Prioritized themes
A table with columns: Rank | Theme | Frequency | Impact | Severity | Comment ids. In the Frequency, Impact and Severity columns write only High, Medium or Low (Severity as "High x Medium" style).
## Theme details
For each theme give [Observed] with quote and ids, [Inference] if any, and a UI suggestion (hypothesis).
## Contradictions
Both sides, with ids and quotes.
## Feature requests
List with ids.
## Excluded comments
Id and reason.
## Limitations
One or two lines on sample size and data quality.

SELF-CHECK BEFORE ANSWERING
Confirm every quoted text appears word for word in the input, every cited id exists, and no claim lacks an [Observed] or [Inference] label. Fix any problem silently.
"""

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root{
  --bg:#0B0D10; --surface:#12151A; --surface2:#181C22; --border:#232933;
  --text:#E6E8EB; --muted:#8B93A1; --accent:#8AB4FF;
  --high:#FF7A7A; --med:#F2B84B; --low:#5ED6A4; --inf:#C7A6FF;
}
html, body, .stApp, button, input, textarea { font-family:'Inter',system-ui,-apple-system,sans-serif !important; }
.stApp{
  background: radial-gradient(1100px 460px at 50% -10%, rgba(138,180,255,.09), transparent 60%), var(--bg);
  color: var(--text);
}
header[data-testid="stHeader"]{ background:transparent; }
#MainMenu, footer, [data-testid="stToolbar"]{ visibility:hidden; }
.block-container{ max-width:860px; padding-top:3.2rem; padding-bottom:5rem; }

@keyframes fadeUp{ from{opacity:0; transform:translateY(14px);} to{opacity:1; transform:none;} }
@keyframes shimmer{ 0%{background-position:-600px 0;} 100%{background-position:600px 0;} }
@keyframes pop{ from{opacity:0; transform:scale(.85);} to{opacity:1; transform:none;} }
@keyframes breathe{ 0%,100%{opacity:.45;} 50%{opacity:1;} }

.hero{ animation:fadeUp .7s ease both; margin-bottom:1.6rem; }
.eyebrow{ display:inline-flex; align-items:center; gap:8px; font-size:.74rem; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); }
.eyebrow i{ width:7px; height:7px; border-radius:50%; background:var(--accent); animation:breathe 2.4s ease-in-out infinite; }
.hero h1{ font-size:2.5rem; line-height:1.1; font-weight:600; letter-spacing:-.025em; margin:.7rem 0 .6rem; padding:0; }
.hero h1 span{ color:var(--accent); }
.hero p{ color:var(--muted); font-size:1rem; max-width:560px; margin:0; }

.stats{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:1.2rem 0 1rem; }
.stat{ background:var(--surface); border:1px solid var(--border); border-radius:14px; padding:14px 16px; animation:fadeUp .6s ease both; transition:transform .25s ease, border-color .25s ease; }
.stat:hover{ transform:translateY(-3px); border-color:#34404f; }
.stat b{ display:block; font-size:1.55rem; font-weight:600; letter-spacing:-.02em; }
.stat span{ color:var(--muted); font-size:.72rem; text-transform:uppercase; letter-spacing:.08em; }

.grid{ display:grid; grid-template-columns:repeat(auto-fill,minmax(250px,1fr)); gap:10px; margin-top:.4rem; }
.card{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:12px 14px; animation:fadeUp .5s ease both; animation-delay:calc(var(--i) * 45ms); transition:border-color .2s ease, transform .2s ease; }
.card:hover{ border-color:var(--accent); transform:translateY(-2px); }
.card .meta{ display:flex; gap:8px; flex-wrap:wrap; font-size:.7rem; color:var(--muted); margin-bottom:6px; }
.card .meta .seg{ color:var(--accent); }
.card p{ margin:0; font-size:.86rem; line-height:1.5; color:#CBD0D8; }

.stButton > button, .stDownloadButton > button{
  background:var(--surface); border:1px solid var(--border); color:var(--text);
  border-radius:12px; padding:.6rem 1.1rem; font-weight:500;
  transition:transform .2s ease, border-color .2s ease, box-shadow .2s ease, background .2s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover{ border-color:var(--accent); transform:translateY(-2px); box-shadow:0 8px 24px rgba(138,180,255,.12); }
.stButton > button:active{ transform:translateY(0) scale(.99); }
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"]{
  background:var(--accent); color:#0B0D10; border-color:var(--accent); font-weight:600;
}
.stButton > button[kind="primary"]:hover, .stButton > button[data-testid="stBaseButton-primary"]:hover{ box-shadow:0 10px 30px rgba(138,180,255,.35); }

textarea, [data-testid="stFileUploaderDropzone"]{ background:var(--surface) !important; border:1px solid var(--border) !important; border-radius:12px !important; transition:border-color .2s ease; }
textarea:focus, [data-testid="stFileUploaderDropzone"]:hover{ border-color:var(--accent) !important; }
[data-testid="stExpander"], [data-testid="stStatusWidget"], details{ background:var(--surface); border:1px solid var(--border) !important; border-radius:12px !important; }

button[data-baseweb="tab"]{ background:transparent; color:var(--muted); font-weight:500; transition:color .2s ease; }
button[data-baseweb="tab"]:hover{ color:var(--text); }
button[data-baseweb="tab"][aria-selected="true"]{ color:var(--text); }
[data-baseweb="tab-panel"]{ animation:fadeUp .45s ease both; padding-top:1rem; }

.sec-title{ font-size:.74rem; text-transform:uppercase; letter-spacing:.1em; color:var(--muted); margin:1.3rem 0 .5rem; }
.stMarkdown table{ width:100%; border-collapse:separate; border-spacing:0; border:1px solid var(--border); border-radius:12px; overflow:hidden; }
.stMarkdown th{ background:var(--surface2); color:var(--muted); font-weight:500; font-size:.74rem; text-transform:uppercase; letter-spacing:.06em; }
.stMarkdown th, .stMarkdown td{ padding:10px 12px; border-bottom:1px solid var(--border); text-align:left; }
.stMarkdown tr:last-child td{ border-bottom:none; }
.stMarkdown tr:hover td{ background:rgba(255,255,255,.025); }
.stMarkdown blockquote{ border-left:2px solid var(--accent); color:#CBD0D8; background:var(--surface); border-radius:0 10px 10px 0; padding:.4rem 1rem; }

.pill{ display:inline-block; padding:2px 10px; border-radius:999px; font-size:.74rem; font-weight:600; margin-right:4px; }
.pill.high{ background:rgba(255,122,122,.14); color:var(--high); }
.pill.medium{ background:rgba(242,184,75,.14); color:var(--med); }
.pill.low{ background:rgba(94,214,164,.14); color:var(--low); }
.tag{ display:inline-block; padding:1px 8px; border-radius:6px; font-size:.7rem; font-weight:600; letter-spacing:.02em; }
.tag.obs{ background:rgba(138,180,255,.14); color:var(--accent); }
.tag.inf{ background:rgba(199,166,255,.14); color:var(--inf); }

.verify{ display:inline-flex; align-items:center; gap:8px; padding:6px 14px; border-radius:999px; border:1px solid; font-size:.8rem; animation:pop .55s cubic-bezier(.2,1.4,.4,1) both; }
.verify.ok{ color:var(--low); border-color:rgba(94,214,164,.4); background:rgba(94,214,164,.08); }
.verify.warn{ color:var(--med); border-color:rgba(242,184,75,.4); background:rgba(242,184,75,.08); }

.skel{ display:grid; gap:12px; margin:1.2rem 0; }
.bar{ height:14px; border-radius:8px; background:linear-gradient(90deg,var(--surface) 0%,var(--surface2) 40%,var(--surface) 80%); background-size:1200px 100%; animation:shimmer 1.5s infinite linear; }
.w90{width:90%;} .w75{width:75%;} .w60{width:60%;} .w40{width:40%;}

.hint{ color:var(--muted); font-size:.9rem; padding:1rem 0; }
@media (max-width:640px){ .stats{ grid-template-columns:repeat(2,1fr);} .hero h1{ font-size:2rem; } }
@media (prefers-reduced-motion: reduce){ *{ animation:none !important; transition:none !important; } }
</style>
"""

SKELETON = (
    '<div class="skel"><div class="bar w60"></div><div class="bar w90"></div>'
    '<div class="bar w75"></div><div class="bar w40"></div></div>'
)

GROUP_ORDER = ["Overview", "Themes", "Conflicts", "Requests", "Excluded"]
GROUP_KEYWORDS = [
    ("Conflicts", ["contradict"]),
    ("Requests", ["feature"]),
    ("Excluded", ["exclu"]),
    ("Overview", ["summary", "limitation"]),
    ("Themes", ["theme", "priorit"]),
]


# ---------- data helpers ----------

def esc(text):
    """Escape for HTML and stop Streamlit from reading $ as LaTeX."""
    return html.escape(str(text)).replace("$", "&#36;").replace("\n", " ")


def load_csv(src):
    try:
        df = pd.read_csv(src)
    except Exception as e:
        st.error(f"Could not read that CSV: {e}")
        return None
    df.columns = [c.strip().lower() for c in df.columns]
    if "comment" not in df.columns:
        st.error("The CSV needs a 'comment' column. Optional columns: id, source, user_segment.")
        return None
    if "id" not in df.columns:
        df.insert(0, "id", range(1, len(df) + 1))
    for col in ("source", "user_segment"):
        if col not in df.columns:
            df[col] = "Unspecified"
    df = df.dropna(subset=["comment"]).fillna("Unspecified")
    return df[["id", "source", "user_segment", "comment"]]


def parse_pasted(text):
    rows = [line.strip() for line in text.splitlines() if line.strip()]
    return pd.DataFrame(
        {"id": range(1, len(rows) + 1), "source": "Pasted", "user_segment": "Unspecified", "comment": rows}
    )


def to_prompt_text(df):
    lines = []
    for _, r in df.iterrows():
        lines.append(f'[{r["id"]}] source={r["source"]} | segment={r["user_segment"]} | comment: "{r["comment"]}"')
    return "\n".join(lines)


# ---------- model call and checks ----------

def get_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.environ.get("GEMINI_API_KEY")


def get_model():
    try:
        return st.secrets["MODEL"]
    except Exception:
        return os.environ.get("MODEL", DEFAULT_MODEL)


def make_config(use_thinking):
    kwargs = {"system_instruction": SYSTEM_PROMPT, "temperature": 0.2, "max_output_tokens": 6000}
    if use_thinking:
        # Gemini 3 models think at a high level by default, which is slow. Low is plenty here.
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="low")
    return types.GenerateContentConfig(**kwargs)


def stream_analysis(feedback):
    """Yield the answer in pieces as Gemini writes it, retrying if the model is busy."""
    client = genai.Client(api_key=get_api_key())
    waits = [1, 3, 6]  # seconds between retries when the model is busy
    use_thinking = True
    attempt = 0
    while True:
        started = False
        try:
            for chunk in client.models.generate_content_stream(
                model=get_model(), contents=feedback, config=make_config(use_thinking)
            ):
                if chunk.text:
                    started = True
                    yield chunk.text
            return
        except Exception as e:
            msg = str(e)
            if use_thinking and not started and "thinking" in msg.lower():
                use_thinking = False  # this model does not accept the thinking setting
                continue
            busy = any(code in msg for code in ("503", "429", "UNAVAILABLE"))
            if busy and not started and attempt < len(waits):
                time.sleep(waits[attempt])
                attempt += 1
                continue
            raise


def normalize(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def quote_stats(output, feedback):
    quotes = re.findall(r'["\u201c]([^"\u201d]{15,})["\u201d]', output)
    source = normalize(feedback)
    missing = [q for q in quotes if normalize(q) not in source]
    return len(quotes) - len(missing), len(quotes), missing


# ---------- rendering helpers ----------

def decorate(md):
    """Turn High/Medium/Low table cells and [Observed]/[Inference] labels into colored chips."""
    sev = r"(High|Medium|Low)"
    cell_re = re.compile(rf"{sev}(\s*[x\u00d7]\s*{sev})?")

    def pill(m):
        return f'<span class="pill {m.group(1).lower()}">{m.group(1)}</span>'

    out = []
    for line in md.splitlines():
        if line.lstrip().startswith("|"):
            cells = line.split("|")
            for i, cell in enumerate(cells):
                if cell_re.fullmatch(cell.strip()):
                    cells[i] = " " + re.sub(sev, pill, cell.strip()) + " "
            line = "|".join(cells)
        out.append(line)
    text = "\n".join(out)
    text = text.replace("[Observed]", '<span class="tag obs">Observed</span>')
    text = text.replace("[Inference]", '<span class="tag inf">Inference</span>')
    return text


def split_sections(md):
    parts = re.split(r"(?m)^##\s+(.+?)\s*$", md)
    sections = []
    for i in range(1, len(parts) - 1, 2):
        name = re.sub(r"^\d+[.)]\s*", "", parts[i]).strip()
        sections.append((name, parts[i + 1].strip()))
    return sections


def group_of(name):
    low = name.lower()
    for group, keys in GROUP_KEYWORDS:
        if any(k in low for k in keys):
            return group
    return "Overview"


def render_results(result):
    verified, total, missing = quote_stats(result["text"], result["input"])
    head_l, head_r = st.columns([3, 1])
    with head_l:
        if total:
            cls = "ok" if not missing else "warn"
            label = f"{verified} of {total} quotes verified against your data"
            st.markdown(f'<div class="verify {cls}">&#10003; {label}</div>', unsafe_allow_html=True)
    with head_r:
        st.download_button(
            "Download report",
            data=result["text"],
            file_name="usability_insights.md",
            mime="text/markdown",
            use_container_width=True,
        )
    if missing:
        with st.expander("Quotes not found in the input, please review"):
            for q in missing:
                st.markdown(f"- {q}")

    sections = split_sections(result["text"])
    if not sections:
        st.markdown(decorate(result["text"]), unsafe_allow_html=True)
        return

    grouped = {g: [] for g in GROUP_ORDER}
    for name, body in sections:
        grouped[group_of(name)].append((name, body))
    names = [g for g in GROUP_ORDER if grouped[g]]
    tabs = st.tabs(names)
    for tab, g in zip(tabs, names):
        with tab:
            for name, body in grouped[g]:
                st.markdown(f'<div class="sec-title">{esc(name)}</div>', unsafe_allow_html=True)
                st.markdown(decorate(body), unsafe_allow_html=True)


# ---------- page ----------

st.set_page_config(page_title="Design Companion", page_icon="\u25d0", layout="centered")
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    '<div class="hero"><div class="eyebrow"><i></i>Design Companion</div>'
    "<h1>Usability feedback, <span>distilled.</span></h1>"
    "<p>Turn raw comments into prioritized, evidence-backed UI insights. "
    "Every finding cites its source and is checked against your data. Dummy data only.</p></div>",
    unsafe_allow_html=True,
)

mode = st.segmented_control(
    "Input", ["Sample data", "Upload CSV", "Paste text"], default="Sample data", label_visibility="collapsed"
) or "Sample data"

df = None
if mode == "Sample data":
    df = load_csv(SAMPLE_PATH)
elif mode == "Upload CSV":
    up = st.file_uploader("CSV file", type="csv", label_visibility="collapsed")
    st.caption("Columns: comment (required), id, source, user_segment (optional)")
    if up is not None:
        df = load_csv(up)
else:
    pasted = st.text_area("Comments", height=200, placeholder="One comment per line", label_visibility="collapsed")
    df = parse_pasted(pasted)

if df is None or df.empty:
    st.markdown('<div class="hint">Add some feedback to get started.</div>', unsafe_allow_html=True)
    st.stop()

segments = sorted(df["user_segment"].astype(str).unique())
picked = []
if len(segments) > 1:
    st.markdown('<div class="sec-title">Focus on a segment (optional)</div>', unsafe_allow_html=True)
    picked = st.pills("Segments", segments, selection_mode="multi", label_visibility="collapsed") or []
view = df[df["user_segment"].astype(str).isin(picked)] if picked else df

focus_label = f"{len(picked)} selected" if picked else "All"
stats = [
    (len(view), "Comments"),
    (view["source"].nunique(), "Sources"),
    (view["user_segment"].nunique(), "Segments"),
    (focus_label, "Focus"),
]
tiles = "".join(
    f'<div class="stat" style="animation-delay:{i * 70}ms"><b>{esc(v)}</b><span>{esc(k)}</span></div>'
    for i, (v, k) in enumerate(stats)
)
st.markdown(f'<div class="stats">{tiles}</div>', unsafe_allow_html=True)

with st.expander(f"Browse {len(view)} comments"):
    cards = "".join(
        f'<div class="card" style="--i:{min(i, 14)}"><div class="meta"><span>#{esc(r.id)}</span>'
        f'<span>{esc(r.source)}</span><span class="seg">{esc(r.user_segment)}</span></div>'
        f"<p>{esc(r.comment)}</p></div>"
        for i, r in enumerate(view.itertuples())
    )
    st.markdown(f'<div class="grid">{cards}</div>', unsafe_allow_html=True)

run = st.button(f"Analyze {len(view)} comments", type="primary", use_container_width=True)

if run:
    if not get_api_key():
        st.error("No API key found. Add GEMINI_API_KEY in your app secrets.")
    else:
        feedback_text = to_prompt_text(view)
        skeleton = st.empty()
        skeleton.markdown(SKELETON, unsafe_allow_html=True)
        live = st.empty()
        started_at = time.time()
        with st.status("Analyzing feedback", expanded=True) as status:
            try:
                st.write(f"Sent {len(view)} comments to Gemini")
                text = ""
                for piece in stream_analysis(feedback_text):
                    if not text:
                        skeleton.empty()
                        st.write("Writing insights")
                    text += piece
                    live.markdown(text)
                elapsed = time.time() - started_at
                st.write("Checking quotes against your data")
                st.session_state["result"] = {
                    "text": text, "input": feedback_text, "n": len(view), "secs": elapsed,
                }
                status.update(label=f"Analysis complete in {elapsed:.0f}s", state="complete", expanded=False)
            except Exception as e:
                status.update(label="Analysis failed", state="error", expanded=True)
                st.error(f"Something went wrong: {e}")
        skeleton.empty()
        live.empty()

result = st.session_state.get("result")
if result:
    st.markdown(f'<div class="sec-title">Insights from {result["n"]} comments in {result.get("secs", 0):.0f}s</div>', unsafe_allow_html=True)
    render_results(result)
