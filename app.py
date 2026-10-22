from pathlib import Path
from datetime import datetime
import io
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------------
# CreditWise | Modern Fintech SaaS UI (Streamlit-only)
# Compatible with models/loan_model.pkl created by train_model.py
# ------------------------------------------------------------
st.set_page_config(
    page_title="CreditWise | Loan Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "loan_model.pkl"
DATA_PATH = ROOT / "loan_approval_data.csv"

# A restrained fintech visual system: no default Streamlit sidebar layout.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#172b4d; --muted:#75839a; --line:#e8edf4; --blue:#3457d5; --blue2:#edf2ff; --green:#16836b; --bg:#f5f7fb; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:var(--bg); color:var(--ink); }
.block-container { max-width:1440px; padding-top:1.1rem; padding-bottom:3rem; }
header[data-testid="stHeader"] { background:transparent; }
#MainMenu, footer {visibility:hidden;}
div[data-testid="stToolbar"] {visibility:hidden;height:0;}
.brand {display:flex;align-items:center;gap:11px;}
.brand-mark {width:42px;height:42px;border-radius:14px;background:#172b4d;color:#fff;display:flex;align-items:center;justify-content:center;font-size:24px;font-weight:800;}
.brand-name {font-family:'Manrope',sans-serif;font-weight:800;font-size:21px;letter-spacing:-.7px;color:#172b4d;}
.brand-sub {font-size:11px;color:#8b97aa;letter-spacing:1.5px;text-transform:uppercase;margin-top:-2px;}
.topline {height:1px;background:#e8edf4;margin:14px 0 22px;}
.eyebrow {font-size:11px;font-weight:700;letter-spacing:1.6px;text-transform:uppercase;color:#73829a;margin-bottom:8px;}
.page-title {font-family:'Manrope',sans-serif;font-size:31px;line-height:1.2;font-weight:800;letter-spacing:-1.1px;color:#172b4d;margin:0 0 7px;}
.page-sub {font-size:14px;color:#7b879b;margin-bottom:20px;}
.hero {background:linear-gradient(112deg,#172b4d 0%,#253f78 65%,#3457d5 100%);border-radius:22px;padding:27px 30px;color:white;position:relative;overflow:hidden;margin:6px 0 22px;}
.hero:after {content:"";position:absolute;width:260px;height:260px;border:1px solid rgba(255,255,255,.13);border-radius:50%;right:-45px;top:-110px;box-shadow:0 0 0 30px rgba(255,255,255,.035),0 0 0 62px rgba(255,255,255,.025);}
.hero-kicker {font-size:11px;letter-spacing:1.7px;text-transform:uppercase;color:#bdcbff;font-weight:700;}
.hero-title {font-family:'Manrope',sans-serif;font-size:27px;line-height:1.25;font-weight:800;letter-spacing:-.7px;margin:10px 0 8px;color:white;}
.hero-copy {font-size:13px;line-height:1.6;color:#d9e2ff;max-width:650px;}
.hero-chip {display:inline-block;border:1px solid rgba(255,255,255,.22);padding:7px 11px;border-radius:100px;font-size:11px;color:#eef2ff;margin-top:15px;}
.section-head {display:flex;justify-content:space-between;align-items:center;margin:22px 0 12px;}
.section-title {font-family:'Manrope',sans-serif;font-size:17px;font-weight:800;color:#172b4d;letter-spacing:-.3px;}
.section-note {font-size:12px;color:#8490a3;}
.kpi {background:#fff;border:1px solid #e8edf4;border-radius:17px;padding:17px 18px;min-height:116px;}
.kpi-label {font-size:12px;color:#78869a;font-weight:600;margin-bottom:10px;}
.kpi-value {font-family:'Manrope',sans-serif;font-size:27px;font-weight:800;color:#172b4d;letter-spacing:-.8px;}
.kpi-foot {font-size:11px;color:#8a96a8;margin-top:7px;}
.panel {background:white;border:1px solid #e8edf4;border-radius:18px;padding:20px 21px;}
.panel-title {font-family:'Manrope',sans-serif;font-size:15px;font-weight:800;color:#172b4d;margin-bottom:3px;}
.panel-sub {font-size:12px;color:#8a96a8;margin-bottom:16px;}
.status-pill {display:inline-block;border-radius:100px;padding:5px 9px;font-size:11px;font-weight:700;}
.status-approved {background:#e4f6ef;color:#13795f;}
.status-rejected {background:#fff0ef;color:#c64d49;}
.status-review {background:#fff4db;color:#9b6a08;}
.mini-label {font-size:11px;color:#8490a3;text-transform:uppercase;letter-spacing:1px;font-weight:700;}
.result-big {font-family:'Manrope',sans-serif;font-size:30px;font-weight:800;letter-spacing:-.8px;margin:3px 0 4px;}
div[data-testid="stForm"] {background:#fff;border:1px solid #e8edf4;border-radius:18px;padding:22px;}
div[data-testid="stTextInput"] input, div[data-testid="stNumberInput"] input, div[data-testid="stSelectbox"] div[data-baseweb="select"] {border-radius:10px!important;}
.stButton>button, .stDownloadButton>button, button[kind="primary"] {border-radius:10px;font-weight:700;min-height:42px;}
.stButton>button[kind="primary"], .stFormSubmitButton>button[kind="primary"] {background:#3457d5;border-color:#3457d5;}
.stTabs [data-baseweb="tab-list"] {gap:8px;background:#edf1f7;padding:5px;border-radius:12px;}
.stTabs [data-baseweb="tab"] {height:38px;border-radius:8px;padding:0 16px;font-weight:700;}
.stTabs [aria-selected="true"] {background:white!important;color:#3457d5!important;}
div[data-testid="stDataFrame"] {border:1px solid #e8edf4;border-radius:12px;overflow:hidden;}
div[data-testid="stAlert"] {border-radius:12px;}
div[data-testid="stMetric"] {background:transparent;}
hr {border-color:#e8edf4;}
div[data-testid="stProgressBar"] > div > div {background:#3457d5;}
.small-muted {font-size:12px;color:#8793a5;}
.footer {text-align:center;color:#9aa5b5;font-size:11px;padding-top:28px;}
</style>
""", unsafe_allow_html=True)

# ---------- Model / data loading ----------
@st.cache_resource
def load_artifact(path_string, modified):
    return joblib.load(path_string)

@st.cache_data
def load_dataset(path_string, modified):
    return pd.read_csv(path_string)

def get_model_bundle():
    if not MODEL_PATH.exists():
        return None, f"Model file not found: {MODEL_PATH.relative_to(ROOT)}"
    try:
        artifact = load_artifact(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
        if isinstance(artifact, dict) and "model" in artifact:
            return artifact, None
        # Also tolerate a plain estimator if metadata is available separately.
        return {"model": artifact}, None
    except Exception as exc:
        return None, f"Could not load model: {exc}"

def get_data():
    if not DATA_PATH.exists():
        return None, f"Dataset file not found: {DATA_PATH.name}"
    try:
        return load_dataset(str(DATA_PATH), DATA_PATH.stat().st_mtime), None
    except Exception as exc:
        return None, f"Could not load dataset: {exc}"

bundle, model_error = get_model_bundle()
dataset, data_error = get_data()
model = bundle.get("model") if bundle else None
feature_columns = bundle.get("feature_columns", []) if bundle else []
numeric_columns = bundle.get("numeric_columns", []) if bundle else []
categorical_columns = bundle.get("categorical_columns", []) if bundle else []
metrics = bundle.get("metrics", {}) if bundle else {}

if "applications" not in st.session_state:
    st.session_state.applications = []
if "active_page" not in st.session_state:
    st.session_state.active_page = "Overview"
if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

# ---------- Helpers ----------
def fmt_int(value):
    try: return f"{int(value):,}"
    except Exception: return "—"

def status_label(pred):
    return "Approved" if int(pred) == 1 else "Not approved"

def probability_for_positive(estimator, frame, predicted):
    if hasattr(estimator, "predict_proba"):
        probs = estimator.predict_proba(frame)[0]
        classes = list(getattr(estimator, "classes_", []))
        if 1 in classes:
            return float(probs[classes.index(1)])
        if len(probs) > 1:
            return float(probs[-1])
    return float(predicted)

def pretty_feature(name):
    return str(name).replace("_", " ").replace("-", " ").title()

def numeric_default(col):
    if dataset is not None and col in dataset.columns:
        vals = pd.to_numeric(dataset[col], errors="coerce").dropna()
        if not vals.empty:
            return float(vals.median())
    return 0.0

def build_input_form():
    # Prefer the exact feature order saved with the model.
    if feature_columns:
        cols = list(feature_columns)
    elif dataset is not None:
        target = "Loan_Approved"
        cols = [c for c in dataset.columns if c not in {target, "Applicant_ID", "Loan_ID"}]
    else:
        cols = []
    if not cols:
        st.error("No feature columns found. Check the saved model metadata or dataset.")
        return None

    # If metadata is missing, infer types from dataset.
    numeric = set(numeric_columns)
    categorical = set(categorical_columns)
    if not numeric and dataset is not None:
        numeric = {c for c in cols if c in dataset.columns and pd.api.types.is_numeric_dtype(dataset[c])}
    if not categorical:
        categorical = set(cols) - numeric

    values = {}
    # Sections are visual grouping only; actual feature names/order stay unchanged.
    numeric_cols = [c for c in cols if c in numeric]
    categorical_cols = [c for c in cols if c not in numeric]
    with st.form("loan_application_form", clear_on_submit=False):
        st.markdown('<div class="mini-label">01 / Applicant profile</div>', unsafe_allow_html=True)
        st.markdown("### Personal & financial information")
        left, right = st.columns(2, gap="large")
        for i, col in enumerate(categorical_cols):
            target_col = left if i % 2 == 0 else right
            with target_col:
                options = []
                if dataset is not None and col in dataset.columns:
                    options = dataset[col].dropna().astype(str).unique().tolist()
                if options and len(options) <= 40:
                    values[col] = st.selectbox(pretty_feature(col), options=options, key=f"app_{col}")
                else:
                    values[col] = st.text_input(pretty_feature(col), key=f"app_{col}")
        st.markdown("---")
        st.markdown('<div class="mini-label">02 / Loan & affordability</div>', unsafe_allow_html=True)
        st.markdown("### Requested loan details")
        numeric_cols = numeric_cols or [c for c in cols if c not in categorical_cols]
        num_left, num_right = st.columns(2, gap="large")
        for i, col in enumerate(numeric_cols):
            target_col = num_left if i % 2 == 0 else num_right
            with target_col:
                default = numeric_default(col)
                # Keep UI fields usable and safe for integer-looking data.
                if float(default).is_integer():
                    values[col] = st.number_input(pretty_feature(col), value=int(default), step=1, key=f"app_{col}")
                else:
                    values[col] = st.number_input(pretty_feature(col), value=float(default), step=0.1, key=f"app_{col}")
        st.markdown("---")
        applicant_name = st.text_input("Applicant reference name (optional)", placeholder="e.g. Applicant 001")
        submitted = st.form_submit_button("Run eligibility assessment  →", type="primary", use_container_width=True)
    if submitted:
        # Pipeline expects raw feature names; keep exactly the stored order.
        frame = pd.DataFrame([{col: values.get(col, np.nan) for col in cols}], columns=cols)
        return {"submitted": True, "frame": frame, "name": applicant_name.strip()}
    return {"submitted": False}

def predict_and_store(form_result):
    if model is None:
        st.error(model_error or "Model is not available.")
        return
    frame = form_result["frame"]
    try:
        prediction = int(model.predict(frame)[0])
        confidence = probability_for_positive(model, frame, prediction)
        # A prediction label is based on the project's 0/1 target mapping.
        label = status_label(prediction)
        row = {
            "Application ID": f"CW-{datetime.now().strftime('%y%m%d')}-{len(st.session_state.applications)+1:04d}",
            "Applicant": form_result.get("name") or f"Applicant {len(st.session_state.applications)+1:03d}",
            "Decision": label,
            "Approval probability": round(confidence * 100, 2),
            "Created at": datetime.now().strftime("%d %b %Y, %I:%M %p"),
            "Input": frame.iloc[0].to_dict(),
            "Prediction": prediction,
        }
        st.session_state.applications.insert(0, row)
        st.session_state.latest_result = row
        st.session_state.active_page = "Decision report"
        st.rerun()
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.info("Confirm that the form fields match the features used to train the saved model.")

def page_header(kicker, title, subtitle):
    st.markdown(f'<div class="eyebrow">{kicker}</div><div class="page-title">{title}</div><div class="page-sub">{subtitle}</div>', unsafe_allow_html=True)

def render_nav():
    left, right = st.columns([2.2, 3.8], vertical_alignment="center")
    with left:
        st.markdown("""
        <div class="brand">
          <div class="brand-mark">◈</div>
          <div><div class="brand-name">CreditWise</div><div class="brand-sub">Loan intelligence</div></div>
        </div>""", unsafe_allow_html=True)
    with right:
        nav_cols = st.columns(5, gap="small")
        pages = ["Overview", "New application", "Applications", "Analytics", "Model report"]
        for col, page in zip(nav_cols, pages):
            with col:
                if st.button(page, key=f"nav_{page}", use_container_width=True,
                             type="primary" if st.session_state.active_page == page else "secondary"):
                    st.session_state.active_page = page
                    st.rerun()
    st.markdown('<div class="topline"></div>', unsafe_allow_html=True)

def render_overview():
    page_header("WORKSPACE / OVERVIEW", "Good to see you.", "Your loan assessment workspace, at a glance.")
    st.markdown("""
    <div class="hero">
      <div class="hero-kicker">CreditWise Intelligence</div>
      <div class="hero-title">Make every application<br>easier to understand.</div>
      <div class="hero-copy">Review applicant information, run a model-based eligibility estimate, and keep your assessment history organized in one workspace.</div>
      <div class="hero-chip">◈ &nbsp; Machine-learning assisted assessment</div>
    </div>""", unsafe_allow_html=True)

    apps = st.session_state.applications
    total = len(apps)
    approved = sum(1 for a in apps if a["Prediction"] == 1)
    not_approved = total - approved
    approval_rate = (approved / total * 100) if total else 0

    k1, k2, k3, k4 = st.columns(4, gap="medium")
    kpis = [
        ("ASSESSMENTS RUN", fmt_int(total), "Current browser session"),
        ("APPROVED ESTIMATES", fmt_int(approved), "Model output = 1"),
        ("NOT APPROVED", fmt_int(not_approved), "Model output = 0"),
        ("APPROVAL SHARE", f"{approval_rate:.1f}%", "Among session assessments"),
    ]
    for col, (label, value, foot) in zip([k1,k2,k3,k4], kpis):
        with col:
            st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-foot">{foot}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-head"><div class="section-title">Your workspace</div><div class="section-note">Choose what you want to do next</div></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown('<div class="panel"><div class="mini-label">01 / START</div><div class="panel-title" style="margin-top:10px">Assess a new application</div><div class="panel-sub" style="margin-top:8px">Enter applicant and loan details to get a model estimate.</div></div>', unsafe_allow_html=True)
        if st.button("New assessment  →", key="overview_new", type="primary", use_container_width=True):
            st.session_state.active_page = "New application"; st.rerun()
    with c2:
        st.markdown('<div class="panel"><div class="mini-label">02 / REVIEW</div><div class="panel-title" style="margin-top:10px">Application register</div><div class="panel-sub" style="margin-top:8px">Search and review assessments created in this session.</div></div>', unsafe_allow_html=True)
        if st.button("View applications  →", key="overview_apps", use_container_width=True):
            st.session_state.active_page = "Applications"; st.rerun()
    with c3:
        st.markdown('<div class="panel"><div class="mini-label">03 / INSIGHTS</div><div class="panel-title" style="margin-top:10px">Model performance</div><div class="panel-sub" style="margin-top:8px">Review the metrics saved during model training.</div></div>', unsafe_allow_html=True)
        if st.button("Open model report  →", key="overview_model", use_container_width=True):
            st.session_state.active_page = "Model report"; st.rerun()

    st.markdown('<div class="section-head"><div class="section-title">Recent assessments</div><div class="section-note">Latest first</div></div>', unsafe_allow_html=True)
    if apps:
        df = pd.DataFrame([{k:v for k,v in a.items() if k not in ("Input","Prediction")} for a in apps[:6]])
        st.dataframe(df, use_container_width=True, hide_index=True,
                     column_config={"Approval probability": st.column_config.ProgressColumn("Approval probability", min_value=0, max_value=100, format="%.1f%%")})
    else:
        st.markdown('<div class="panel"><div class="panel-title">No assessments yet</div><div class="panel-sub" style="margin:8px 0 0">Your first assessment will appear here once you run a prediction.</div></div>', unsafe_allow_html=True)

def render_new_application():
    page_header("APPLICATIONS / NEW", "New loan assessment", "Complete the applicant profile, then run the model against your saved training pipeline.")
    if model is None:
        st.error(model_error or "The model could not be loaded.")
        st.code("Expected file: models/loan_model.pkl")
        return
    st.markdown('<div class="panel" style="margin-bottom:16px"><div class="mini-label">ASSESSMENT WORKFLOW</div><div class="panel-title" style="margin-top:8px">Applicant details <span style="color:#9aa5b5;font-weight:500">　→　 Financial profile　 →　 Model estimate</span></div><div class="panel-sub" style="margin:8px 0 0">The fields below are generated from the feature columns saved with your model.</div></div>', unsafe_allow_html=True)
    form_result = build_input_form()
    if form_result and form_result.get("submitted"):
        predict_and_store(form_result)
    st.caption("Demo note: predictions are estimates from your ML model, not official credit decisions. This app stores assessment history in Streamlit session state only.")

def render_applications():
    page_header("WORKSPACE / REGISTER", "Application register", "Search, filter and export the assessments created during this browser session.")
    apps = st.session_state.applications
    if not apps:
        st.info("No applications yet. Start by creating a new assessment.")
        if st.button("Create first assessment  →", type="primary"):
            st.session_state.active_page = "New application"; st.rerun()
        return
    a, b, c = st.columns([2,1,1])
    with a: search = st.text_input("Search records", placeholder="Search applicant or application ID…")
    with b: decision_filter = st.selectbox("Decision", ["All decisions", "Approved", "Not approved"])
    with c: sort_order = st.selectbox("Sort by", ["Newest first", "Highest probability", "Lowest probability"])
    rows = []
    for app in apps:
        if search.lower() not in (app["Applicant"] + " " + app["Application ID"]).lower(): continue
        if decision_filter != "All decisions" and app["Decision"] != decision_filter: continue
        rows.append({k:v for k,v in app.items() if k not in ("Input","Prediction")})
    if sort_order == "Highest probability": rows.sort(key=lambda x:x["Approval probability"], reverse=True)
    if sort_order == "Lowest probability": rows.sort(key=lambda x:x["Approval probability"])
    st.markdown(f'<div class="section-head"><div class="section-title">{len(rows)} matching records</div><div class="section-note">Session-only data</div></div>', unsafe_allow_html=True)
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True,
                     column_config={"Approval probability": st.column_config.ProgressColumn("Approval probability", min_value=0, max_value=100, format="%.1f%%")})
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Export visible records as CSV  ↓", data=csv, file_name="creditwise_applications.csv", mime="text/csv")
    else:
        st.warning("No records match these filters.")
    with st.expander("Manage session data"):
        st.warning("Clearing history removes records from this active session only.")
        if st.button("Clear all session applications", type="secondary"):
            st.session_state.applications = []
            st.session_state.latest_result = None
            st.rerun()

def render_decision_report():
    page_header("ASSESSMENT / RESULT", "Decision report", "A transparent summary of the most recent model estimate.")
    result = st.session_state.latest_result
    if not result:
        st.info("No result to display yet. Run a new assessment first.")
        if st.button("Start an assessment  →", type="primary"):
            st.session_state.active_page = "New application"; st.rerun()
        return
    approved = result["Prediction"] == 1
    color = "#16836b" if approved else "#c64d49"
    st.markdown(f"""
    <div class="hero" style="background:linear-gradient(112deg,#172b4d 0%,#253f78 100%)">
      <div class="hero-kicker">ASSESSMENT RESULT · {result['Application ID']}</div>
      <div class="hero-title" style="font-size:29px">{result['Decision']}</div>
      <div class="hero-copy">Applicant reference: {result['Applicant']}<br>Created {result['Created at']}</div>
      <div class="hero-chip">◈ &nbsp; Model-generated estimate · Not an official lending decision</div>
    </div>""", unsafe_allow_html=True)
    left, right = st.columns([1.05,1.4], gap="large")
    with left:
        st.markdown('<div class="panel"><div class="panel-title">Model score</div><div class="panel-sub">Probability of the positive class (target value 1)</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Indicator(mode="gauge+number", value=result["Approval probability"],
            number={"suffix":"%","font":{"size":32,"color":"#172b4d"}},
            gauge={"axis":{"range":[0,100],"tickwidth":1,"tickcolor":"#dfe5ef"},
                   "bar":{"color":"#3457d5","thickness":0.25},
                   "bgcolor":"white","borderwidth":0,
                   "steps":[{"range":[0,40],"color":"#fce9e8"},{"range":[40,70],"color":"#fff3d9"},{"range":[70,100],"color":"#e4f6ef"}]}))
        fig.update_layout(height=245, margin=dict(l=18,r=18,t=18,b=0), paper_bgcolor="white", font={"family":"DM Sans"})
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel"><div class="panel-title">Applicant input summary</div><div class="panel-sub">Values passed into the model</div>', unsafe_allow_html=True)
        inp = result.get("Input", {})
        summary = pd.DataFrame([{"Field":pretty_feature(k),"Value":str(v)} for k,v in inp.items()])
        st.dataframe(summary, use_container_width=True, hide_index=True, height=285)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("")
    if st.button("Create another assessment  →", type="primary"):
        st.session_state.active_page = "New application"; st.rerun()

def render_analytics():
    page_header("INSIGHTS / ANALYTICS", "Portfolio insights", "Explore the training dataset and the assessments made in this session.")
    apps = st.session_state.applications
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown('<div class="panel"><div class="panel-title">Session decisions</div><div class="panel-sub">Distribution of model estimates created in this session</div>', unsafe_allow_html=True)
        if apps:
            counts = pd.Series([a["Decision"] for a in apps]).value_counts().rename_axis("Decision").reset_index(name="Applications")
            fig = px.pie(counts, names="Decision", values="Applications", hole=.68,
                         color="Decision", color_discrete_map={"Approved":"#16836b","Not approved":"#e26d68"})
            fig.update_layout(height=310, margin=dict(l=0,r=0,t=10,b=0), legend_title=None, paper_bgcolor="white")
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        else:
            st.info("Run at least one assessment to populate session analytics.")
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel"><div class="panel-title">Training dataset target</div><div class="panel-sub">Observed target distribution in the original CSV</div>', unsafe_allow_html=True)
        if dataset is not None and "Loan_Approved" in dataset.columns:
            target = dataset["Loan_Approved"].astype(str).value_counts().rename_axis("Outcome").reset_index(name="Records")
            fig = px.bar(target, x="Outcome", y="Records", text="Records", color="Outcome",
                         color_discrete_sequence=["#3457d5"])
            fig.update_layout(height=310, margin=dict(l=0,r=0,t=10,b=0), showlegend=False,
                              paper_bgcolor="white", plot_bgcolor="white", xaxis_title=None, yaxis_title="Records")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        elif dataset is not None:
            st.warning("Target column 'Loan_Approved' was not found in the dataset.")
        else:
            st.warning(data_error or "Dataset unavailable.")
        st.markdown('</div>', unsafe_allow_html=True)
    if dataset is not None:
        st.markdown('<div class="section-head"><div class="section-title">Dataset profile</div><div class="section-note">Loaded from local CSV</div></div>', unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        for col, label, val in [(c1,"ROWS",len(dataset)),(c2,"COLUMNS",len(dataset.columns)),(c3,"MISSING CELLS",int(dataset.isna().sum().sum()))]:
            with col:
                st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{fmt_int(val)}</div><div class="kpi-foot">Source dataset</div></div>', unsafe_allow_html=True)
        with st.expander("Preview dataset"):
            st.dataframe(dataset.head(100), use_container_width=True, hide_index=True)

def render_model_report():
    page_header("MODEL / PERFORMANCE", "Model report", "Metrics stored with your trained model artifact, plus details about the loaded files.")
    if model is None:
        st.error(model_error or "Model not available.")
        return
    # Display metrics only if supplied by training artifact; do not invent scores.
    if metrics:
        cols = st.columns(min(4, len(metrics)))
        for col, (key, value) in zip(cols, metrics.items()):
            with col:
                try:
                    number = float(value)
                    display = f"{number:.3f}" if abs(number) <= 1 else f"{number:.2f}"
                except Exception:
                    display = str(value)
                st.markdown(f'<div class="kpi"><div class="kpi-label">{pretty_feature(key)}</div><div class="kpi-value" style="font-size:24px">{display}</div><div class="kpi-foot">Saved training metric</div></div>', unsafe_allow_html=True)
    else:
        st.info("No metrics were saved inside the model artifact. This report will not invent performance values.")
    st.markdown('<div class="section-head"><div class="section-title">Model configuration</div></div>', unsafe_allow_html=True)
    a,b = st.columns(2, gap="large")
    with a:
        st.markdown('<div class="panel"><div class="panel-title">Loaded artifact</div>', unsafe_allow_html=True)
        st.write("**Model file**", str(MODEL_PATH.relative_to(ROOT)))
        st.write("**Estimator**", type(model).__name__)
        st.write("**Feature count**", len(feature_columns) if feature_columns else "Not saved in artifact")
        st.write("**Numeric columns**", len(numeric_columns) if numeric_columns else "Not saved in artifact")
        st.write("**Categorical columns**", len(categorical_columns) if categorical_columns else "Not saved in artifact")
        st.markdown('</div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="panel"><div class="panel-title">Dataset source</div>', unsafe_allow_html=True)
        if dataset is not None:
            st.write("**File**", DATA_PATH.name)
            st.write("**Rows**", fmt_int(len(dataset)))
            st.write("**Columns**", fmt_int(len(dataset.columns)))
            st.write("**Target**", "Loan_Approved" if "Loan_Approved" in dataset.columns else "Not detected")
        else:
            st.error(data_error or "Dataset not available")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-head"><div class="section-title">Important interpretation notes</div></div>', unsafe_allow_html=True)
    # st.warning("A model probability is not a guarantee that a lender will approve a loan. This educational project is not connected to a bank or credit bureau. Evaluate accuracy, precision, recall, calibration, fairness, and data quality before any real lending use.")

# ---------- App shell ----------
render_nav()
if model_error:
    st.warning(f"Model setup: {model_error}")
if data_error:
    st.caption(f"Dataset note: {data_error}")

routes = {
    "Overview": render_overview,
    "New application": render_new_application,
    "Applications": render_applications,
    "Decision report": render_decision_report,
    "Analytics": render_analytics,
    "Model report": render_model_report,
}
routes.get(st.session_state.active_page, render_overview)()
# st.markdown('<div class="footer">CREDITWISE · LOAN INTELLIGENCE WORKSPACE &nbsp; • &nbsp; Educational ML demo, not a lending decision engine</div>', unsafe_allow_html=True)
