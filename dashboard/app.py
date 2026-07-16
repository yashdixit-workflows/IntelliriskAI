import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import json
import requests
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agents.coordinator_agent import CoordinatorAgent

try:
    from streamlit_lottie import st_lottie
    LOTTIE_AVAILABLE = True
except ImportError:
    LOTTIE_AVAILABLE = False

try:
    from streamlit_pills import pills
    PILLS_AVAILABLE = True
except ImportError:
    PILLS_AVAILABLE = False

st.set_page_config(page_title="IntelliRisk AI", page_icon="hexagon", layout="wide", initial_sidebar_state="expanded")


def load_lottie_inline(json_str):
    try:
        return json.loads(json_str)
    except Exception:
        return None

LOTTIE_BARS = '{"v":"5.7.14","fr":30,"ip":0,"op":90,"w":120,"h":120,"nm":"bars","layers":[{"ddd":0,"ind":1,"ty":4,"nm":"b","sr":1,"ks":{"o":{"a":0,"k":100},"p":{"a":0,"k":[60,60,0]},"a":{"a":0,"k":[0,0,0]},"s":{"a":0,"k":[100,100,100]}},"ao":0,"shapes":[{"ty":"rc","nm":"b1","p":{"a":0,"k":[-25,10]},"s":{"a":1,"k":[{"i":{"x":[0.5],"y":[1]},"o":{"x":[0.5],"y":[0]},"t":0,"s":[30,0]},{"t":25,"s":[30,50]}]},"r":{"a":0,"k":4}},{"ty":"fl","c":{"a":0,"k":[0.145,0.38,0.925,1]},"o":{"a":0,"k":100},"r":1,"nm":"f1"},{"ty":"rc","nm":"b2","p":{"a":0,"k":[0,15]},"s":{"a":1,"k":[{"i":{"x":[0.5],"y":[1]},"o":{"x":[0.5],"y":[0]},"t":8,"s":[30,0]},{"t":38,"s":[30,30]}]},"r":{"a":0,"k":4}},{"ty":"fl","c":{"a":0,"k":[0.086,0.639,0.29,1]},"o":{"a":0,"k":100},"r":1,"nm":"f2"},{"ty":"rc","nm":"b3","p":{"a":0,"k":[25,5]},"s":{"a":1,"k":[{"i":{"x":[0.5],"y":[1]},"o":{"x":[0.5],"y":[0]},"t":16,"s":[30,0]},{"t":50,"s":[30,60]}]},"r":{"a":0,"k":4}},{"ty":"fl","c":{"a":0,"k":[0.851,0.467,0.024,1]},"o":{"a":0,"k":100},"r":1,"nm":"f3"}],"ip":0,"op":90,"st":0,"bm":0}]}'

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
*,*::before,*::after{box-sizing:border-box;}
:root{
  --bg:#f4f3ef;--card:#ffffff;--border:#e0ddd5;--border-s:#c5c0b4;
  --text:#1a1916;--muted:#6b6860;--dim:#a09e98;
  --accent:#2563eb;--al:rgba(37,99,235,.08);--ag:rgba(37,99,235,.18);
  --green:#15803d;--gl:rgba(21,128,61,.08);
  --red:#b91c1c;--rl:rgba(185,28,28,.08);
  --amber:#b45309;--aml:rgba(180,83,9,.08);
  --r:12px;--rl2:18px;
  --ss:0 1px 3px rgba(0,0,0,.06),0 1px 2px rgba(0,0,0,.04);
  --sm:0 4px 16px rgba(0,0,0,.07),0 2px 6px rgba(0,0,0,.04);
  --sh:0 8px 32px rgba(37,99,235,.13),0 4px 12px rgba(0,0,0,.06);
  --tr:all 0.22s cubic-bezier(.4,0,.2,1);
}
header[data-testid="stHeader"] {
  background-color: transparent !important;
  box-shadow: none !important;
  pointer-events: none !important;
}
footer,#MainMenu{display:none!important;}
[data-testid="stToolbar"]{display:none!important;}
[data-testid="stDecoration"]{display:none!important;}
header[data-testid="stHeader"] button { display: none !important; }
/* ── Permanently hide ALL sidebar toggle controls ── */
button[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebarCollapseButton"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:var(--border-s);border-radius:4px;}
.stApp,[data-testid="stAppViewContainer"]{background:var(--bg)!important;font-family:'Inter',-apple-system,sans-serif!important;color:var(--text)!important;}
.block-container{padding:0!important;max-width:100%!important;}

/* LANDING */
.land{min-height:100vh;background:#f2f0eb;background-image:linear-gradient(rgba(37,99,235,.022) 1px,transparent 1px),linear-gradient(90deg,rgba(37,99,235,.022) 1px,transparent 1px),radial-gradient(ellipse at 22% 35%,rgba(37,99,235,.07) 0%,transparent 55%),radial-gradient(ellipse at 78% 10%,rgba(37,99,235,.05) 0%,transparent 45%);background-size:44px 44px,44px 44px,100% 100%,100% 100%;}
.hero{display:flex;flex-direction:column;align-items:center;text-align:center;padding:5.5rem 2rem 2.5rem;}
.eyebrow{font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.12em;color:var(--accent);background:var(--al);border:1px solid rgba(37,99,235,.15);padding:3px 12px;border-radius:20px;margin-bottom:1.2rem;display:inline-block;}
.hero-h{font-size:clamp(2.2rem,5vw,3.6rem);font-weight:900;color:var(--text);line-height:1.1;letter-spacing:-.04em;margin-bottom:1rem;}
.hero-h span{color:var(--accent);}
.hero-sub{font-size:.97rem;color:var(--muted);max-width:560px;line-height:1.65;margin:0 auto 2.2rem;}
.logo-row{display:inline-flex;align-items:center;gap:.6rem;margin-bottom:.4rem;}
.hex{width:36px;height:36px;background:var(--accent);clip-path:polygon(50% 0%,100% 25%,100% 75%,50% 100%,0% 75%,0% 25%);display:flex;align-items:center;justify-content:center;color:#fff;font-weight:900;font-size:.9rem;box-shadow:0 4px 14px rgba(37,99,235,.35);}

/* FEATURE CARDS */
.fc{background:var(--card);border:1px solid var(--border);border-radius:var(--rl2);padding:1.5rem 1.3rem;box-shadow:var(--sm);transition:var(--tr);cursor:default;position:relative;overflow:hidden;}
.fc::before{content:'';position:absolute;inset:0;background:linear-gradient(135deg,rgba(37,99,235,.04) 0%,transparent 60%);opacity:0;transition:var(--tr);}
.fc:hover{box-shadow:var(--sh);transform:translateY(-4px);border-color:rgba(37,99,235,.2);}
.fc:hover::before{opacity:1;}
.fc-chip{display:inline-block;font-size:.6rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;padding:2px 7px;border-radius:20px;background:var(--al);color:var(--accent);margin-bottom:.55rem;border:1px solid rgba(37,99,235,.15);}
.fc-icon{width:40px;height:40px;background:var(--al);border:1px solid rgba(37,99,235,.15);border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:1.1rem;margin-bottom:.8rem;}
.fc-t{font-size:.87rem;font-weight:700;color:var(--text);margin-bottom:.35rem;letter-spacing:-.01em;}
.fc-d{font-size:.75rem;color:var(--muted);line-height:1.55;}

/* STATUS BAR */
.sbar{display:flex;align-items:center;justify-content:center;gap:1.6rem;font-size:.75rem;color:var(--muted);padding:1.1rem 2rem;border-top:1px solid var(--border);margin-top:2rem;}
.sdot{width:7px;height:7px;border-radius:50%;background:#22c55e;display:inline-block;margin-right:5px;box-shadow:0 0 8px #22c55e;animation:pdot 2s ease-in-out infinite;}
@keyframes pdot{0%,100%{opacity:1;transform:scale(1);}50%{opacity:.55;transform:scale(1.35);}}

/* APP SHELL */
.stApp{background:#f4f3ef!important;background-image:linear-gradient(rgba(37,99,235,.016) 1px,transparent 1px),linear-gradient(90deg,rgba(37,99,235,.016) 1px,transparent 1px)!important;background-size:38px 38px!important;}

/* NAV */
.tnav{background:rgba(255,255,255,.88);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);border-bottom:1px solid var(--border);padding:.65rem 2.4rem;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100;box-shadow:0 2px 12px rgba(0,0,0,.04);}
.nbrand{display:flex;align-items:center;gap:.55rem;font-size:1.05rem;font-weight:800;color:var(--text);letter-spacing:-.03em;}
.nhex{width:26px;height:26px;background:var(--accent);clip-path:polygon(50% 0%,100% 25%,100% 75%,50% 100%,0% 75%,0% 25%);}
.nstat{font-size:.68rem;color:#15803d;background:rgba(21,128,61,.08);border:1px solid rgba(21,128,61,.18);padding:3px 9px;border-radius:20px;font-weight:600;}

/* TABS */
[data-testid="stTabs"]>div:first-child{background:rgba(255,255,255,.75)!important;border:1px solid var(--border)!important;border-radius:12px!important;padding:4px!important;backdrop-filter:blur(8px);gap:2px!important;}
button[data-baseweb="tab"]{background:transparent!important;color:var(--muted)!important;font-size:.79rem!important;font-weight:500!important;padding:.38rem .82rem!important;border:1px solid transparent!important;border-radius:8px!important;transition:var(--tr)!important;letter-spacing:-.01em!important;}
button[data-baseweb="tab"]:hover{background:var(--card)!important;color:var(--text)!important;border-color:var(--border)!important;box-shadow:var(--ss)!important;}
button[data-baseweb="tab"][aria-selected="true"]{background:var(--card)!important;color:var(--accent)!important;border-color:rgba(37,99,235,.22)!important;font-weight:600!important;box-shadow:var(--ss)!important;}
[data-baseweb="tab-highlight"],[data-baseweb="tab-border"]{display:none!important;}

/* KPI */
.kc{background:var(--card);border:1px solid var(--border);border-radius:var(--r);padding:1.15rem 1.35rem;box-shadow:var(--ss);transition:var(--tr);position:relative;overflow:hidden;}
.kc::after{content:'';position:absolute;top:0;left:0;width:3px;height:100%;background:var(--accent);opacity:0;transition:var(--tr);}
.kc:hover{box-shadow:var(--sh);transform:translateY(-2px);}
.kc:hover::after{opacity:1;}
.kl{font-size:.68rem;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.45rem;}
.kv{font-size:1.6rem;font-weight:800;color:var(--text);letter-spacing:-.03em;line-height:1;}
.ks{font-size:.7rem;color:var(--muted);margin-top:.3rem;}

/* CHART CARD */
.cc{background:var(--card);border:1px solid var(--border);border-radius:var(--r);padding:1.25rem 1.35rem;box-shadow:var(--ss);margin-bottom:1.15rem;transition:var(--tr);}
.cc:hover{box-shadow:var(--sm);border-color:rgba(37,99,235,.12);}
.ct{font-size:.83rem;font-weight:700;color:var(--text);letter-spacing:-.01em;margin-bottom:.18rem;}
.cs{font-size:.7rem;color:var(--dim);margin-bottom:.85rem;}

/* BADGES */
.badge{display:inline-block;padding:2px 8px;border-radius:6px;font-size:.67rem;font-weight:700;text-transform:uppercase;letter-spacing:.05em;}
.bg{color:var(--green);background:var(--gl);}
.br{color:var(--red);background:var(--rl);}
.ba{color:var(--amber);background:var(--aml);}
.bb{color:var(--accent);background:var(--al);}

/* DATA TABLE */
.dt{width:100%;border-collapse:separate;border-spacing:0;font-size:.77rem;}
.dt th{text-align:left;padding:.52rem .78rem;color:var(--muted);font-weight:600;font-size:.67rem;text-transform:uppercase;letter-spacing:.06em;border-bottom:1px solid var(--border);background:#fafaf8;}
.dt td{padding:.58rem .78rem;color:var(--text);border-bottom:1px solid rgba(0,0,0,.035);}
.dt tr:hover td{background:#fafaf8;}

/* SIDEBAR */
/* SIDEBAR — permanently visible, no collapse */
section[data-testid="stSidebar"] {
  display: flex !important;
  flex-direction: column !important;
  visibility: visible !important;
  opacity: 1 !important;
  transform: none !important;
  min-width: 290px !important;
  width: 290px !important;
  background: #ffffff !important;
  border-right: 1px solid var(--border) !important;
  box-shadow: 2px 0 12px rgba(0,0,0,.05) !important;
}
section[data-testid="stSidebar"] > div:first-child {
  padding-top: 1rem !important;
}

section[data-testid="stSidebar"] .stButton>button{background:var(--accent)!important;color:#fff!important;font-weight:700!important;border:none!important;box-shadow:0 4px 14px rgba(37,99,235,.25)!important;width:100%!important;}
section[data-testid="stSidebar"] .stButton>button:hover{background:#1d4ed8!important;box-shadow:0 6px 20px rgba(37,99,235,.35)!important;}
/* upload zone */
.upload-zone{border:2px dashed var(--border);border-radius:12px;padding:1.2rem 1rem;text-align:center;background:var(--al);transition:var(--tr);}
.upload-zone:hover{border-color:var(--accent);background:rgba(37,99,235,.05);}
/* info chip row */
.chip-row{display:flex;flex-wrap:wrap;gap:.4rem;margin:.5rem 0;}
.chip{display:inline-flex;align-items:center;gap:.3rem;font-size:.68rem;font-weight:600;padding:3px 9px;border-radius:20px;border:1px solid var(--border);background:var(--card);color:var(--muted);}
/* pipeline steps */
.step-row{display:flex;align-items:flex-start;gap:.75rem;margin-bottom:.85rem;}
.step-num{width:26px;height:26px;border-radius:50%;background:var(--accent);color:#fff;font-size:.72rem;font-weight:800;display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:.1rem;}
.step-body{flex:1;}
.step-title{font-size:.8rem;font-weight:700;color:var(--text);margin-bottom:.1rem;}
.step-desc{font-size:.72rem;color:var(--muted);line-height:1.45;}
/* format card */
.fmt-card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:.85rem 1rem;display:flex;align-items:center;gap:.7rem;transition:var(--tr);}
.fmt-card:hover{border-color:rgba(37,99,235,.25);box-shadow:var(--ss);}
.fmt-icon{font-size:1.3rem;}
.fmt-title{font-size:.78rem;font-weight:700;color:var(--text);}
.fmt-desc{font-size:.68rem;color:var(--muted);}

/* CHAT */
.cu{background:var(--accent);color:#fff;border-radius:14px 14px 4px 14px;padding:.72rem .95rem;margin-bottom:.55rem;font-size:.82rem;max-width:82%;margin-left:auto;box-shadow:var(--ss);}
.ca{background:var(--card);color:var(--text);border:1px solid var(--border);border-radius:14px 14px 14px 4px;padding:.72rem .95rem;margin-bottom:.55rem;font-size:.82rem;max-width:82%;box-shadow:var(--ss);}

/* DIVIDER */
.div-row{display:flex;align-items:center;gap:.75rem;margin:1.4rem 0 .9rem;}
.div-label{font-size:.75rem;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;white-space:nowrap;}
.div-line{flex:1;height:1px;background:var(--border);}

/* BUTTONS */
.stButton>button{font-family:'Inter',sans-serif!important;font-weight:600!important;border-radius:9px!important;transition:var(--tr)!important;}
.stButton>button:hover{transform:translateY(-1px)!important;box-shadow:var(--sm)!important;}
.pad{padding:1.8rem 2.4rem;}
.accent-c{border-left:3.5px solid var(--accent)!important;}
.green-c{border-left:3.5px solid var(--green)!important;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

if "coordinator" not in st.session_state:
    st.session_state.coordinator = CoordinatorAgent()
    st.session_state.is_initialized = False

# Auto-restore session from query parameters on refresh
if not st.session_state.is_initialized:
    fpath_param = st.query_params.get("fpath")
    if fpath_param and os.path.exists(fpath_param):
        try:
            ok, err = st.session_state.coordinator.initialize_pipeline(fpath_param)
            if ok:
                st.session_state.is_initialized = True
            else:
                st.query_params.pop("fpath", None)
        except Exception:
            st.query_params.pop("fpath", None)

if "current_page" not in st.session_state:
    st.session_state.current_page = st.query_params.get("page", "landing")



if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


coord = st.session_state.coordinator

PLOT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter,sans-serif", color="#6b6860", size=11),
    margin=dict(l=40, r=20, t=30, b=30),
    xaxis=dict(gridcolor="rgba(0,0,0,.04)", zerolinecolor="rgba(0,0,0,.04)", tickfont=dict(size=10)),
    yaxis=dict(gridcolor="rgba(0,0,0,.04)", zerolinecolor="rgba(0,0,0,.04)", tickfont=dict(size=10)),
)

def kpi(label, value, sub=""):
    s = f'<div class="ks">{sub}</div>' if sub else ""
    st.markdown(f'<div class="kc"><div class="kl">{label}</div><div class="kv">{value}</div>{s}</div>', unsafe_allow_html=True)

def card(title, sub="", extra_class=""):
    s = f'<div class="cs">{sub}</div>' if sub else ""
    st.markdown(f'<div class="cc {extra_class}"><div class="ct">{title}</div>{s}', unsafe_allow_html=True)

def end():
    st.markdown("</div>", unsafe_allow_html=True)

def divider(label):
    st.markdown(f'<div class="div-row"><div class="div-line"></div><div class="div-label">{label}</div><div class="div-line"></div></div>', unsafe_allow_html=True)

def show_preview_page():
    st.markdown('<div class="pad">', unsafe_allow_html=True)
    st.markdown('''
    <h2 style="font-size:1.65rem;font-weight:900;letter-spacing:-.04em;color:#1a1916;margin:.2rem 0 .5rem;">
      📊 Portfolio Data Preview & Editor</h2>
    <p style="color:#6b6860;font-size:.88rem;line-height:1.65;margin-bottom:1.5rem;">
      Inspect a sample of your dataset, filter and search records, edit cell values directly in the table, and apply your changes to update predictions and KPIs throughout the platform.
    </p>''', unsafe_allow_html=True)
    
    state = st.session_state.coordinator.state
    df_clean = state.get("cleaned_df")
    schema = state.get("schema", {})
    
    if df_clean is None:
        st.info("📂 No active dataset loaded. Please select a dataset and click '🚀 Initialize Agents & Pipeline' in the sidebar to get started.")
        st.markdown("</div>", unsafe_allow_html=True)
        return
        
    c1, c2, c3 = st.columns([2, 3, 3])
    with c1:
        sample_size = st.selectbox("Sample size (rows)", [50, 100, 200, 500], index=1)
    with c2:
        all_cols = list(df_clean.columns)
        default_cols = all_cols[:10] if len(all_cols) >= 10 else all_cols
        selected_cols = st.multiselect("Columns to display", all_cols, default=default_cols)
    with c3:
        search_query = st.text_input("🔍 Search rows", placeholder="Type to filter...")

    df_sample = df_clean.copy()
    
    idc = schema.get("identifier_column", "Customer_ID")
    if idc not in selected_cols and idc in df_sample.columns:
        selected_cols = [idc] + selected_cols
        
    df_filtered = df_sample[selected_cols]
    
    if search_query:
        mask = df_filtered.astype(str).apply(lambda x: x.str.contains(search_query, case=False)).any(axis=1)
        df_filtered = df_filtered[mask]
        
    df_filtered = df_filtered.head(sample_size)
    
    card("Interactive Data Editor", f"Showing {len(df_filtered)} of {len(df_clean)} rows. Double-click any cell to edit.")
    
    edited_df = st.data_editor(
        df_filtered,
        use_container_width=True,
        num_rows="dynamic",
        key="data_editor_component"
    )
    end()
    
    st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
    col_act1, col_act2, col_act3 = st.columns([2, 2, 4])
    with col_act1:
        if st.button("💾 Save & Apply Changes", use_container_width=True, key="save_edits_btn"):
            with st.spinner("Applying edits and updating analytics..."):
                orig_indices = set(df_filtered.index)
                edited_indices = set(edited_df.index)
                deleted_indices = orig_indices - edited_indices
                
                # Update modified values and add new rows
                for idx in edited_df.index:
                    if idx in df_clean.index:
                        df_clean.loc[idx, edited_df.columns] = edited_df.loc[idx]
                    else:
                        new_row = pd.DataFrame([edited_df.loc[idx]], columns=edited_df.columns)
                        df_clean = pd.concat([df_clean, new_row], ignore_index=True)
                
                if deleted_indices:
                    df_clean = df_clean.drop(index=list(deleted_indices))
                
                st.session_state.coordinator.state["cleaned_df"] = df_clean
                st.session_state.coordinator._calculate_kpis(df_clean)
                
                active_model = st.session_state.coordinator.state.get("active_model_name")
                if active_model:
                    st.session_state.coordinator.run_predictions()
                    
                st.success("✓ Changes saved successfully! KPIs and predictions updated.")
                st.rerun()
                
    with col_act2:
        if st.button("🔄 Reset to Original Data", use_container_width=True, key="reset_edits_btn"):
            with st.spinner("Resetting to raw data..."):
                raw_df = state.get("raw_df")
                if raw_df is not None:
                    cleaned_df = st.session_state.coordinator.data_agent.clean_dataset(raw_df)
                    st.session_state.coordinator.state["cleaned_df"] = cleaned_df
                    st.session_state.coordinator._calculate_kpis(cleaned_df)
                    
                    active_model = st.session_state.coordinator.state.get("active_model_name")
                    if active_model:
                        st.session_state.coordinator.run_predictions()
                        
                    st.success("✓ Reset complete.")
                    st.rerun()
                else:
                    st.error("No raw dataset available to reset to.")
                    
    with col_act3:
        csv_data = edited_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Edited Sample (CSV)",
            data=csv_data,
            file_name="edited_portfolio_sample.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    st.markdown("</div>", unsafe_allow_html=True)


# ─── LANDING ───
if st.session_state.current_page == "landing":
    st.markdown("""
<style>
.stApp {
  background: #0f172a !important;
  background-image: 
    radial-gradient(at 0% 0%, rgba(59, 130, 246, 0.22) 0px, transparent 50%),
    radial-gradient(at 50% 0%, rgba(139, 92, 246, 0.22) 0px, transparent 50%),
    radial-gradient(at 100% 0%, rgba(236, 72, 153, 0.22) 0px, transparent 50%),
    radial-gradient(at 0% 100%, rgba(20, 184, 166, 0.15) 0px, transparent 50%),
    radial-gradient(at 100% 100%, rgba(245, 158, 11, 0.15) 0px, transparent 50%) !important;
  color: #f8fafc !important;
}
.land {
  min-height: auto;
  color: #f8fafc !important;
}
.hero {
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  padding: 3.5rem 2rem 0.5rem !important;
}
.logo-row span {
  color: #ffffff !important;
}
.eyebrow {
  font-size: .7rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .12em;
  color: #60a5fa !important;
  background: rgba(59, 130, 246, 0.15) !important;
  border: 1px solid rgba(59, 130, 246, 0.25) !important;
  padding: 3px 12px;
  border-radius: 20px;
  margin-bottom: 1rem;
  display: inline-block;
}
.hero-h {
  font-size: clamp(2.2rem, 5vw, 3.6rem);
  font-weight: 900;
  color: #ffffff !important;
  line-height: 1.1;
  letter-spacing: -.04em;
  margin-bottom: 0.8rem;
}
.hero-h span {
  color: #3b82f6;
}
.hero-sub {
  font-size: .97rem;
  color: #94a3b8 !important;
  max-width: 560px;
  line-height: 1.65;
  margin: 0 auto 0.8rem !important;
}
.fc {
  background: #1e293b !important;
  border: 1px solid #334155 !important;
  border-radius: var(--rl2);
  padding: 1.5rem 1.3rem;
  box-shadow: var(--sm);
  transition: var(--tr);
  cursor: default;
  position: relative;
  overflow: hidden;
}
.fc:hover {
  border-color: rgba(59, 130, 246, 0.4) !important;
  box-shadow: 0 8px 32px rgba(59, 130, 246, 0.15) !important;
  transform: translateY(-4px);
}
.fc-chip {
  display: inline-block;
  font-size: .6rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .07em;
  padding: 2px 7px;
  border-radius: 20px;
  background: rgba(59, 130, 246, 0.15) !important;
  color: #60a5fa !important;
  margin-bottom: .55rem;
  border: 1px solid rgba(59, 130, 246, 0.25) !important;
}
.fc-icon {
  width: 40px;
  height: 40px;
  background: rgba(59, 130, 246, 0.1) !important;
  border: 1px solid rgba(59, 130, 246, 0.2) !important;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.1rem;
  margin-bottom: .8rem;
}
.fc-t {
  font-size: .87rem;
  font-weight: 700;
  color: #ffffff !important;
  margin-bottom: .35rem;
  letter-spacing: -.01em;
}
.fc-d {
  font-size: .75rem;
  color: #94a3b8 !important;
  line-height: 1.55;
}
.sbar {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 1.6rem;
  font-size: .75rem;
  color: #94a3b8 !important;
  padding: 1.1rem 2rem;
  border-top: 1px solid #334155 !important;
  margin-top: 2rem;
}
</style>
<div class="land">
  <div class="hero">
   <div class="logo-row">
    <div class="hex">IR</div>
    <span style="font-size:1rem;font-weight:800;letter-spacing:-.03em;color:#ffffff;">IntelliRisk AI</span>
   </div>
   <div class="eyebrow">Multi-Agent Credit Analytics Platform</div>
   <h1 class="hero-h">Credit Risk Intelligence,<br/><span>Reimagined.</span></h1>
   <p class="hero-sub">An autonomous AI platform that understands your data, profiles risk, trains prediction models, and explains every decision in plain language — all in one workflow.</p>
  </div>
 </div>""", unsafe_allow_html=True)

    if LOTTIE_AVAILABLE:
        ld = load_lottie_inline(LOTTIE_BARS)
        if ld:
            _, lc, _ = st.columns([3,1,3])
            with lc: st_lottie(ld, speed=1.1, height=60, key="hl")

    _, cc, _ = st.columns([2,1,2])
    with cc:
        if st.session_state.is_initialized:
            launch_text = "Go to Operation Dashboard →"
        else:
            launch_text = "Launch Platform →"
            
        if st.button(launch_text, use_container_width=True, key="launch_btn"):
            st.session_state.current_page = "operation"
            st.query_params["page"] = "operation"
            st.rerun()

    st.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)

    feats = [
        ("🧬","SCHEMA","Semantic Schema Understanding","Auto-detects types, PII, target variables, and temporal sequences — zero config."),
        ("🔍","QUALITY","Automated Data Quality","Detects missing values, duplicates, outliers, and applies smart imputation."),
        ("📊","EDA","Exploratory Data Analysis","Correlation matrices, distributions, trend charts, and narrative summaries."),
        ("🤖","PREDICT","Multi-Model Prediction","Random Forest, Logistic Regression, Decision Tree, XGBoost, LightGBM."),
        ("💡","EXPLAIN","AI Explainability Engine","Personalized reason codes per account using global + local feature attribution."),
        ("📥","REPORTS","Executive Report Generation","One-click Word, PDF, and CSV export with KPIs, charts, action registries."),
    ]
    c1,c2,c3 = st.columns(3)
    for i,(icon,chip,title,desc) in enumerate(feats):
        with [c1,c2,c3][i%3]:
            st.markdown(f"""
<div class="fc">
  <div class="fc-chip">{chip}</div>
  <div class="fc-icon">{icon}</div>
  <div class="fc-t">{title}</div>
  <div class="fc-d">{desc}</div>
 </div>""", unsafe_allow_html=True)
            st.markdown("<div style='height:.1rem'></div>", unsafe_allow_html=True)

    st.markdown("""
<div class="sbar">
  <span><span class="sdot"></span>All Systems Operational</span>
  <span>· Multi-Agent Orchestration</span>
  <span>· Rule-Based + LLM Dual Mode</span>
  <span>· IntelliRisk AI v2.0</span>
 </div>""", unsafe_allow_html=True)
    st.stop()

# ─── SIDEBAR ───
with st.sidebar:
    # Brand header
    st.markdown('''
    <div style="display:flex;align-items:center;gap:.55rem;padding:.4rem 0 .8rem;border-bottom:1px solid #e0ddd5;margin-bottom:.9rem;">
      <div style="width:28px;height:28px;background:#2563eb;clip-path:polygon(50% 0%,100% 25%,100% 75%,50% 100%,0% 75%,0% 25%);flex-shrink:0;"></div>
      <div><div style="font-size:.97rem;font-weight:800;letter-spacing:-.03em;color:#1a1916;">IntelliRisk AI</div>
      <div style="font-size:.62rem;color:#a09e98;font-weight:500;">Multi-Agent Analytics v2.0</div></div>
    </div>''', unsafe_allow_html=True)

    # ── AI Mode ──
    st.markdown("**⚙️ AI Mode**")
    api_key = st.text_input("Gemini API Key (optional)", type="password",
                            placeholder="AIza...  leave blank for rule-based mode")
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
        if hasattr(coord,"schema_agent") and coord.schema_agent:
            from agents import get_gemini_client
            cl = get_gemini_client()
            for ag in [coord.schema_agent, coord.data_agent, coord.eda_agent,
                       coord.explainability_agent, coord.recommendation_agent, coord.chat_agent]:
                if ag: ag.client = cl
        st.success("✓ Gemini LLM mode active")
    else:
        st.markdown('''
        <div style="background:rgba(37,99,235,.06);border:1px solid rgba(37,99,235,.18);border-radius:8px;
                    padding:.55rem .75rem;font-size:.72rem;color:#2563eb;margin-bottom:.4rem;">
          ⚡ <strong>Rule-Based Mode</strong> — All agents run locally.<br/>
          Add a Gemini API key above for LLM-powered narratives, chat, and explanations.
        </div>''', unsafe_allow_html=True)

    st.markdown("<div style='height:.3rem'></div>", unsafe_allow_html=True)
    st.divider()

    # ── Dataset Source ──
    st.markdown("**📂 Data Source**")
    src_opts = ["📤 Upload File (CSV/Excel)", "📋 Paste File Path", "📊 Google Sheets"]
    src = st.radio("Choose source", src_opts, label_visibility="collapsed")
    fpath = None
    gsheet_df = None   # holds the fetched Google Sheet dataframe

    if src == src_opts[0]:   # ── File uploader ──────────────────────────────
        st.markdown('<p style="font-size:.72rem;color:#6b6860;margin-bottom:.4rem;">Supports CSV, Excel (.xlsx/.xls)</p>', unsafe_allow_html=True)
        uf = st.file_uploader("Drop or browse file", type=["csv","xlsx","xls"],
                               label_visibility="collapsed")
        if uf:
            td = r"x:\creditguard-ai\datasets"
            os.makedirs(td, exist_ok=True)
            fpath = os.path.join(td, uf.name)
            try:
                with open(fpath, "wb") as f:
                    f.write(uf.getbuffer())
            except PermissionError:
                if not os.path.exists(fpath):
                    st.error("Permission denied: Unable to save uploaded file.")
            sz = round(uf.size/1024, 1)
            st.markdown(f'''
            <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;
                        padding:.55rem .75rem;font-size:.72rem;color:#15803d;">
              ✅ <strong>{uf.name}</strong><br/>
              <span style="color:#6b6860;">📦 {sz} KB · Ready to analyze</span>
            </div>''', unsafe_allow_html=True)
        else:
            st.markdown('''
            <div class="upload-zone">
              <div style="font-size:1.6rem;margin-bottom:.35rem;">📂</div>
              <div style="font-size:.75rem;font-weight:600;color:#1a1916;">Drop files here</div>
              <div style="font-size:.68rem;color:#6b6860;">CSV · XLSX · XLS · up to 200 MB</div>
            </div>''', unsafe_allow_html=True)

    elif src == src_opts[1]:   # ── Paste file path ────────────────────────────
        manual_path = st.text_input("Absolute file path",
                                    placeholder=r"C:\Users\you\data\portfolio.csv")
        if manual_path:
            manual_path = manual_path.strip().strip('"').strip("'")
            if os.path.exists(manual_path):
                fpath = manual_path
                ext = os.path.splitext(manual_path)[1].lower()
                sz = round(os.path.getsize(manual_path)/1024, 1)
                st.markdown(f'''
                <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;
                            padding:.55rem .75rem;font-size:.72rem;color:#15803d;">
                  ✅ <strong>{os.path.basename(manual_path)}</strong><br/>
                  <span style="color:#6b6860;">📦 {sz} KB · {ext.upper()} file detected</span>
                </div>''', unsafe_allow_html=True)
            elif manual_path:
                st.error(f"File not found: {manual_path}")

    else:   # ── Google Sheets ──────────────────────────────────────────────
        st.markdown('''
        <div style="background:rgba(37,99,235,.06);border:1px solid rgba(37,99,235,.18);
                    border-radius:8px;padding:.6rem .8rem;font-size:.72rem;color:#2563eb;margin-bottom:.55rem;">
          📊 <strong>Google Sheets Connector</strong><br/>
          <span style="color:#6b6860;">Paste a share URL. Public sheets work instantly.
          Private sheets need a Service Account JSON.</span>
        </div>''', unsafe_allow_html=True)

        gs_url = st.text_input(
            "Google Sheets URL",
            placeholder="https://docs.google.com/spreadsheets/d/…",
            label_visibility="collapsed",
            key="gs_url_input"
        )

        with st.expander("🔐 Private Sheet? Upload Service Account JSON"):
            sa_file = st.file_uploader(
                "Service Account JSON",
                type=["json"],
                key="sa_json_uploader",
                label_visibility="collapsed"
            )
            gs_sheet_name = st.text_input(
                "Sheet tab name (leave blank for first tab)",
                placeholder="Sheet1",
                key="gs_sheet_name"
            )

        if gs_url and gs_url.strip():
            sa_path = None
            if sa_file:
                sa_path = os.path.join(r"x:\creditguard-ai\datasets", sa_file.name)
                os.makedirs(r"x:\creditguard-ai\datasets", exist_ok=True)
                with open(sa_path, "wb") as f:
                    f.write(sa_file.getbuffer())

            if st.button("🔗 Connect & Load Sheet", use_container_width=True, key="gs_load_btn"):
                with st.spinner("Fetching data from Google Sheets…"):
                    from tools.data_loader import DataLoader as DL
                    df_gs, gs_err = DL.load_google_sheet(
                        gs_url.strip(),
                        creds_json_path=sa_path,
                        sheet_name=gs_sheet_name.strip() or None,
                    )
                    if gs_err:
                        st.error(gs_err)
                    else:
                        # Save to local CSV so the pipeline can use it
                        os.makedirs(r"x:\creditguard-ai\datasets", exist_ok=True)
                        save_path = r"x:\creditguard-ai\datasets\gsheet_import.csv"
                        df_gs.to_csv(save_path, index=False)
                        st.session_state["gs_loaded_path"] = save_path
                        st.success(f"✅ Loaded {len(df_gs):,} rows × {len(df_gs.columns)} columns from Google Sheets!")

        if st.session_state.get("gs_loaded_path") and os.path.exists(st.session_state["gs_loaded_path"]):
            fpath = st.session_state["gs_loaded_path"]
            st.markdown(f'''
            <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;
                        padding:.55rem .75rem;font-size:.72rem;color:#15803d;">
              ✅ <strong>Google Sheet ready</strong><br/>
              <span style="color:#6b6860;">📦 Cached locally as gsheet_import.csv</span>
            </div>''', unsafe_allow_html=True)



    st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)
    st.divider()

    # ── Initialize ──
    init_disabled = fpath is None
    if fpath:
        if st.button("🚀 Initialize Agents & Pipeline", use_container_width=True, key="init_btn"):
            with st.spinner("Orchestrating all agents…"):
                ok, err = coord.initialize_pipeline(fpath)
                if ok:
                    st.session_state.is_initialized = True
                    st.session_state.current_page = "operation"
                    st.query_params["page"] = "operation"
                    st.query_params["fpath"] = fpath
                    st.success("✓ Pipeline initialized!")
                    st.rerun()
                else:
                    st.error(f"Initialization failed: {err}")
    else:
        st.button("🚀 Initialize Agents & Pipeline", use_container_width=True,
                  disabled=True, key="init_btn_dis",
                  help="Select a dataset source above first.")

    st.divider()

    # ── Quick stats & Navigation ──
    # Navigation is always available so user can toggle between Landing, Dashboard, and Preview
    st.markdown("**🧭 Navigation**")
    current_p = st.session_state.get("current_page", "operation")
    if current_p not in ["landing", "operation", "preview"]:
        current_p = "operation"

    nav_idx = ["landing", "operation", "preview"].index(current_p)
    nav_sel = st.radio(
        "Go to page:", 
        ["🏠 Landing Page", "⚙️ Operation Dashboard", "📊 Data Preview & Editor"], 
        index=nav_idx,
        key="nav_radio_select"
    )

    new_page = "landing"
    if "Operation Dashboard" in nav_sel:
        new_page = "operation"
    elif "Data Preview" in nav_sel:
        new_page = "preview"

    if new_page != current_p:
        st.session_state.current_page = new_page
        st.query_params["page"] = new_page
        st.rerun()

    # Session stats and Reset are only shown if a dataset has been initialized
    if st.session_state.is_initialized:
        st.divider()
        s = coord.state
        st.markdown("**📊 Session Stats**")
        df_s = s.get("cleaned_df")
        sch = s.get("schema", {})
        if df_s is not None:
            st.markdown(f'''
            <div style="font-size:.72rem;color:#6b6860;line-height:1.7;">
            🗂️ <strong>{len(df_s):,}</strong> rows · <strong>{len(df_s.columns)}</strong> columns<br/>
            🧬 <strong>{len(sch.get("numerical_features",[]))}</strong> numeric · <strong>{len(sch.get("categorical_features",[]))}</strong> categorical<br/>
            🎯 Target: <code style="background:rgba(37,99,235,.08);border-radius:4px;padding:1px 5px;">{sch.get("target_column","—")}</code>
            </div>''', unsafe_allow_html=True)

        st.divider()
        st.markdown("**🧹 Reset Platform**")
        if st.button("🧹 Reset Workspace", use_container_width=True, key="reset_btn_sidebar"):
            st.session_state.coordinator = CoordinatorAgent()
            st.session_state.is_initialized = False
            st.session_state.current_page = "landing"
            st.session_state.chat_history = []
            if "report_paths_cached" in st.session_state:
                del st.session_state["report_paths_cached"]
            st.query_params.clear()
            st.rerun()

# ─── TOP NAV ───
sb = ('<span class="nstat"><span class="sdot" style="width:6px;height:6px;"></span> Initialized</span>'
      if st.session_state.is_initialized else
      '<span style="font-size:.68rem;color:#b45309;background:rgba(180,83,9,.08);border:1px solid rgba(180,83,9,.18);padding:3px 9px;border-radius:20px;font-weight:600;">⏳ Awaiting Init</span>')
st.markdown(f"""
<div class="tnav">
 <div class="nbrand"><div class="nhex"></div>IntelliRisk <span style="color:#2563eb;">AI</span><span style="font-size:.65rem;font-weight:500;color:#a09e98;margin-left:.4rem;border-left:1px solid #e0ddd5;padding-left:.55rem;">Multi-Agent Analytics & Prediction Platform</span></div>
 <div style="display:flex;align-items:center;gap:.75rem;">{sb}<span style="font-size:.65rem;color:#a09e98;font-family:monospace;">{datetime.now().strftime('%H:%M  %d %b %Y')}</span></div>
</div>""", unsafe_allow_html=True)

if not st.session_state.is_initialized:
    st.markdown('<div class="pad">', unsafe_allow_html=True)

    # ── Hero splash ──
    splash_l, splash_r = st.columns([3, 2], gap="large")

    with splash_l:
        if LOTTIE_AVAILABLE:
            ld = load_lottie_inline(LOTTIE_BARS)
            if ld:
                st_lottie(ld, speed=1, height=90, key="il")

        st.markdown('''
        <h2 style="font-size:1.65rem;font-weight:900;letter-spacing:-.04em;color:#1a1916;margin:.2rem 0 .5rem;">
          Ready to Analyze</h2>
        <p style="color:#6b6860;font-size:.88rem;line-height:1.65;max-width:480px;margin-bottom:1.2rem;">
          Select a dataset in the sidebar using any of the upload options,
          then click <strong>Initialize Agents &amp; Pipeline</strong> to launch
          the full multi-agent workflow.
        </p>''', unsafe_allow_html=True)

        # Pills for capabilities
        caps = ["🧬 Schema","🔍 Quality","📊 EDA","🤖 Prediction","💡 Explain","💬 Chat","📥 Export"]
        if PILLS_AVAILABLE:
            pills("Platform Modules", caps, index=None, key="splash_p")
        else:
            st.markdown(" &nbsp;·&nbsp; ".join(caps), unsafe_allow_html=True)

        # ── Pipeline Steps ──
        st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="ct" style="margin-bottom:.6rem;">⚙️ How It Works</div>', unsafe_allow_html=True)
        steps = [
            ("1", "Load & Understand",
             "Schema Agent scans every column — infers type, business meaning, PII flag, and target variable automatically."),
            ("2", "Clean & Profile",
             "Data Quality Agent detects missing values, duplicates, outliers and runs grouped-median smart imputation."),
            ("3", "Explore & Visualize",
             "EDA Agent builds correlation matrices, trend charts, distributions, and writes a portfolio narrative."),
            ("4", "Predict & Score",
             "Prediction Manager trains your chosen model (RF / XGB / LGB / LR / DT) and scores every account."),
            ("5", "Explain & Recommend",
             "Explainability + Recommendation Agents produce per-account reason codes and business mitigations."),
            ("6", "Export Reports",
             "Report Agent compiles Word, PDF, and CSV audit documents ready for download."),
        ]
        for num, title, desc in steps:
            st.markdown(f'''
            <div class="step-row">
              <div class="step-num">{num}</div>
              <div class="step-body">
                <div class="step-title">{title}</div>
                <div class="step-desc">{desc}</div>
              </div>
            </div>''', unsafe_allow_html=True)

    with splash_r:
        # ── Supported formats ──
        st.markdown('<div class="ct" style="margin-bottom:.7rem;">📁 Supported Data Formats</div>', unsafe_allow_html=True)
        formats = [
            ("📊", "Excel (.xlsx / .xls)",
             "Standard spreadsheet format. Multi-sheet files use the first sheet."),
            ("📄", "CSV (.csv)",
             "Comma or semicolon delimited. UTF-8 and Latin-1 encodings supported."),
        ]
        for icon, fmt_title, fmt_desc in formats:
            st.markdown(f'''
            <div class="fmt-card" style="margin-bottom:.55rem;">
              <div class="fmt-icon">{icon}</div>
              <div><div class="fmt-title">{fmt_title}</div>
                   <div class="fmt-desc">{fmt_desc}</div></div>
            </div>''', unsafe_allow_html=True)

        st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)

        # ── What the agents detect ──
        st.markdown('<div class="ct" style="margin-bottom:.65rem;">🧬 Auto-Detected Column Types</div>', unsafe_allow_html=True)
        col_types = [
            ("bb", "NUMERICAL", "Age, Income, Credit Score, Loan Balance …"),
            ("ba", "CATEGORICAL", "Employment Status, Loan Type, Region …"),
            ("br", "TARGET", "Default flag, Risk label, Delinquency indicator …"),
            ("ba", "TEMPORAL", "Payment_Month_1 … Payment_Month_12 sequences"),
            ("bb", "IDENTIFIER", "Customer_ID, Account_ID, unique key column"),
            ("br", "PII", "Name, SSN, Email — flagged and protected"),
        ]
        for cls, lbl, example in col_types:
            st.markdown(f'''
            <div style="display:flex;align-items:flex-start;gap:.55rem;margin-bottom:.5rem;">
              <span class="badge {cls}" style="flex-shrink:0;margin-top:2px;">{lbl}</span>
              <span style="font-size:.72rem;color:#6b6860;line-height:1.45;">{example}</span>
            </div>''', unsafe_allow_html=True)

        st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)

        # ── Model options ──
        st.markdown('<div class="ct" style="margin-bottom:.65rem;">🤖 Available Prediction Models</div>', unsafe_allow_html=True)
        models = [
            ("Random Forest", "Ensemble · handles non-linearity well"),
            ("XGBoost", "Gradient boosting · high accuracy"),
            ("LightGBM", "Fast gradient boosting · large datasets"),
            ("Logistic Regression", "Interpretable · linearly separable data"),
            ("Decision Tree", "Explainable · visual rule extraction"),
        ]
        for mname, mdesc in models:
            st.markdown(f'''
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:.38rem .6rem;border-bottom:1px solid #f0ede8;">
              <span style="font-size:.76rem;font-weight:600;color:#1a1916;">{mname}</span>
              <span style="font-size:.66rem;color:#a09e98;">{mdesc}</span>
            </div>''', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

if st.session_state.current_page == "preview":
    show_preview_page()
    st.stop()

# ─── MAIN ───
state = coord.state
schema = state.get("schema",{})
df = state.get("cleaned_df")
dq = state.get("data_quality_report",{})
eda = state.get("eda_report",{})
preds = state.get("predictions_df")
kpis_ = state.get("kpis",{})
mm = state.get("model_metrics")

st.markdown('<div class="pad">', unsafe_allow_html=True)
t1,t2,t3,t4,t5,t6,t7 = st.tabs(["📊 Overview","🔍 Schema & Quality","📈 Auto-EDA","🤖 Predictions & Models","💡 Risk Explainers","💬 Chat Assistant","📥 Export Reports"])

# OVERVIEW
with t1:
    k1,k2,k3,k4 = st.columns(4)
    with k1: kpi("Total Portfolio Accounts", kpis_.get("total_accounts","—"), "accounts scored")
    with k2: kpi("Avg Credit Score", kpis_.get("average_credit_score","—"), "portfolio average")
    with k3: kpi("Avg Risk Probability", kpis_.get("average_predicted_risk","Run Predictions"), "predicted delinquency rate")
    with k4: kpi("High Risk Accounts", kpis_.get("predicted_high_risk_accounts","Run Predictions"), "requires intervention")
    st.markdown("<div style='height:1.1rem'></div>", unsafe_allow_html=True)
    cl, cr = st.columns(2)
    with cl:
        card("Executive Summary","Narrative portfolio insights from the EDA Agent")
        st.markdown(state.get("narrative_insights","No insights — initialize pipeline first."))
        end()
    with cr:
        card("Risk Distribution","Breakdown by predicted risk level")
        if preds is not None:
            cnt = preds["risk_category"].value_counts().reset_index()
            cnt.columns = ["Risk Level","Count"]
            cmap = {"Low Risk":"#15803d","Medium Risk":"#b45309","Medium-High Risk (Potential High)":"#ea580c","High Risk":"#b91c1c"}
            fig = px.pie(cnt, values="Count", names="Risk Level", color="Risk Level", color_discrete_map=cmap, hole=0.45)
            fig.update_layout(**PLOT, showlegend=True, legend=dict(orientation="h",yanchor="bottom",y=-.15,xanchor="center",x=.5))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        else:
            st.info("⚠️ Train a prediction model to see risk distribution.")
        end()
    divider("AI Action Registry")
    card("Portfolio Action Registry","Macro-level credit strategies from the Recommendation Agent")
    bulk = state.get("bulk_recs",{})
    if bulk and "portfolio_actions" in bulk:
        for r in bulk["portfolio_actions"]: st.markdown(f"— &nbsp; {r}")
    else:
        st.info("⚠️ Train predictive models to populate the action registry.")
    end()

# SCHEMA & QUALITY
with t2:
    cs, cq = st.columns(2)
    with cs:
        card("Inferred Semantic Schema","Auto-detected by the Schema Understanding Agent")
        sc = schema.get("columns",{})
        scm = {"identifier":"bb","numerical":"bb","categorical":"ba","target":"br","temporal":"ba","date":"bb"}
        rows = ""
        for cn, det in sc.items():
            stype = det.get("semantic_type","unknown")
            pii = '<span class="badge br">PII</span>' if det.get("is_pii") else '<span class="badge bg">Safe</span>'
            sb2 = f'<span class="badge {scm.get(stype,"bb")}">{stype}</span>'
            rows += f"<tr><td><strong>{cn}</strong></td><td>{sb2}</td><td>{pii}</td><td style='color:#6b6860;font-size:.72rem;'>{det.get('business_meaning','')}</td></tr>"
        st_md = f'<table class="dt"><thead><tr><th>Column</th><th>Type</th><th>PII</th><th>Business Meaning</th></tr></thead><tbody>{rows}</tbody></table>'
        st.markdown(st_md, unsafe_allow_html=True)
        end()
    with cq:
        card("Data Quality Profile","Issues identified by the Data Quality Agent")
        st.markdown(f"**Duplicate rows:** `{dq.get('duplicate_rows_count',0)}`")
        miss = dq.get("missing_values",{})
        if miss:
            st.markdown("##### Missing Values")
            mr = "".join(f"<tr><td><strong>{c}</strong></td><td>{d['count']}</td><td><span class='badge ba'>{d['percentage']:.1f}%</span></td></tr>" for c,d in miss.items())
            st.markdown(f'<table class="dt"><thead><tr><th>Column</th><th>Count</th><th>Rate</th></tr></thead><tbody>{mr}</tbody></table>', unsafe_allow_html=True)
        else:
            st.success("✓ No missing values.")
        out = dq.get("outliers",{})
        if out:
            st.markdown("##### Outliers (IQR)")
            or_ = "".join(f"<tr><td><strong>{c}</strong></td><td>{d['count']}</td><td><span class='badge ba'>{d['percentage']:.1f}%</span></td></tr>" for c,d in out.items())
            st.markdown(f'<table class="dt"><thead><tr><th>Column</th><th>Count</th><th>Ratio</th></tr></thead><tbody>{or_}</tbody></table>', unsafe_allow_html=True)
        end()
        card("Cleaning Pipeline","Auto-executed transformations")
        for rec in dq.get("cleaning_recommendations",[]): st.markdown(f"✔ {rec}")
        end()

# EDA
with t3:
    if LOTTIE_AVAILABLE:
        ld = load_lottie_inline(LOTTIE_BARS)
        if ld:
            _,lc,_ = st.columns([5,1,5])
            with lc: st_lottie(ld, speed=1, height=70, key="el")
    el, er = st.columns(2)
    with el:
        card("Correlation Heatmap","Linear relationships between numerical features")
        vn = [c for c in schema.get("numerical_features",[]) if c in df.columns]
        if len(vn)>=2:
            cdf = df[vn].corr()
            fig = px.imshow(cdf, text_auto=".2f", color_continuous_scale="Blues", zmin=-1, zmax=1)
            fig.update_layout(**PLOT)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        else:
            st.info("Not enough numerical columns.")
        end()
    with er:
        card("Historical Delinquency Trend","Portfolio payment delinquency across periods")
        tc = schema.get("temporal_features",[])
        if tc:
            li = ["late","missed","delinquent"]
            rates = [(df[c].astype(str).str.strip().str.lower().isin(li).sum()/len(df))*100 for c in tc if c in df.columns]
            if rates:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=tc, y=rates, mode="lines+markers",
                    line=dict(color="#2563eb",width=3), marker=dict(size=8,color="#2563eb",line=dict(color="white",width=2)),
                    fill="tozeroy", fillcolor="rgba(37,99,235,.06)", name="Rate"))
                fig.update_layout(**PLOT, xaxis_title="Period", yaxis_title="Rate (%)")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
            else: st.info("No temporal data.")
        else: st.info("No temporal columns.")
        end()
    divider("Feature Distributions")
    nc = [c for c in schema.get("numerical_features",[]) if c in df.columns]
    if nc:
        pal = ["#2563eb","#15803d","#b45309"]
        cd = st.columns(min(3,len(nc)))
        for i,col in enumerate(nc[:3]):
            with cd[i]:
                card(f"Distribution — {col}")
                fig = px.histogram(df, x=col, color_discrete_sequence=[pal[i%3]])
                fig.update_layout(**{**PLOT,"margin":dict(l=10,r=10,t=10,b=10)})
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
                end()

# PREDICTIONS
with t4:
    divider("Model Configuration")
    tc1, tc2 = st.columns([2,3])
    with tc1:
        card("Algorithm Selection","Prediction Manager handles all feature transformations")
        mopts = ["Random Forest","Logistic Regression","Decision Tree","XGBoost","LightGBM"]
        if PILLS_AVAILABLE:
            mc = pills("Select Algorithm", mopts, key="mp")
            if mc is None: mc = "Random Forest"
        else:
            mc = st.selectbox("Algorithm", mopts)
        st.markdown("<div style='height:.45rem'></div>", unsafe_allow_html=True)
        if st.button("⚡ Train Predictor Model", use_container_width=True):
            with st.spinner(f"Training {mc}…"):
                try:
                    coord.train_predictive_model(mc)
                    st.success(f"✓ {mc} trained!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        end()
        if mm:
            card("Test Performance Metrics","Out-of-sample evaluation")
            mdata = {"Accuracy":f"{mm.get('accuracy',0):.4f}","F1-Score":f"{mm.get('f1_score',0):.4f}","Precision":f"{mm.get('precision',0):.4f}","Recall":f"{mm.get('recall',0):.4f}","ROC-AUC":f"{mm.get('roc_auc',0):.4f}"}
            rows2 = "".join(f"<tr><td style='color:#6b6860;font-weight:600;'>{k}</td><td><code style='background:rgba(37,99,235,.07);border-radius:4px;padding:1px 6px;'>{v}</code></td></tr>" for k,v in mdata.items())
            st.markdown(f'<table class="dt"><tbody>{rows2}</tbody></table>', unsafe_allow_html=True)
            end()
    with tc2:
        card("Feature Importance","Top predictive contributors")
        if mm and "feature_importances" in mm:
            imp = mm["feature_importances"]
            si = sorted(imp.items(), key=lambda x:x[1], reverse=True)[:10]
            fs,ss = zip(*si)
            fig = px.bar(x=list(ss), y=list(fs), orientation="h", color=list(ss), color_continuous_scale=[[0,"#e0e9ff"],[1,"#2563eb"]])
            fig.update_layout(**PLOT, xaxis_title="Importance", coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        else:
            st.info("Train a model to see feature importance.")
        end()
    divider("Scored Account Registry")
    if preds is not None:
        idc = schema.get("identifier_column","Customer_ID")
        f1,f2,f3 = st.columns(3)
        with f1: sid = st.text_input("🔎 Search ID", placeholder="e.g. CUST0001")
        with f2:
            if PILLS_AVAILABLE:
                rfs = pills("Risk Filter",["All","Low Risk","Medium Risk","Medium-High Risk (Potential High)","High Risk"],key="rfp")
                rf = ["Low Risk","Medium Risk","Medium-High Risk (Potential High)","High Risk"] if not rfs or rfs=="All" else [rfs]
            else:
                rf = st.multiselect("Risk Level",["Low Risk","Medium Risk","Medium-High Risk (Potential High)","High Risk"],default=["Low Risk","Medium Risk","Medium-High Risk (Potential High)","High Risk"])
        with f3: ms = st.slider("Min Risk Prob",0.0,1.0,0.0,step=.05)
        fp = preds.copy()
        if sid: fp = fp[fp[idc].astype(str).str.contains(sid.strip(), case=False)]
        fp = fp[fp["risk_category"].isin(rf)]
        fp = fp[fp["risk_probability"]>=ms]
        dcols = [c for c in [idc,"Age","Income","Credit_Score","risk_probability","risk_category","confidence_score"] if c in fp.columns]
        if "personal_explanation" in fp.columns: dcols.append("personal_explanation")
        st.dataframe(fp[dcols], use_container_width=True, column_config={"risk_probability": st.column_config.ProgressColumn("Risk Prob",format="%.2f",min_value=0,max_value=1)})
    else:
        st.info("⚠️ Train a model first.")

# EXPLAINERS
with t5:
    divider("Individual Account Intelligence")
    if preds is not None:
        idc = schema.get("identifier_column","Customer_ID")
        ids = preds[idc].unique().tolist()
        sel = st.selectbox("Select Account", ids, key="exsel")
        if sel:
            with st.spinner(f"Running agents for {sel}…"):
                rx = coord.explain_individual(sel)
            row = preds[preds[idc]==sel].iloc[0]
            cat = rx.get("risk_category","Unknown")
            bc = "br" if cat == "High Risk" else ("ba" if "Medium" in cat else "bg")
            cx1, cx2 = st.columns(2)
            with cx1:
                card(f"Account Profile — {sel}","Demographics & credit attributes")
                skip = {"personal_explanation","personal_recommendation","predicted_label","risk_probability","risk_category","confidence_score",idc}
                ph = "".join(f"<tr><td style='color:#6b6860;font-weight:600;'>{k}</td><td>{v}</td></tr>" for k,v in row.to_dict().items() if k not in skip)
                st.markdown(f'<table class="dt"><tbody>{ph}</tbody></table>', unsafe_allow_html=True)
                end()
            with cx2:
                card("AI Risk Scoring","Model output parameters")
                st.markdown(f'<div style="font-size:1.05rem;font-weight:700;margin-bottom:.75rem;">Status: <span class="badge {bc}" style="font-size:.88rem;padding:3px 11px;">{cat}</span></div>', unsafe_allow_html=True)
                st.markdown(f"Delinquency Probability: `{rx.get('risk_probability','—')}`")
                st.markdown(f"Confidence Score: `{row.get('confidence_score',0.0):.4f}`")
                end()
                st.markdown('<div class="cc accent-c"><div class="ct" style="color:#2563eb;">⚡ Explainability Agent Insights</div><div class="cs">Why this risk level was assigned</div>', unsafe_allow_html=True)
                st.markdown(rx.get("explanation",""))
                end()
                st.markdown('<div class="cc green-c"><div class="ct" style="color:#15803d;">✅ Recommendation Agent Actions</div><div class="cs">Tailored mitigations for this risk profile</div>', unsafe_allow_html=True)
                st.markdown(rx.get("recommendation",""))
                end()
    else:
        st.info("⚠️ Generate predictions first.")

# CHAT
with t6:
    divider("Conversational Portfolio Assistant")
    card("Ask anything about the portfolio","Natural language queries powered by the Chat Agent")
    st.markdown("Try: *Show highest risk customers* · *Why is CUST0001 high risk?* · *Overall delinquency rate?*")
    end()
    for msg in st.session_state.chat_history:
        if msg["role"]=="user":
            st.markdown(f'<div class="cu">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="ca">{msg["content"]}</div>', unsafe_allow_html=True)
    cq = st.chat_input("Ask about portfolio risk…")
    if cq:
        st.session_state.chat_history.append({"role":"user","content":cq})
        with st.spinner("Thinking…"):
            ans = coord.chat_query(cq)
        st.session_state.chat_history.append({"role":"assistant","content":ans})
        st.rerun()

# EXPORT
with t7:
    divider("Document Generation & Export")
    g1, g2 = st.columns(2)
    with g1:
        card("Generate Audit Documents","Report Agent compiles visuals, KPIs, and narrative")
        st.markdown("Generates **Word (.docx)** and **PDF** executive reports with risk distributions, model metrics, and action registries.")
        if st.button("📄 Generate PDF & Word Reports", use_container_width=True):
            with st.spinner("Assembling reports…"):
                try:
                    paths = coord.generate_reports()
                    st.success("✓ Reports saved to outputs/")
                    st.session_state.report_paths_cached = paths
                except Exception as e:
                    st.error(f"Error: {e}")
        end()
    with g2:
        card("Download Files","Export to your machine")
        cp = st.session_state.get("report_paths_cached", state.get("report_paths"))
        if cp:
            dp_ = cp.get("docx")
            if dp_ and os.path.exists(dp_):
                with open(dp_,"rb") as f:
                    st.download_button("📥 Word Report (.docx)", data=f.read(), file_name="intellirisk_report.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True)
            pp_ = cp.get("pdf")
            if pp_ and os.path.exists(pp_):
                with open(pp_,"rb") as f:
                    st.download_button("📥 PDF Report", data=f.read(), file_name="intellirisk_report.pdf", mime="application/pdf", use_container_width=True)
            csvp = cp.get("csv")
            if csvp and os.path.exists(csvp):
                with open(csvp,"rb") as f:
                    st.download_button("📥 Scored Accounts (.csv)", data=f.read(), file_name="intellirisk_predictions.csv", mime="text/csv", use_container_width=True)
        else:
            st.info("Generate reports above to enable downloads.")
        end()

# Clear dataset tab removed. Global workspace reset is now handled via the sidebar "🧹 Reset Workspace" button.

st.markdown("</div>", unsafe_allow_html=True)
st.markdown('<div style="text-align:center;padding:1.8rem 0 .8rem;border-top:1px solid #e0ddd5;margin-top:1.5rem;color:#a09e98;font-size:.7rem;"><strong style="color:#6b6860;">IntelliRisk AI</strong> &nbsp;·&nbsp; Multi-Agent Analytics & Prediction Platform &nbsp;·&nbsp; v2.0</div>', unsafe_allow_html=True)