import random
import time
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils import Point, AlgorithmResult, brute_force, divide_and_conquer, build_svg

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Closest Point Analyzer",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=JetBrains+Mono:wght@400;600&family=Sora:wght@300;400;600;700&display=swap');

:root {
  --bg:#06061a; --panel:rgba(16,12,48,0.75); --border:rgba(100,80,200,0.25);
  --accent1:#f72585; --accent2:#00f5d4; --accent3:#7209b7;
  --text:#d0cef5; --muted:#7070a0;
}
            

.stApp {
  background: radial-gradient(ellipse at 20% 20%, #1a0533 0%, #080820 40%, #020210 100%) !important;
  font-family: 'Sora', sans-serif; color: var(--text);
}

.stApp::before {
  content: '0 1 1 0 1 0 0 1 1 0 1 0 0 1 0 1 1 0 1 0 1 0 1 1 0 1 0 0 1 0 1 1 0 1 0 0 1';
  position: fixed; top:0; left:0; width:200%; height:200%;
  font-family:'JetBrains Mono',monospace; font-size:20px;
  color:rgba(80,50,180,0.035); word-spacing:24px; line-height:50px;
  pointer-events:none; z-index:0;
  animation:floatBg 30s linear infinite; white-space:pre-wrap; letter-spacing:16px;
}
@keyframes floatBg {
  0%{transform:translate(0,0)} 50%{transform:translate(-4%,-7%)} 100%{transform:translate(0,0)}
}

[data-testid="stSidebar"] {
  background:linear-gradient(180deg,rgba(15,8,42,0.97) 0%,rgba(8,5,28,0.97) 100%) !important;
  border-right:1px solid var(--border) !important;
  backdrop-filter:blur(12px);
}
[data-testid="stSidebar"] * { color:var(--text) !important; }

h1,h2,h3 { font-family:'Rajdhani',sans-serif !important; letter-spacing:1px; }
h1 {
  background:linear-gradient(90deg,var(--accent1),var(--accent2));
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  font-size:2rem !important; font-weight:700 !important;
}

.glass-card {
  background:var(--panel); border:1px solid var(--border); border-radius:18px;
  padding:22px 24px; backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px);
  margin-bottom:18px; box-shadow:0 8px 32px rgba(0,0,0,0.4);
  position:relative; overflow:hidden;
}
.glass-card::before {
  content:''; position:absolute; top:0; left:0; right:0; height:1px;
  background:linear-gradient(90deg,transparent,rgba(150,100,255,0.5),transparent);
}

.metric-box {
  background:rgba(8,6,28,0.8); border:1px solid rgba(100,80,200,0.3);
  border-radius:12px; padding:12px 16px; margin:6px 0;
  font-family:'JetBrains Mono',monospace; font-size:13px;
}
.metric-label { color:var(--muted); font-size:11px; margin-bottom:4px; }
.metric-value { color:var(--accent2); font-weight:600; font-size:15px; }
.metric-value.pink { color:var(--accent1); }
.metric-value.purple { color:#b388ff; }

.stButton>button {
  background:linear-gradient(135deg,rgba(114,9,183,0.4),rgba(247,37,133,0.3)) !important;
  color:#e0d8ff !important; border:1px solid rgba(180,100,255,0.4) !important;
  border-radius:12px !important; font-family:'Rajdhani',sans-serif !important;
  font-weight:600 !important; font-size:14px !important; letter-spacing:0.6px !important;
  padding:8px 18px !important; transition:all 0.25s ease !important; width:100% !important;
}
.stButton>button:hover {
  background:linear-gradient(135deg,rgba(247,37,133,0.5),rgba(0,245,212,0.3)) !important;
  border-color:rgba(0,245,212,0.6) !important;
  box-shadow:0 0 20px rgba(0,245,212,0.3),0 0 40px rgba(247,37,133,0.15) !important;
  transform:translateY(-1px) !important; color:#ffffff !important;
}
.stButton>button:active { transform:translateY(0) !important; }

.stNumberInput>div>div>input,
.stTextInput>div>div>input {
  background:rgba(10,8,30,0.8) !important;
  border:1px solid rgba(100,80,200,0.35) !important;
  border-radius:10px !important; color:var(--text) !important;
  font-family:'JetBrains Mono',monospace !important;
}

.stTabs [data-baseweb="tab-list"] {
  background:rgba(10,8,30,0.6) !important;
  border-radius:12px !important; padding:4px !important;
  border:1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
  font-family:'Rajdhani',sans-serif !important; font-weight:600 !important;
  color:var(--muted) !important; border-radius:8px !important;
}
.stTabs [aria-selected="true"] {
  background:linear-gradient(135deg,rgba(114,9,183,0.5),rgba(247,37,133,0.3)) !important;
  color:white !important;
}

hr {
  border:none !important; height:1px !important;
  background:linear-gradient(90deg,transparent,var(--border),transparent) !important;
  margin:16px 0 !important;
}

.svg-wrap {
  border-radius:18px; overflow:hidden;
  box-shadow:0 0 40px rgba(114,9,183,0.2),0 0 80px rgba(247,37,133,0.05);
  background:#02020f;
}

.point-chip {
  display:inline-block; background:rgba(114,9,183,0.2);
  border:1px solid rgba(100,80,200,0.3); border-radius:8px;
  padding:3px 10px; font-family:'JetBrains Mono',monospace;
  font-size:11px; color:#b0a0e8; margin:3px 3px 3px 0;
}

#MainMenu,footer,header { visibility:hidden; }
.block-container { padding-top:1.5rem !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SESSION STATE INIT
# ─────────────────────────────────────────────
def _init_state():
    defaults = dict(
        points=[], result=None, algo="Brute Force",
        step=-1, show_splits=False, next_id=0,
        bf_time=None, dc_time=None, bf_comp=None, dc_comp=None,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def _add_point(x, y):
    st.session_state.points.append(
        Point(x=round(x,3), y=round(y,3), id=st.session_state.next_id)
    )
    st.session_state.next_id += 1
    st.session_state.result = None
    st.session_state.step = -1


def _run_algo():
    pts = st.session_state.points
    if len(pts) < 2:
        st.warning("Add at least 2 points first.")
        return
    try:
        res = brute_force(pts) if st.session_state.algo == "Brute Force" else divide_and_conquer(pts)
        st.session_state.result = res
        st.session_state.step = -1
    except Exception as e:
        st.error(f"Error: {e}")


def _run_both():
    pts = st.session_state.points
    if len(pts) < 2:
        st.warning("Add at least 2 points first.")
        return
    try:
        bf = brute_force(pts)
        dc = divide_and_conquer(pts)
        st.session_state.bf_time = bf.time_ms
        st.session_state.dc_time = dc.time_ms
        st.session_state.bf_comp = bf.comparisons
        st.session_state.dc_comp = dc.comparisons
        st.session_state.result = bf if st.session_state.algo == "Brute Force" else dc
    except Exception as e:
        st.error(f"Error: {e}")


def _perf_chart():
    sizes = list(range(2, 181, 10))
    bf_t, dc_t = [], []
    for n in sizes:
        pts = [Point(x=random.uniform(0,100), y=random.uniform(0,100), id=i) for i in range(n)]
        t0 = time.perf_counter(); brute_force(pts); bf_t.append((time.perf_counter()-t0)*1000)
        t0 = time.perf_counter(); divide_and_conquer(pts); dc_t.append((time.perf_counter()-t0)*1000)

    fig, ax = plt.subplots(figsize=(8,3.8))
    fig.patch.set_facecolor("#06061a"); ax.set_facecolor("#08081e")
    ax.plot(sizes, bf_t, color="#f72585", lw=2.2, label="Brute Force O(n²)", marker="o", ms=3)
    ax.plot(sizes, dc_t, color="#00f5d4", lw=2.2, label="D&C O(n log n)", marker="s", ms=3)
    ax.set_xlabel("Points (n)", color="#7070a0", fontsize=10)
    ax.set_ylabel("Time (ms)", color="#7070a0", fontsize=10)
    ax.set_title("Runtime Benchmark", color="#c5a0f0", fontsize=13, fontweight="bold", pad=10)
    ax.tick_params(colors="#555580")
    for sp in ax.spines.values(): sp.set_edgecolor("#2a2a5a")
    ax.grid(color="#1a1a3a", lw=0.5, alpha=0.6)
    ax.legend(facecolor="#0e0b2e", edgecolor="#3a2a6a", labelcolor="#c5a0f0", fontsize=9)
    buf = io.BytesIO(); fig.tight_layout()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight"); plt.close(fig); buf.seek(0)
    return buf


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:8px 0 16px;'>
      <div style='font-family:Rajdhani,sans-serif;font-size:22px;font-weight:700;
                  background:linear-gradient(90deg,#f72585,#00f5d4);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
        ⬡ CLOSEST PAIR
      </div>
      <div style='font-family:JetBrains Mono,monospace;font-size:10px;color:#444488;
                  letter-spacing:2px;margin-top:4px;'>ANALYZER  v2.0</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Algorithm
    st.markdown('<p style="font-family:Rajdhani,sans-serif;font-size:13px;color:#9b8fd4;'
                'letter-spacing:1px;margin-bottom:4px;">▸ ALGORITHM</p>', unsafe_allow_html=True)
    algo = st.radio("Select Algorithm", ["Brute Force","Divide & Conquer"],
                    index=0 if st.session_state.algo=="Brute Force" else 1,
                    label_visibility="collapsed", horizontal=True)
    if algo != st.session_state.algo:
        st.session_state.algo = algo
        st.session_state.result = None
        st.session_state.step = -1

    cinfo = {"Brute Force":("O(n²)","Compare every pair"),"Divide & Conquer":("O(n log n)","Recursive split+strip")}
    cl, cd = cinfo[st.session_state.algo]
    st.markdown(f'<div class="metric-box"><span class="metric-label">Complexity</span>'
                f'<span class="metric-value purple">{cl}</span>'
                f'<br><span style="color:#444478;font-size:10px;">{cd}</span></div>',
                unsafe_allow_html=True)

    st.markdown("---")

    # Manual input
    st.markdown('<p style="font-family:Rajdhani,sans-serif;font-size:13px;color:#9b8fd4;'
                'letter-spacing:1px;margin-bottom:4px;">▸ ADD POINT</p>', unsafe_allow_html=True)
    cx, cy = st.columns(2)
    with cx: ix = st.number_input("X", value=0.0, step=1.0, format="%.2f", key="ix")
    with cy: iy = st.number_input("Y", value=0.0, step=1.0, format="%.2f", key="iy")
    if st.button("＋  Add Point"):
        _add_point(ix, iy); st.rerun()

    st.markdown("---")

    # Random
    st.markdown('<p style="font-family:Rajdhani,sans-serif;font-size:13px;color:#9b8fd4;'
                'letter-spacing:1px;margin-bottom:4px;">▸ RANDOM GENERATOR</p>', unsafe_allow_html=True)
    n_rand = st.slider("Points", 2, 50, 10)
    rng = st.slider("Range", 10, 500, 100)
    if st.button("⚡  Generate Random"):
        for i in range(n_rand):
            _add_point(round(random.uniform(0,rng),2), round(random.uniform(0,rng),2))
        st.rerun()

    st.markdown("---")

    # CSV
    st.markdown('<p style="font-family:Rajdhani,sans-serif;font-size:13px;color:#9b8fd4;'
                'letter-spacing:1px;margin-bottom:4px;">▸ CSV UPLOAD</p>', unsafe_allow_html=True)
    csv_f = st.file_uploader("CSV (x,y columns)", type=["csv"], label_visibility="collapsed")
    if csv_f:
        try:
            df = pd.read_csv(csv_f); df.columns = [c.strip().lower() for c in df.columns]
            if "x" not in df.columns or "y" not in df.columns:
                st.error("Need 'x' and 'y' columns.")
            else:
                df = df[["x","y"]].dropna()
                for _, row in df.iterrows():
                    _add_point(float(row.x), float(row.y))
                st.success(f"✓ Loaded {len(df)} points"); st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("---")

    # Actions
    st.markdown('<p style="font-family:Rajdhani,sans-serif;font-size:13px;color:#9b8fd4;'
                'letter-spacing:1px;margin-bottom:4px;">▸ ACTIONS</p>', unsafe_allow_html=True)
    if st.button("🔍  Find Closest Pair"):
        _run_algo(); st.rerun()
    if st.button("⚖️  Compare Both"):
        _run_both(); st.rerun()
    if st.button("🗑️  Clear All"):
        for k in ["points","result","bf_time","dc_time","bf_comp","dc_comp"]:
            st.session_state[k] = [] if k=="points" else None
        st.session_state.step = -1; st.rerun()

    n_pts = len(st.session_state.points)
    st.markdown(f"""<div style='text-align:center;margin-top:16px;'>
      <div style='font-family:JetBrains Mono,monospace;font-size:28px;font-weight:600;
                  background:linear-gradient(90deg,#f72585,#00f5d4);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>{n_pts}</div>
      <div style='color:#444488;font-size:10px;letter-spacing:1px;'>POINTS LOADED</div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MAIN AREA
# ─────────────────────────────────────────────
st.markdown(
    '<h1 style="margin-bottom:4px;">⬡ Closest Pair Analyzer</h1>'
    '<p style="color:#4a4a8a;font-family:JetBrains Mono,monospace;font-size:11px;'
    'letter-spacing:2px;margin-top:0;margin-bottom:18px;">'
    'BRUTE FORCE · DIVIDE &amp; CONQUER · INTERACTIVE VISUALIZATION</p>',
    unsafe_allow_html=True)

tab_vis, tab_step, tab_perf, tab_data = st.tabs(
    ["📡  Visualization","🎬  Step-by-Step","📊  Performance","📋  Data Table"])

# ── TAB 1: VISUALIZATION ──────────────────────────────────────────
with tab_vis:
    c1, c2 = st.columns([3, 1.1])
    with c1:
        svg = build_svg(st.session_state.points, result=st.session_state.result, active_step=-1)
        st.markdown(f'<div class="svg-wrap">{svg}</div>', unsafe_allow_html=True)
        if st.session_state.points:
            chips = "".join(
                f'<span class="point-chip">#{p.id} ({p.x},{p.y})</span>'
                for p in st.session_state.points[-20:])
            dots = '<span class="point-chip" style="color:#3a3a7a;">+more</span>' \
                   if len(st.session_state.points)>20 else ""
            st.markdown(f'<div style="margin-top:12px;">{chips}{dots}</div>', unsafe_allow_html=True)

    with c2:
        res = st.session_state.result
        if res:
            st.markdown(f"""<div class="glass-card">
  <div style="font-family:Rajdhani,sans-serif;font-size:18px;font-weight:700;
              color:#c5a0f0;margin-bottom:14px;letter-spacing:1px;">▸ RESULT</div>
  <div class="metric-box">
    <div class="metric-label">ALGORITHM</div>
    <div class="metric-value purple">{res.algorithm}</div>
  </div>
  <div class="metric-box">
    <div class="metric-label">POINT A</div>
    <div class="metric-value">({res.point_a.x:.4f}, {res.point_a.y:.4f})</div>
  </div>
  <div class="metric-box">
    <div class="metric-label">POINT B</div>
    <div class="metric-value">({res.point_b.x:.4f}, {res.point_b.y:.4f})</div>
  </div>
  <div class="metric-box">
    <div class="metric-label">MIN DISTANCE</div>
    <div class="metric-value pink">{res.distance:.6f}</div>
  </div>
  <div class="metric-box">
    <div class="metric-label">EXEC TIME</div>
    <div class="metric-value">{res.time_ms:.4f} ms</div>
  </div>
  <div class="metric-box">
    <div class="metric-label">COMPARISONS</div>
    <div class="metric-value purple">{res.comparisons:,}</div>
  </div>
  <div class="metric-box">
    <div class="metric-label">COMPLEXITY</div>
    <div class="metric-value">{"O(n²)" if res.algorithm=="Brute Force" else "O(n log n)"}</div>
  </div>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="glass-card" style="text-align:center;padding:40px 16px;">
  <div style="font-size:32px;margin-bottom:12px;">🔮</div>
  <div style="font-family:Rajdhani,sans-serif;font-size:16px;color:#3a3a7a;">No result yet</div>
  <div style="font-family:JetBrains Mono,monospace;font-size:10px;color:#252545;margin-top:8px;">
    Add points &amp; click<br>Find Closest Pair
  </div>
</div>""", unsafe_allow_html=True)

        if st.session_state.bf_time is not None:
            st.markdown(f"""<div class="glass-card">
  <div style="font-family:Rajdhani,sans-serif;font-size:16px;font-weight:700;
              color:#c5a0f0;margin-bottom:12px;">⚖ COMPARISON</div>
  <div class="metric-box">
    <div class="metric-label">BF TIME</div>
    <div class="metric-value pink">{st.session_state.bf_time:.4f} ms</div>
  </div>
  <div class="metric-box">
    <div class="metric-label">D&amp;C TIME</div>
    <div class="metric-value">{st.session_state.dc_time:.4f} ms</div>
  </div>
  <div class="metric-box">
    <div class="metric-label">BF COMPARISONS</div>
    <div class="metric-value purple">{st.session_state.bf_comp:,}</div>
  </div>
  <div class="metric-box">
    <div class="metric-label">D&amp;C COMPARISONS</div>
    <div class="metric-value purple">{st.session_state.dc_comp:,}</div>
  </div>
</div>""", unsafe_allow_html=True)


# ── TAB 2: STEP-BY-STEP ──────────────────────────────────────────
with tab_step:
    if not st.session_state.result:
        st.markdown("""<div class="glass-card" style="text-align:center;padding:48px;">
  <div style="font-size:38px;">🎬</div>
  <div style="font-family:Rajdhani,sans-serif;font-size:17px;color:#3a3a7a;margin-top:12px;">
    Run an algorithm first to enable step-by-step playback</div></div>""",
            unsafe_allow_html=True)
    else:
        res = st.session_state.result
        n_steps = len(res.steps)

        cc1, cc2 = st.columns([2,1])
        with cc1:
            sv = st.slider("Step", 0, max(n_steps-1,0),
                           value=max(0,min(st.session_state.step, n_steps-1)), key="ss")
            st.session_state.step = sv
        with cc2:
            sp = st.toggle("Show D&C splits", value=st.session_state.show_splits)
            st.session_state.show_splits = sp

        b1,b2,b3,b4 = st.columns(4)
        with b1:
            if st.button("⏮ First"): st.session_state.step=0; st.rerun()
        with b2:
            if st.button("◀ Prev"): st.session_state.step=max(0,st.session_state.step-1); st.rerun()
        with b3:
            if st.button("Next ▶"): st.session_state.step=min(n_steps-1,st.session_state.step+1); st.rerun()
        with b4:
            if st.button("Last ⏭"): st.session_state.step=n_steps-1; st.rerun()

        svg2 = build_svg(st.session_state.points, result=res,
                         active_step=st.session_state.step,
                         show_splits=st.session_state.show_splits)
        st.markdown(f'<div class="svg-wrap">{svg2}</div>', unsafe_allow_html=True)

        if 0 <= st.session_state.step < n_steps:
            step = res.steps[st.session_state.step]
            st_type = step.get("type","")
            icons = {"compare":"🔀","base_compare":"🔀","strip_compare":"📏","split":"⚡"}
            icon = icons.get(st_type, "•")
            if st_type in ("compare","base_compare","strip_compare") and "a" in step:
                desc = f"Comparing {step['a']} ↔ {step['b']}  |  d = {step.get('dist',0):.4f}  {'✨ NEW BEST!' if step.get('is_best') else ''}"
            elif st_type == "split":
                desc = f"Split at x = {step.get('mid_x',0):.3f}  (depth {step.get('depth',0)})"
            else:
                desc = str(step)
            st.markdown(f"""<div class="glass-card" style="margin-top:14px;padding:14px 20px;">
  <span style="font-size:18px;">{icon}</span>
  <span style="font-family:JetBrains Mono,monospace;font-size:12px;color:#b0a0e8;margin-left:10px;">
    {desc}</span>
</div>""", unsafe_allow_html=True)


# ── TAB 3: PERFORMANCE ───────────────────────────────────────────
with tab_perf:
    st.markdown("""<div class="glass-card">
  <div style="font-family:Rajdhani,sans-serif;font-size:18px;font-weight:700;
              color:#c5a0f0;margin-bottom:10px;">📊 Runtime Benchmark</div>
  <p style="color:#555588;font-size:12px;font-family:JetBrains Mono,monospace;">
    Runs both algorithms on random sets from n=2 to n=180</p></div>""",
        unsafe_allow_html=True)

    if st.button("🚀  Run Benchmark"):
        with st.spinner("Benchmarking all sizes..."):
            buf = _perf_chart()
        st.image(buf, use_container_width=True)
    else:
        st.markdown("""<div class="glass-card" style="text-align:center;padding:56px;">
  <div style="font-size:38px;">📈</div>
  <div style="font-family:Rajdhani,sans-serif;font-size:16px;color:#3a3a7a;margin-top:12px;">
    Click to run the benchmark</div></div>""", unsafe_allow_html=True)

    st.markdown("""<div class="glass-card" style="margin-top:18px;">
  <div style="font-family:Rajdhani,sans-serif;font-size:17px;font-weight:700;
              color:#c5a0f0;margin-bottom:14px;">📐 Complexity Reference</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;">
    <div class="metric-box">
      <div class="metric-label">BRUTE FORCE</div>
      <div class="metric-value pink">O(n²) time · O(1) space</div>
      <div style="color:#444478;font-size:10px;margin-top:4px;">Checks every pair</div>
    </div>
    <div class="metric-box">
      <div class="metric-label">DIVIDE &amp; CONQUER</div>
      <div class="metric-value">O(n log n) time · O(n) space</div>
      <div style="color:#444478;font-size:10px;margin-top:4px;">Sort + recurse + strip merge</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)


# ── TAB 4: DATA TABLE ────────────────────────────────────────────
with tab_data:
    pts = st.session_state.points
    if not pts:
        st.markdown("""<div class="glass-card" style="text-align:center;padding:48px;">
  <div style="font-size:34px;">📋</div>
  <div style="font-family:Rajdhani,sans-serif;font-size:16px;color:#3a3a7a;margin-top:12px;">
    No points loaded yet</div></div>""", unsafe_allow_html=True)
    else:
        res = st.session_state.result
        cl_ids = {res.point_a.id, res.point_b.id} if res else set()
        records = [{"ID":p.id,"X":p.x,"Y":p.y,"Closest Pair?":"✨ YES" if p.id in cl_ids else ""} for p in pts]
        df = pd.DataFrame(records)
        st.dataframe(df, use_container_width=True, hide_index=True,
            column_config={
                "ID": st.column_config.NumberColumn(width="small"),
                "X":  st.column_config.NumberColumn(format="%.3f"),
                "Y":  st.column_config.NumberColumn(format="%.3f"),
                "Closest Pair?": st.column_config.TextColumn(width="medium"),
            })
        st.download_button("⬇  Download CSV", data=df.to_csv(index=False),
                           file_name="closest_pair_points.csv", mime="text/csv")