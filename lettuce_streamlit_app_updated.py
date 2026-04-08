import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ── ThingSpeak config ────────────────────────────────────────────────────────
CHANNEL_ID  = "3326913"
READ_API_KEY = "1YBWG6QWWPA9TNLT"

# ── Domain constants ─────────────────────────────────────────────────────────
RANGES = {"Temp": (18.0, 33.5), "Hum": (50.0, 80.0), "TDS": (400.0, 800.0), "pH": (6.0, 6.8)}
OPTIMA = {"Temp": (27.0, 30.0), "Hum": (62.0, 72.0), "TDS": (560.0, 680.0), "pH": (6.2, 6.5)}
UNITS  = {"Temp": "°C",         "Hum": "%",            "TDS": " ppm",          "pH": ""}
KEYS   = ["Temp", "Hum", "TDS", "pH"]
STEP   = 0.12

CLR_GOOD  = "#2e7d32"
CLR_WARN  = "#e65100"
CLR_BAD   = "#b71c1c"
CLR_PHASE = ["#9FE1CB", "#5DCAA5", "#1D9E75"]

# ── Helper functions ─────────────────────────────────────────────────────────
def clamp(v, mn, mx):
    return max(mn, min(mx, v))

def health_status(key, val):
    lo, hi = OPTIMA[key]
    if lo <= val <= hi:
        return "good"
    span = RANGES[key][1] - RANGES[key][0]
    dist = (lo - val) / span if val < lo else (val - hi) / span
    return "bad" if dist > 0.15 else "warn"

def growth_curve(model, temp, hum, tds, ph, days=48, seed=None):
    rng   = np.random.default_rng(seed)
    final = model.predict([[temp, hum, tds, ph]])[0]
    pts   = []
    for d in range(1, days + 1):
        logistic = final / (1 + np.exp(-0.18 * (d - 12)))
        pts.append(max(0.0, logistic + rng.normal(0, 0.4)))
    return np.array(pts)

def get_suggestions(model, temp, hum, tds, ph):
    vals = [temp, hum, tds, ph]
    base = model.predict([vals])[0]
    tips = []
    for i, k in enumerate(KEYS):
        rMin, rMax = RANGES[k]
        up_v  = clamp(vals[i] * 1.12, rMin, rMax)
        dn_v  = clamp(vals[i] * 0.88, rMin, rMax)
        up_in = vals[:]; up_in[i] = up_v
        dn_in = vals[:]; dn_in[i] = dn_v
        pred_up = model.predict([up_in])[0]
        pred_dn = model.predict([dn_in])[0]
        h = health_status(k, vals[i])
        if pred_up - base > 0.2 and pred_up >= pred_dn:
            tips.append({"key": k, "dir": "up",   "boost": pred_up - base, "new_val": up_v,    "health": h})
        elif pred_dn - base > 0.2:
            tips.append({"key": k, "dir": "down", "boost": pred_dn - base, "new_val": dn_v,    "health": h})
        else:
            tips.append({"key": k, "dir": "ok",   "boost": 0,              "new_val": vals[i], "health": h})
    tips.sort(key=lambda x: -x["boost"])
    return tips, base

# ── Load data from ThingSpeak ─────────────────────────────────────────────────
@st.cache_data(show_spinner="Fetching data from ThingSpeak...")
def load_thingspeak():
    url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json"
    resp = requests.get(url, params={"api_key": READ_API_KEY, "results": 8000})
    data = resp.json()
    feeds = data.get("feeds", [])
    if not feeds:
        return None
    df = pd.DataFrame(feeds)
    df = df.rename(columns={
        "field1": "Temp",
        "field2": "Hum",
        "field3": "TDS",
        "field4": "pH",
        "field5": "Growth_Days",
        "field6": "Growth_Length",
    })
    for col in ["Temp","Hum","TDS","pH","Growth_Days","Growth_Length"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Temp","Hum","TDS","pH","Growth_Length"])
    df = df.reset_index(drop=True)
    return df

# ── App layout ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="🥬 Lettuce Growth ML Predictor", layout="wide")
st.title("🥬 Lettuce Growth ML Predictor")

# Load data
df = load_thingspeak()

if df is None or df.empty:
    st.error("⚠️ No data found in ThingSpeak channel. Please upload data first.")
    st.stop()

st.success(f"✅ Loaded **{len(df)} rows** from ThingSpeak channel `{CHANNEL_ID}`")

# ── Session state defaults ────────────────────────────────────────────────────
if "model"        not in st.session_state: st.session_state.model        = None
if "mae"          not in st.session_state: st.session_state.mae          = None
if "iter_count"   not in st.session_state: st.session_state.iter_count   = 0
if "iter_history" not in st.session_state: st.session_state.iter_history = []
if "prev_curve"   not in st.session_state: st.session_state.prev_curve   = None
if "temp_val"     not in st.session_state: st.session_state.temp_val     = 28.0
if "hum_val"      not in st.session_state: st.session_state.hum_val      = 65.0
if "tds_val"      not in st.session_state: st.session_state.tds_val      = 550.0
if "ph_val"       not in st.session_state: st.session_state.ph_val       = 6.4

# ═══════════════════════════════════════════════════════════════════════════════
#  TOP BAR — three columns
# ═══════════════════════════════════════════════════════════════════════════════
col_model, col_sensor, col_result = st.columns(3)

# ── Column 1: Data & Model ────────────────────────────────────────────────────
with col_model:
    st.subheader("📊 Data & Model")
    if st.button("🚀 Train Model", use_container_width=True):
        X = df[["Temp","Hum","TDS","pH"]]
        y = df["Growth_Length"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        st.session_state.model = model
        st.session_state.mae   = mae
        st.session_state.X     = X
        st.session_state.y     = y
        st.session_state.r2    = r2
        st.session_state.acc   = 100 - mape
        st.success("✓ Model trained!")

    if st.session_state.model:
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy",  f"{st.session_state.acc:.2f}%")
        m2.metric("MAE (cm)",  f"{st.session_state.mae:.3f}")
        m3.metric("R² Score",  f"{st.session_state.r2:.4f}")
    else:
        st.info("Model not trained yet.")

# ── Column 2: Sensor Inputs ───────────────────────────────────────────────────
with col_sensor:
    st.subheader("🌡️ Sensor Inputs")
    temp = st.number_input("Temperature (°C)",     min_value=18.0, max_value=33.5, value=st.session_state.temp_val, step=0.5)
    hum  = st.number_input("Humidity (%)",         min_value=50.0, max_value=80.0, value=st.session_state.hum_val,  step=1.0)
    tds  = st.number_input("TDS / Nutrients (ppm)",min_value=400.0,max_value=800.0,value=st.session_state.tds_val,  step=10.0)
    ph   = st.number_input("pH",                   min_value=6.0,  max_value=6.8,  value=st.session_state.ph_val,   step=0.1)

    st.session_state.temp_val = temp
    st.session_state.hum_val  = hum
    st.session_state.tds_val  = tds
    st.session_state.ph_val   = ph

    b1, b2, b3 = st.columns(3)
    do_predict    = b1.button("🔮 Predict",           use_container_width=True)
    do_simulate   = b2.button("📈 Simulate",          use_container_width=True)
    do_apply_sim  = b3.button("✦ Apply & Re-simulate",use_container_width=True)

# ── Column 3: Prediction Result ───────────────────────────────────────────────
with col_result:
    st.subheader("🎯 Prediction Result")

    if do_predict or do_simulate or do_apply_sim:
        if not st.session_state.model:
            st.warning("Please train the model first.")
        else:
            model = st.session_state.model
            mae   = st.session_state.mae

            # Apply & Re-simulate — nudge sensor values
            if do_apply_sim and st.session_state.iter_history:
                tips, _ = get_suggestions(model, temp, hum, tds, ph)
                vals = {"Temp": temp, "Hum": hum, "TDS": tds, "pH": ph}
                for t in tips:
                    if t["dir"] == "up":
                        vals[t["key"]] = clamp(vals[t["key"]] * (1 + STEP), *RANGES[t["key"]])
                    elif t["dir"] == "down":
                        vals[t["key"]] = clamp(vals[t["key"]] * (1 - STEP), *RANGES[t["key"]])
                temp, hum, tds, ph = vals["Temp"], vals["Hum"], vals["TDS"], vals["pH"]
                st.session_state.temp_val = round(temp, 1)
                st.session_state.hum_val  = round(hum,  1)
                st.session_state.tds_val  = round(tds,  1)
                st.session_state.ph_val   = round(ph,   2)

            prediction = model.predict([[temp, hum, tds, ph]])[0]
            st.session_state.prediction = prediction

            st.metric("Predicted Growth Length", f"{prediction:.2f} cm")
            st.info(f"Confidence Range: **{prediction - mae:.2f} – {prediction + mae:.2f} cm**")

            if do_simulate or do_apply_sim:
                st.session_state.iter_count += 1
                curve = growth_curve(model, temp, hum, tds, ph, seed=st.session_state.iter_count)
                tips, base = get_suggestions(model, temp, hum, tds, ph)
                delta = prediction - st.session_state.iter_history[-1]["pred"] if st.session_state.iter_history else None
                st.session_state.iter_history.append({
                    "iter": st.session_state.iter_count, "pred": prediction,
                    "delta": delta, "temp": temp, "hum": hum, "tds": tds, "ph": ph
                })
                st.session_state.prev_curve    = st.session_state.get("curr_curve")
                st.session_state.curr_curve    = curve
                st.session_state.last_tips     = tips
                st.session_state.last_base     = base
                st.session_state.last_sim_vals = (temp, hum, tds, ph)

    if st.button("🔄 Reset History", use_container_width=True):
        st.session_state.iter_count   = 0
        st.session_state.iter_history = []
        st.session_state.prev_curve   = None
        st.session_state.pop("curr_curve", None)
        st.session_state.pop("last_tips", None)
        st.rerun()

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🌱 Simulation & AI Suggestions",
    "📉 Analysis Plots",
    "🗄️ Dataset Preview",
    "📜 Optimization History"
])

# ── Tab 1: Simulation & AI Suggestions ───────────────────────────────────────
with tab1:
    if "curr_curve" not in st.session_state:
        st.info("Run a simulation using the **Simulate** or **Apply & Re-simulate** button above.")
    else:
        curve      = st.session_state.curr_curve
        prev_curve = st.session_state.prev_curve
        tips       = st.session_state.last_tips
        base       = st.session_state.last_base
        mae        = st.session_state.mae
        prediction = st.session_state.prediction
        sim_temp, sim_hum, sim_tds, sim_ph = st.session_state.last_sim_vals
        iter_num   = st.session_state.iter_count

        left_col, right_col = st.columns([3, 1])

        with left_col:
            days = np.arange(1, len(curve) + 1)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            fig.suptitle(f"Growth Simulation — Iteration {iter_num}", fontsize=12, fontweight="bold")

            # Growth bar chart
            ax = axes[0]
            phase_spans  = [(1, 16), (17, 32), (33, 48)]
            phase_labels = ["Seedling (d1–16)", "Vegetative (d17–32)", "Mature (d33–48)"]
            for idx, (lo, hi) in enumerate(phase_spans):
                ax.axvspan(lo, hi, alpha=0.07, color=CLR_PHASE[idx])
            if prev_curve is not None:
                ax.plot(days, prev_curve, color="#B4B2A9", linewidth=1.0, linestyle="--", alpha=0.6, label="Previous run")
            bar_colors = [CLR_PHASE[0] if d <= 16 else CLR_PHASE[1] if d <= 32 else CLR_PHASE[2] for d in days]
            ax.bar(days, curve, color=bar_colors, width=0.8, alpha=0.85)
            ax.axhline(prediction, color=CLR_PHASE[2], linestyle=":", linewidth=1.4, alpha=0.8)
            ax.text(49, prediction, f" {prediction:.1f}cm", va="center", fontsize=8, color=CLR_PHASE[2])
            ax.set_xlabel("Day"); ax.set_ylabel("Length (cm)")
            ax.set_xlim(0, 50); ax.grid(axis="y", alpha=0.25)
            handles = [mpatches.Patch(color=c, label=l, alpha=0.7) for c, l in zip(CLR_PHASE, phase_labels)]
            if prev_curve is not None:
                handles.append(plt.Line2D([0],[0], color="#B4B2A9", linestyle="--", label="Previous run"))
            ax.legend(handles=handles, fontsize=7, loc="upper left")

            # Sensor health bars
            ax2 = axes[1]
            ax2.set_title("Sensor Health vs Optimal Range", fontsize=10)
            current_vals = {"Temp": sim_temp, "Hum": sim_hum, "TDS": sim_tds, "pH": sim_ph}
            clr_map = {"good": CLR_GOOD, "warn": CLR_WARN, "bad": CLR_BAD}
            for i, k in enumerate(KEYS):
                rMin, rMax = RANGES[k]
                oLo,  oHi  = OPTIMA[k]
                val = current_vals[k]
                norm = lambda v, mn=rMin, mx=rMax: (v - mn) / (mx - mn)
                ax2.barh(i, 1,                    left=0,           color="#F1EFE8", height=0.5)
                ax2.barh(i, norm(oHi)-norm(oLo),  left=norm(oLo),   color="#EAF3DE", height=0.5)
                s = health_status(k, val)
                ax2.plot(norm(val), i, "o", markersize=9, color=clr_map[s])
                ax2.text(1.03, i, f"{val}{UNITS[k]}", va="center", fontsize=8)
                ax2.text(-0.03, i, k, va="center", ha="right", fontsize=8)
            ax2.set_xlim(-0.18, 1.25); ax2.set_ylim(-0.6, len(KEYS)-0.4); ax2.axis("off")
            legend_els = [mpatches.Patch(color=c, label=l) for c, l in
                          [("#EAF3DE","Optimal zone"),(CLR_GOOD,"Good"),(CLR_WARN,"Suboptimal"),(CLR_BAD,"Off-range")]]
            ax2.legend(handles=legend_els, loc="lower right", fontsize=7)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with right_col:
            # Sensor health labels
            st.markdown("**🔍 Sensor Health**")
            health_map  = {"good": "🟢 Optimal", "warn": "🟠 Suboptimal", "bad": "🔴 Off-range"}
            for k, val in current_vals.items():
                s = health_status(k, val)
                st.markdown(f"**{k}**: {health_map[s]}")

            st.divider()

            # AI Suggestions
            st.markdown("**🤖 AI Suggestions**")
            st.caption(f"Predicted: **{prediction:.2f} cm** (±{mae:.2f} cm)")
            actionable = [t for t in tips if t["dir"] != "ok"]
            if not actionable:
                st.success("✓ All sensors are well-optimised! Growth is near-peak.")
            else:
                for t in tips:
                    k, d = t["key"], t["dir"]
                    nv = t["new_val"]
                    fmt = f"{nv:.1f}" if k in ("Temp","pH") else str(int(round(nv / 10) * 10))
                    h_tag = f"[{t['health'].upper()}]"
                    if d == "ok":
                        st.markdown(f"✅ **{k}** {h_tag} — no change needed")
                    else:
                        arrow = "⬆️ INCREASE" if d == "up" else "⬇️ DECREASE"
                        st.markdown(f"{arrow} **{k}** {h_tag} → ~{fmt}{UNITS[k]} **(+{t['boost']:.2f} cm)**")

# ── Tab 2: Analysis Plots ─────────────────────────────────────────────────────
with tab2:
    if not st.session_state.model:
        st.info("Train the model first to see analysis plots.")
    else:
        model = st.session_state.model
        X     = st.session_state.X
        y     = st.session_state.y

        plot_choice = st.radio("Choose plot:", [
            "Predicted vs Actual",
            "Feature Importance",
            "Correlation Heatmap",
            "Sensor vs Growth"
        ], horizontal=True)

        if plot_choice == "Predicted vs Actual":
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            y_pred = model.predict(X_test)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(y_test, y_pred, alpha=0.5, color="teal", edgecolors="white", linewidth=0.4)
            ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)
            ax.set_xlabel("Actual Length (cm)"); ax.set_ylabel("Predicted Length (cm)")
            ax.set_title("Predicted vs Actual Growth Length")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

        elif plot_choice == "Feature Importance":
            imp_df = pd.DataFrame({"Sensor": KEYS, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
            fig, ax = plt.subplots(figsize=(6, 4))
            colors = sns.color_palette("viridis", len(KEYS))
            bars = ax.barh(imp_df["Sensor"], imp_df["Importance"], color=colors)
            ax.set_xlabel("Importance Score"); ax.set_title("Feature Importance for Growth Length")
            ax.invert_yaxis()
            for bar, val in zip(bars, imp_df["Importance"]):
                ax.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2, f"{val*100:.1f}%", va="center", fontsize=9)
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

        elif plot_choice == "Correlation Heatmap":
            cols = [c for c in ["Temp","Hum","TDS","pH","Growth_Days","Growth_Length"] if c in df.columns]
            corr = df[cols].corr()
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(corr, annot=True, cmap="RdYlGn", center=0, fmt=".2f", ax=ax, annot_kws={"size": 8})
            ax.set_title("Correlation Heatmap")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

        elif plot_choice == "Sensor vs Growth":
            fig, axes = plt.subplots(1, 4, figsize=(14, 4))
            for ax, k in zip(axes, KEYS):
                if k not in df.columns: continue
                oLo, oHi = OPTIMA[k]
                ax.axvspan(oLo, oHi, alpha=0.15, color=CLR_GOOD, label="Optimal zone")
                ax.scatter(df[k], df["Growth_Length"], alpha=0.3, s=8, color="#1D9E75")
                ax.set_xlabel(f"{k} ({UNITS[k].strip()})", fontsize=9)
                ax.set_ylabel("Growth (cm)" if k == "Temp" else "", fontsize=9)
                ax.set_title(k, fontsize=10); ax.grid(alpha=0.2)
            fig.suptitle("Sensor Values vs Growth Length", fontsize=11, fontweight="bold")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

# ── Tab 3: Dataset Preview ────────────────────────────────────────────────────
with tab3:
    st.subheader(f"ThingSpeak Data — {len(df)} rows")
    display_cols = [c for c in ["created_at","Temp","Hum","TDS","pH","Growth_Days","Growth_Length"] if c in df.columns]
    rename_map = {
        "Temp": "Temperature (°C)", "Hum": "Humidity (%)",
        "TDS": "TDS Value (ppm)",   "pH": "pH Level",
        "Growth_Days": "Growth Days", "Growth_Length": "Lettuce Growth Length (cm)"
    }
    st.dataframe(df[display_cols].rename(columns=rename_map), use_container_width=True)

# ── Tab 4: Optimization History ───────────────────────────────────────────────
with tab4:
    st.subheader("📜 Optimization History")
    if not st.session_state.iter_history:
        st.info("No iterations yet. Run a simulation to start tracking.")
    else:
        hist_df = pd.DataFrame(st.session_state.iter_history)
        hist_df["delta"] = hist_df["delta"].apply(lambda x: f"+{x:.2f}" if x and x >= 0 else (f"{x:.2f}" if x else "—"))
        hist_df.columns = [c.capitalize() for c in hist_df.columns]
        st.dataframe(hist_df, use_container_width=True)
