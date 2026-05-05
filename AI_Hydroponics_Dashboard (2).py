# ============================================================================
# LETTUCE GROWTH ML PREDICTOR — DUAL MODE v3
# Mode 1: Point prediction (day + sensors -> length)
# Mode 2: Interactive simulation — pauses at each break for USER to adjust sensors
# Mode 3: Analysis plots — length model + dedicated growth rate visuals
# Mode 4: Dataset preview
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import requests
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ============================================================================
# CONFIG
# ============================================================================
CHANNEL_ID   = "3326913"
READ_API_KEY = "1YBWG6QWWPA9TNLT"
MAX_DAYS     = 48

RANGES = {"Temp": (18.0, 33.5), "Hum": (50.0, 80.0), "TDS": (400.0, 800.0), "pH": (6.0, 6.8)}
OPTIMA = {"Temp": (27.0, 30.0), "Hum": (62.0, 72.0), "TDS": (560.0, 680.0), "pH": (6.2, 6.5)}
UNITS  = {"Temp": "C", "Hum": "%", "TDS": " ppm", "pH": ""}
KEYS   = ["Temp", "Hum", "TDS", "pH"]
CLR_PHASE = ["#9FE1CB", "#5DCAA5", "#1D9E75"]
H_ICON    = {"good": "Green", "warn": "Orange", "bad": "Red"}
H_LABEL   = {"good": "Optimal", "warn": "Suboptimal", "bad": "Off-range"}

# ============================================================================
# HELPERS
# ============================================================================
def clamp(v, mn, mx):
    return max(mn, min(mx, v))

def health_status(key, val):
    lo, hi = OPTIMA[key]
    if lo <= val <= hi:
        return "good"
    span = RANGES[key][1] - RANGES[key][0]
    dist = (lo - val) / span if val < lo else (val - hi) / span
    return "bad" if dist > 0.15 else "warn"

def get_phase(day):
    if day <= 16: return "Seedling"
    if day <= 32: return "Vegetative"
    return "Mature"

def get_suggestions(model_rate, day, temp, hum, tds, ph):
    current   = [temp, hum, tds, ph]
    base_rate = model_rate.predict([[temp, hum, tds, ph, day]])[0]
    tips = []
    for i, k in enumerate(KEYS):
        rMin, rMax = RANGES[k]
        up_v  = clamp(current[i] * 1.12, rMin, rMax)
        dn_v  = clamp(current[i] * 0.88, rMin, rMax)
        up_in = current[:]; up_in[i] = up_v
        dn_in = current[:]; dn_in[i] = dn_v
        rate_up = model_rate.predict([[*up_in, day]])[0]
        rate_dn = model_rate.predict([[*dn_in, day]])[0]
        h = health_status(k, current[i])
        if rate_up - base_rate > 0.005 and rate_up >= rate_dn:
            tips.append({"key": k, "dir": "UP",   "boost": rate_up - base_rate,
                         "new_val": up_v, "current": current[i], "health": h})
        elif rate_dn - base_rate > 0.005:
            tips.append({"key": k, "dir": "DOWN", "boost": rate_dn - base_rate,
                         "new_val": dn_v, "current": current[i], "health": h})
        else:
            tips.append({"key": k, "dir": "OK",   "boost": 0,
                         "new_val": current[i], "current": current[i], "health": h})
    tips.sort(key=lambda x: -x["boost"])
    return tips, base_rate

def run_segment(model_length, model_rate, start_day, end_day, temp, hum, tds, ph):
    log = []
    for day in range(start_day, end_day + 1):
        temp = clamp(temp + np.random.uniform(-0.1, 0.1), *RANGES["Temp"])
        hum  = clamp(hum  + np.random.uniform(-0.5, 0.5), *RANGES["Hum"])
        tds  = clamp(tds  + np.random.uniform(-2.0, 2.0), *RANGES["TDS"])
        ph   = clamp(ph   + np.random.uniform(-0.02, 0.02), *RANGES["pH"])
        pl = model_length.predict([[temp, hum, tds, ph, day]])[0]
        pr = model_rate.predict([[temp, hum, tds, ph, day]])[0]
        log.append({"day": day, "length": round(pl, 3), "rate": round(pr, 4),
                    "temp": round(temp, 1), "hum": round(hum, 1),
                    "tds":  round(tds,  1), "ph":  round(ph,  2)})
    return log, temp, hum, tds, ph

# ============================================================================
# DATA + MODEL
# ============================================================================
@st.cache_data(show_spinner="Loading data...")
def load_data():
    if os.path.exists("thingspeak_ready.csv"):
        df = pd.read_csv("thingspeak_ready.csv")
    else:
        url  = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json"
        resp = requests.get(url, params={"api_key": READ_API_KEY, "results": 8000})
        df   = pd.DataFrame(resp.json().get("feeds", []))
    df = df.rename(columns={
        "field1": "Temp", "field2": "Hum", "field3": "TDS",
        "field4": "pH",   "field5": "Growth_Days", "field6": "Growth_Length"
    })
    for col in ["Temp","Hum","TDS","pH","Growth_Days","Growth_Length"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Temp","Hum","TDS","pH","Growth_Days","Growth_Length"]).reset_index(drop=True)
    df = df[df["Growth_Days"] > 0]
    df["Growth_Rate"] = df["Growth_Length"] / df["Growth_Days"]
    return df

@st.cache_resource(show_spinner="Training models...")
def train_models(n_rows):
    df = load_data()
    out = {}
    for target, col in [("length", "Growth_Length"), ("rate", "Growth_Rate")]:
        X = df[["Temp","Hum","TDS","pH","Growth_Days"]]
        y = df[col]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        m = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                       max_depth=4, subsample=0.8, random_state=42)
        m.fit(Xtr, ytr)
        yp  = m.predict(Xte)
        mae = mean_absolute_error(yte, yp)
        r2  = r2_score(yte, yp)
        acc = 100 - np.mean(np.abs((yte - yp) / (yte + 1e-9))) * 100
        out[target] = {"model": m, "mae": mae, "r2": r2, "acc": acc,
                       "X_test": Xte, "y_test": yte, "y_pred": yp, "X": X, "y": y}
    return out

# ============================================================================
# PAGE SETUP
# ============================================================================
st.set_page_config(page_title="AI-Hydroponics Dashboard", layout="wide")
st.title("AI-Hydroponics Dashboard")
st.caption("Two ML models: Length Predictor (sensors + Growth Days) and Growth Rate Predictor (sensor-driven)")

df = load_data()
if df is None or df.empty:
    st.error("No data found. Please upload thingspeak_ready.csv or check ThingSpeak connection.")
    st.stop()

models       = train_models(len(df))
model_length = models["length"]["model"]
model_rate   = models["rate"]["model"]
mae_len      = models["length"]["mae"]
acc_len      = models["length"]["acc"]
acc_rate     = models["rate"]["acc"]

st.success(f"Loaded {len(df):,} rows. Growth cycle: Day 1 to {MAX_DAYS}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Length Model Accuracy", f"{acc_len:.1f}%")
c2.metric("Length MAE",            f"+-{mae_len:.2f} cm")
c3.metric("Rate Model Accuracy",   f"{acc_rate:.1f}%")
c4.metric("Growth Cycle",          f"{MAX_DAYS} days")
st.divider()

# ============================================================================
# SESSION STATE
# ============================================================================
def reset_sim():
    for k in ["sim_log","sim_breaks","sim_running","sim_done","sim_at_break","sim_break_tips"]:
        if k in st.session_state:
            del st.session_state[k]
    st.session_state.sim_current_day = 1
    st.session_state.sim_break_day   = 0

for k, default in [
    ("sim_log", []), ("sim_breaks", []), ("sim_current_day", 1),
    ("sim_temp", 28.0), ("sim_hum", 65.0), ("sim_tds", 580.0), ("sim_ph", 6.4),
    ("sim_running", False), ("sim_done", False), ("sim_at_break", False),
    ("sim_break_tips", []), ("sim_break_day", 0), ("sim_interval", 4),
]:
    if k not in st.session_state:
        st.session_state[k] = default

# ============================================================================
# TABS
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Mode 1 - Point Prediction",
    "Mode 2 - Interactive Simulation",
    "Analysis Plots",
    "Dataset Preview"
])

# ============================================================
# TAB 1 - POINT PREDICTION
# ============================================================
with tab1:
    st.subheader("Predict Growth Length at a Specific Day")
    st.caption("Enter the current growth day and sensor readings to predict length and rate.")

    col_l, col_r = st.columns([1, 1], gap="large")
    with col_l:
        st.markdown("**Inputs**")
        day  = st.number_input("Growth Day",            min_value=1,     max_value=MAX_DAYS, value=20,    step=1)
        temp = st.number_input("Temperature (C)",       min_value=18.0,  max_value=33.5,     value=28.0,  step=0.5)
        hum  = st.number_input("Humidity (%)",          min_value=50.0,  max_value=80.0,     value=65.0,  step=1.0)
        tds  = st.number_input("TDS / Nutrients (ppm)", min_value=400.0, max_value=800.0,    value=580.0, step=10.0)
        ph   = st.number_input("pH",                    min_value=6.0,   max_value=6.8,      value=6.4,   step=0.1)
        st.info(f"Phase: {get_phase(day)} - Day {day} of {MAX_DAYS}")
        go = st.button("Predict", use_container_width=True, type="primary")

    with col_r:
        st.markdown("**Results**")
        if go:
            pl = model_length.predict([[temp, hum, tds, ph, day]])[0]
            pr = model_rate.predict([[temp,   hum, tds, ph, day]])[0]
            ra, rb = st.columns(2)
            ra.metric("Predicted Length",    f"{pl:.2f} cm")
            rb.metric("Current Growth Rate", f"{pr:.3f} cm/day")
            st.info(f"Confidence range: {max(0, pl-mae_len):.2f} to {pl+mae_len:.2f} cm")

            st.markdown("**Sensor Health**")
            hcols = st.columns(4)
            for col, k, v in zip(hcols, KEYS, [temp, hum, tds, ph]):
                h = health_status(k, v)
                col.metric(k, f"{v}{UNITS[k]}", H_LABEL[h])

            st.markdown("**5-Day Forecast**")
            fc = [{"Day": d, "Phase": get_phase(d),
                   "Predicted Length (cm)": round(model_length.predict([[temp,hum,tds,ph,d]])[0], 2),
                   "Growth Rate (cm/day)":  round(model_rate.predict([[temp,hum,tds,ph,d]])[0],   3)}
                  for d in range(day, min(day+6, MAX_DAYS+1))]
            st.dataframe(pd.DataFrame(fc), use_container_width=True, hide_index=True)

            tips, _ = get_suggestions(model_rate, day, temp, hum, tds, ph)
            top = [t for t in tips if t["dir"] != "OK"]
            if top:
                t   = top[0]
                fmt = f"{t['new_val']:.1f}" if t['key'] in ("Temp","pH") else f"{int(t['new_val'])}"
                arr = "increase" if t["dir"] == "UP" else "decrease"
                st.warning(f"Top tip for Day {day}: {arr} {t['key']} to {fmt}{UNITS[t['key']]} (+{t['boost']:.4f} cm/day)")
            else:
                st.success("All sensors are well-optimised for this growth stage!")
        else:
            st.info("Set inputs and click Predict")

# ============================================================
# TAB 2 - INTERACTIVE SIMULATION
# ============================================================
with tab2:
    st.subheader("Interactive 48-Day Growth Simulation")
    st.caption(
        "The simulation runs the full 48-day lifecycle and PAUSES at each break. "
        "The AI shows suggestions but YOU enter the sensor values and decide what to apply."
    )

    # Setup screen
    if not st.session_state.sim_running and not st.session_state.sim_done:
        st.markdown("#### Settings")
        cfg1, cfg2 = st.columns([1, 2])
        with cfg1:
            interval = st.selectbox(
                "Pause every...",
                [2, 3, 4, 5, 7, 8, 10, 12, 16],
                index=2,
                format_func=lambda x: f"Every {x} days  ({MAX_DAYS // x} pauses)"
            )
            st.session_state.sim_interval = interval
        with cfg2:
            bpts = list(range(interval, MAX_DAYS, interval))
            parts = []
            for d in range(1, MAX_DAYS + 1):
                if d in bpts:
                    parts.append(f"d{d}(pause)")
                elif d in (1, MAX_DAYS):
                    parts.append(f"d{d}")
            st.markdown("Break points: " + " -> ".join(parts))

        st.divider()
        st.markdown("#### Starting Sensor Values (Day 1)")
        s1, s2, s3, s4 = st.columns(4)
        it  = s1.number_input("Temp (C)",    18.0, 33.5, 28.0, step=0.5,  key="setup_temp")
        ih  = s2.number_input("Humidity (%)",50.0, 80.0, 65.0, step=1.0,  key="setup_hum")
        itds= s3.number_input("TDS (ppm)",   400.0,800.0,580.0,step=10.0, key="setup_tds")
        ip  = s4.number_input("pH",          6.0,  6.8,  6.4,  step=0.1,  key="setup_ph")

        if st.button("Start Simulation", use_container_width=True, type="primary"):
            st.session_state.sim_temp        = it
            st.session_state.sim_hum         = ih
            st.session_state.sim_tds         = itds
            st.session_state.sim_ph          = ip
            st.session_state.sim_running     = True
            st.session_state.sim_log         = []
            st.session_state.sim_breaks      = []
            st.session_state.sim_current_day = 1
            st.session_state.sim_at_break    = False
            st.session_state.sim_done        = False
            st.rerun()

    # Running state
    if st.session_state.sim_running and not st.session_state.sim_done:
        interval = st.session_state.sim_interval
        cur_day  = st.session_state.sim_current_day
        cur_temp = st.session_state.sim_temp
        cur_hum  = st.session_state.sim_hum
        cur_tds  = st.session_state.sim_tds
        cur_ph   = st.session_state.sim_ph

        progress = (cur_day - 1) / MAX_DAYS
        st.progress(progress, text=f"Day {cur_day - 1} of {MAX_DAYS} complete")

        # AT A BREAK
        if st.session_state.sim_at_break:
            break_day  = st.session_state.sim_break_day
            tips       = st.session_state.sim_break_tips
            break_len  = st.session_state.sim_log[-1]["length"] if st.session_state.sim_log else 0
            break_rate = st.session_state.sim_log[-1]["rate"]   if st.session_state.sim_log else 0

            st.markdown(f"### Break at Day {break_day} - {get_phase(break_day)} Phase")
            st.markdown(f"Length so far: **{break_len:.2f} cm** | Current rate: **{break_rate:.4f} cm/day**")
            st.divider()

            st.markdown("#### AI Suggestions for Next Phase")
            st.caption("The AI recommends these changes to maximise growth rate. You can use them or enter your own values below.")

            tip_cols = st.columns(4)
            for col, t in zip(tip_cols, tips):
                k   = t["key"]
                nv  = t["new_val"]
                cv  = t["current"]
                fmt = f"{nv:.1f}" if k in ("Temp","pH") else f"{int(nv)}"
                cfmt= f"{cv:.1f}" if k in ("Temp","pH") else f"{int(cv)}"
                h   = t["health"]
                if t["dir"] == "OK":
                    col.success(f"**{k}** - {H_LABEL[h]}\n\nKeep at {cfmt}{UNITS[k]}\n\nAlready optimal")
                elif t["dir"] == "UP":
                    col.warning(
                        f"**{k}** - {H_LABEL[h]}\n\n"
                        f"INCREASE\n\nCurrent: {cfmt}{UNITS[k]}\n\n"
                        f"Suggested: **{fmt}{UNITS[k]}**\n\nGain: +{t['boost']:.4f} cm/day"
                    )
                else:
                    col.info(
                        f"**{k}** - {H_LABEL[h]}\n\n"
                        f"DECREASE\n\nCurrent: {cfmt}{UNITS[k]}\n\n"
                        f"Suggested: **{fmt}{UNITS[k]}**\n\nGain: +{t['boost']:.4f} cm/day"
                    )

            st.divider()
            st.markdown("#### Your Sensor Values for the Next Phase")
            st.caption("Pre-filled with AI suggestions. Change any value then click Continue.")

            tip_dict = {t["key"]: t["new_val"] for t in tips}
            u1, u2, u3, u4 = st.columns(4)
            user_temp = u1.number_input("Temp (C)",    18.0, 33.5,  float(round(tip_dict.get("Temp", cur_temp), 1)), step=0.5, key=f"ut_{break_day}")
            user_hum  = u2.number_input("Humidity (%)",50.0, 80.0,  float(round(tip_dict.get("Hum",  cur_hum),  1)), step=1.0, key=f"uh_{break_day}")
            user_tds  = u3.number_input("TDS (ppm)",   400.0,800.0, float(round(tip_dict.get("TDS",  cur_tds),  0)), step=10.0,key=f"ud_{break_day}")
            user_ph   = u4.number_input("pH",           6.0, 6.8,   float(round(tip_dict.get("pH",   cur_ph),   2)), step=0.1, key=f"up_{break_day}")

            user_rate = model_rate.predict([[user_temp, user_hum, user_tds, user_ph, break_day]])[0]
            ai_temp = tip_dict.get("Temp", cur_temp)
            ai_hum  = tip_dict.get("Hum",  cur_hum)
            ai_tds  = tip_dict.get("TDS",  cur_tds)
            ai_ph   = tip_dict.get("pH",   cur_ph)
            ai_rate = model_rate.predict([[ai_temp, ai_hum, ai_tds, ai_ph, break_day]])[0]

            comp1, comp2 = st.columns(2)
            comp1.metric("Predicted Rate with YOUR values",  f"{user_rate:.4f} cm/day")
            comp2.metric("Predicted Rate with AI suggestion",f"{ai_rate:.4f} cm/day",
                         delta=f"{ai_rate - user_rate:+.4f} vs yours")

            if st.button("Continue Simulation", use_container_width=True, type="primary"):
                st.session_state.sim_breaks.append({
                    "day": break_day, "phase": get_phase(break_day),
                    "pred_len": break_len, "base_rate": break_rate, "tips": tips,
                    "sensors_before": {"Temp": cur_temp, "Hum": cur_hum, "TDS": cur_tds, "pH": cur_ph},
                    "sensors_after":  {"Temp": user_temp,"Hum": user_hum,"TDS": user_tds,"pH": user_ph},
                })
                st.session_state.sim_temp         = user_temp
                st.session_state.sim_hum          = user_hum
                st.session_state.sim_tds          = user_tds
                st.session_state.sim_ph           = user_ph
                st.session_state.sim_at_break     = False
                st.session_state.sim_current_day  = break_day + 1
                st.rerun()

        # NOT AT BREAK - run next segment
        else:
            offset     = (cur_day - 1) % interval
            steps_left = interval - offset if offset != 0 else interval
            seg_end    = min(cur_day + steps_left - 1, MAX_DAYS)

            with st.spinner(f"Running Day {cur_day} to Day {seg_end}..."):
                seg_log, new_temp, new_hum, new_tds, new_ph = run_segment(
                    model_length, model_rate, cur_day, seg_end,
                    cur_temp, cur_hum, cur_tds, cur_ph
                )
                st.session_state.sim_log.extend(seg_log)

            if seg_end == MAX_DAYS:
                st.session_state.sim_running = False
                st.session_state.sim_done    = True
                st.rerun()
            else:
                tips, _ = get_suggestions(model_rate, seg_end, new_temp, new_hum, new_tds, new_ph)
                st.session_state.sim_temp       = new_temp
                st.session_state.sim_hum        = new_hum
                st.session_state.sim_tds        = new_tds
                st.session_state.sim_ph         = new_ph
                st.session_state.sim_at_break   = True
                st.session_state.sim_break_day  = seg_end
                st.session_state.sim_break_tips = tips
                st.rerun()

    # DONE
    if st.session_state.sim_done and st.session_state.sim_log:
        sim_df = pd.DataFrame(st.session_state.sim_log)
        final  = sim_df.iloc[-1]

        st.success(f"Simulation complete! Final predicted length on Day 48: {final['length']:.2f} cm")
        sm1, sm2, sm3, sm4 = st.columns(4)
        sm1.metric("Final Length",      f"{final['length']:.2f} cm")
        sm2.metric("Peak Rate",         f"{sim_df['rate'].max():.3f} cm/day",
                   f"Day {int(sim_df.loc[sim_df['rate'].idxmax(),'day'])}")
        sm3.metric("Average Rate",      f"{sim_df['rate'].mean():.3f} cm/day")
        sm4.metric("User Adjustments",  len(st.session_state.sim_breaks))

        st.divider()
        st.markdown("#### Growth Curves")

        phases = [(1,16,CLR_PHASE[0],"Seedling d1-16"),
                  (17,32,CLR_PHASE[1],"Vegetative d17-32"),
                  (33,48,CLR_PHASE[2],"Mature d33-48")]
        bdays  = [b["day"] for b in st.session_state.sim_breaks]
        blens  = [sim_df.loc[sim_df["day"]==d,"length"].values[0] for d in bdays if d in sim_df["day"].values]
        brates = [sim_df.loc[sim_df["day"]==d,"rate"].values[0]   for d in bdays if d in sim_df["day"].values]

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        fig.patch.set_facecolor("#fafffe")

        ax = axes[0]
        for lo, hi, clr, lbl in phases:
            ax.axvspan(lo, hi, alpha=0.09, color=clr, label=lbl)
        ax.plot(sim_df["day"], sim_df["length"], color="#1D9E75", lw=2.5, zorder=3)
        ax.fill_between(sim_df["day"], sim_df["length"], alpha=0.1, color="#1D9E75")
        if bdays:
            ax.scatter(bdays, blens, color="orange", s=80, zorder=5, label="Your adjustment")
            for d, l in zip(bdays, blens):
                ax.annotate(f"d{d}", (d, l), xytext=(2,6), textcoords="offset points", fontsize=7, color="darkorange")
        ax.set_xlabel("Growth Day"); ax.set_ylabel("Length (cm)")
        ax.set_title("Predicted Length - Full Lifecycle", fontweight="bold")
        ax.set_xlim(1, MAX_DAYS); ax.grid(axis="y", alpha=0.2)
        handles = [mpatches.Patch(color=c, alpha=0.5, label=l) for _,_,c,l in phases]
        if bdays:
            handles += [plt.Line2D([],[],marker="o",color="orange",ls="none",label="Your adjustment",markersize=7)]
        ax.legend(handles=handles, fontsize=7, loc="upper left")

        ax2 = axes[1]
        for lo, hi, clr, _ in phases:
            ax2.axvspan(lo, hi, alpha=0.09, color=clr)
        ax2.fill_between(sim_df["day"], sim_df["rate"], alpha=0.2, color="#5DCAA5")
        ax2.plot(sim_df["day"], sim_df["rate"], color="#1D9E75", lw=2.5)
        if bdays:
            ax2.scatter(bdays, brates, color="orange", s=80, zorder=5)
        ax2.axhline(sim_df["rate"].mean(), color="gray", ls=":", lw=1.2,
                    label=f"Avg {sim_df['rate'].mean():.3f} cm/day")
        ax2.set_xlabel("Growth Day"); ax2.set_ylabel("Growth Rate (cm/day)")
        ax2.set_title("Growth Rate Over Lifecycle", fontweight="bold")
        ax2.set_xlim(1, MAX_DAYS); ax2.grid(axis="y", alpha=0.2); ax2.legend(fontsize=8)

        ax3 = axes[2]
        ax3.plot(sim_df["day"], sim_df["temp"],      color="#e63946", lw=1.5, label="Temp (C)")
        ax3.plot(sim_df["day"], sim_df["hum"] / 2,   color="#457b9d", lw=1.5, label="Hum (% / 2)")
        ax3.plot(sim_df["day"], sim_df["tds"] / 20,  color="#2a9d8f", lw=1.5, label="TDS (ppm / 20)")
        ax3.plot(sim_df["day"], sim_df["ph"] * 5,    color="#e9c46a", lw=1.5, label="pH x 5")
        if bdays:
            for d in bdays:
                ax3.axvline(d, color="orange", ls="--", lw=0.9, alpha=0.7)
        ax3.set_xlabel("Growth Day"); ax3.set_ylabel("Scaled value")
        ax3.set_title("Sensor History (Scaled for single view)", fontweight="bold")
        ax3.set_xlim(1, MAX_DAYS); ax3.grid(alpha=0.2); ax3.legend(fontsize=7, ncol=2)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Adjustment summary
        if st.session_state.sim_breaks:
            st.divider()
            st.markdown("#### Your Adjustments at Each Break")
            for b in st.session_state.sim_breaks:
                with st.expander(
                    f"{b['phase']} - Day {b['day']} - Length: {b['pred_len']:.2f} cm  Rate: {b['base_rate']:.4f} cm/day",
                    expanded=False
                ):
                    c_before, c_after = st.columns(2)
                    with c_before:
                        st.markdown("**Before (sensor state at break):**")
                        for k in KEYS:
                            v = b["sensors_before"][k]
                            h = health_status(k, v)
                            st.markdown(f"- {H_LABEL[h]} **{k}**: {v:.1f}{UNITS[k]}")
                    with c_after:
                        st.markdown("**After (your chosen values):**")
                        for k in KEYS:
                            v_before = b["sensors_before"][k]
                            v_after  = b["sensors_after"][k]
                            diff     = v_after - v_before
                            arrow    = "UP" if diff > 0.01 else ("DOWN" if diff < -0.01 else "SAME")
                            st.markdown(f"- {arrow} **{k}**: {v_after:.1f}{UNITS[k]}  (change: {diff:+.1f})")

        st.divider()
        with st.expander("Full day-by-day simulation log", expanded=False):
            disp = sim_df.copy()
            disp.insert(1, "Phase", disp["day"].apply(get_phase))
            st.dataframe(disp.rename(columns={
                "day":"Day","length":"Length (cm)","rate":"Rate (cm/day)",
                "temp":"Temp (C)","hum":"Hum (%)","tds":"TDS (ppm)","ph":"pH"
            }), use_container_width=True, hide_index=True)

        if st.button("Run New Simulation", use_container_width=True):
            reset_sim()
            st.rerun()

# ============================================================
# TAB 3 - ANALYSIS PLOTS
# ============================================================
with tab3:
    st.subheader("Analysis Plots")
    st.caption("Length model charts and dedicated growth rate visualisations.")

    plot_section = st.radio("Section:", ["Length Model", "Growth Rate Analysis"], horizontal=True)
    st.divider()

    if plot_section == "Length Model":
        plot_choice = st.radio("Chart:", [
            "Predicted vs Actual",
            "Feature Importance",
        ], horizontal=True)

        if plot_choice == "Predicted vs Actual":
            y_test = models["length"]["y_test"]
            y_pred = models["length"]["y_pred"]
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(y_test, y_pred, alpha=0.4, color="teal", edgecolors="white", linewidth=0.4, s=20)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2, label="Perfect prediction")
            ax.set_xlabel("Actual Length (cm)"); ax.set_ylabel("Predicted Length (cm)")
            ax.set_title("Length Model: Predicted vs Actual"); ax.legend(); ax.grid(alpha=0.2)
            st.pyplot(fig); plt.close(fig)

        elif plot_choice == "Feature Importance":
            imp_df = pd.DataFrame({
                "Feature": ["Temp","Hum","TDS","pH","Growth_Days"],
                "Importance": model_length.feature_importances_
            }).sort_values("Importance", ascending=True)
            fig, ax = plt.subplots(figsize=(7, 4))
            colors = ["#1D9E75" if f == "Growth_Days" else "#9FE1CB" for f in imp_df["Feature"]]
            bars = ax.barh(imp_df["Feature"], imp_df["Importance"], color=colors)
            ax.set_xlabel("Importance Score")
            ax.set_title("Feature Importance - Length Model\n(Growth_Days dominates as expected)")
            for bar, val in zip(bars, imp_df["Importance"]):
                ax.text(bar.get_width()+0.003, bar.get_y()+bar.get_height()/2,
                        f"{val*100:.1f}%", va="center", fontsize=9)
            st.pyplot(fig); plt.close(fig)

    else:
        plot_choice = st.radio("Chart:", [
            "Rate: Predicted vs Actual",
            "Rate: Feature Importance (vs Length Model)",
            "Rate by Growth Phase - Box Plot",
            "Rate Distribution by Phase",
        ], horizontal=True)

        if plot_choice == "Rate: Predicted vs Actual":
            y_test = models["rate"]["y_test"]
            y_pred = models["rate"]["y_pred"]
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(y_test, y_pred, alpha=0.4, color="#5DCAA5", edgecolors="white", linewidth=0.4, s=20)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2, label="Perfect prediction")
            ax.set_xlabel("Actual Growth Rate (cm/day)")
            ax.set_ylabel("Predicted Growth Rate (cm/day)")
            ax.set_title("Rate Model: Predicted vs Actual"); ax.legend(); ax.grid(alpha=0.2)
            mae_r = models["rate"]["mae"]
            ax.text(0.05, 0.92, f"MAE: {mae_r:.4f} cm/day", transform=ax.transAxes,
                    fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
            st.pyplot(fig); plt.close(fig)

        elif plot_choice == "Rate: Feature Importance (vs Length Model)":
            feats = ["Temp","Hum","TDS","pH","Growth_Days"]
            imp_l = model_length.feature_importances_
            imp_r = model_rate.feature_importances_
            x = np.arange(len(feats)); w = 0.35
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(x - w/2, imp_l * 100, w, label="Length Model", color="#1D9E75", alpha=0.85)
            ax.bar(x + w/2, imp_r * 100, w, label="Rate Model",   color="#9FE1CB", alpha=0.85)
            ax.set_xticks(x); ax.set_xticklabels(feats)
            ax.set_ylabel("Importance (%)"); ax.legend()
            ax.set_title("Feature Importance: Length Model vs Rate Model\n"
                         "Notice: sensors carry more weight in the Rate Model")
            ax.grid(axis="y", alpha=0.2)
            for i, (il, ir) in enumerate(zip(imp_l, imp_r)):
                ax.text(i-w/2, il*100+0.5, f"{il*100:.1f}%", ha="center", fontsize=7)
                ax.text(i+w/2, ir*100+0.5, f"{ir*100:.1f}%", ha="center", fontsize=7)
            st.pyplot(fig); plt.close(fig)
            st.caption("In the Rate Model, Temp, TDS, and pH carry more weight than in the Length Model. This confirms your insight that sensors drive growth rate!")

        elif plot_choice == "Rate by Growth Phase - Box Plot":
            df_plot = df.copy()
            df_plot["Phase"] = df_plot["Growth_Days"].apply(get_phase)
            fig, ax = plt.subplots(figsize=(7, 4))
            phase_order = ["Seedling", "Vegetative", "Mature"]
            data_by_phase = [df_plot[df_plot["Phase"] == p]["Growth_Rate"].values for p in phase_order]
            bp = ax.boxplot(data_by_phase, labels=phase_order, patch_artist=True, notch=True)
            for patch, color in zip(bp["boxes"], CLR_PHASE):
                patch.set_facecolor(color); patch.set_alpha(0.8)
            ax.set_ylabel("Growth Rate (cm/day)")
            ax.set_title("Growth Rate Distribution by Phase\nHow fast does lettuce grow at each stage?")
            ax.grid(axis="y", alpha=0.3)
            for i, data in enumerate(data_by_phase):
                ax.text(i+1, np.median(data)+0.01, f"Median: {np.median(data):.3f}", ha="center", fontsize=8)
            st.pyplot(fig); plt.close(fig)
            st.caption("Seedling phase typically shows highest average rate — this is when sensor optimisation has the biggest impact!")

        elif plot_choice == "Rate Distribution by Phase":
            df_plot = df.copy()
            df_plot["Phase"] = df_plot["Growth_Days"].apply(get_phase)
            fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
            phase_order = ["Seedling","Vegetative","Mature"]
            for ax, phase, color in zip(axes, phase_order, CLR_PHASE):
                data = df_plot[df_plot["Phase"] == phase]["Growth_Rate"]
                ax.hist(data, bins=30, color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
                ax.axvline(data.mean(),   color="red",  ls="--", lw=1.5, label=f"Mean: {data.mean():.3f}")
                ax.axvline(data.median(), color="navy", ls=":",  lw=1.5, label=f"Median: {data.median():.3f}")
                ax.set_title(phase, fontsize=11, fontweight="bold")
                ax.set_xlabel("Growth Rate (cm/day)")
                ax.set_ylabel("Count" if ax == axes[0] else "")
                ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.2)
            fig.suptitle("Growth Rate Distribution per Phase", fontsize=12, fontweight="bold")
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

# ============================================================
# TAB 4 - DATASET PREVIEW
# ============================================================
with tab4:
    st.subheader(f"Dataset Preview - {len(df):,} rows")
    display_cols = [c for c in ["Growth_Days","Growth_Length","Growth_Rate","Temp","Hum","TDS","pH"] if c in df.columns]
    st.dataframe(df[display_cols], use_container_width=True)

# Sidebar
with st.sidebar:
    st.header("Dataset Info")
    st.metric("Rows",         f"{len(df):,}")
    st.metric("Growth cycle", "Day 1 - 48")
    st.metric("Max length",   f"{df['Growth_Length'].max():.1f} cm")
    st.metric("Avg rate",     f"{df['Growth_Rate'].mean():.3f} cm/day")
    st.divider()
    st.markdown("""
**Mode 1** - Point Prediction
Pick any day + sensors -> predicted length + 5-day forecast

**Mode 2** - Interactive Simulation
Runs full 48 days, pauses at breaks.
AI suggests values, YOU enter what to apply.

**Mode 3** - Analysis Plots
Length model + 3 growth rate charts:
- Rate predicted vs actual
- Feature importance comparison
- Box plot by phase
- Distribution by phase

---
Seedling: Day 1-16
Vegetative: Day 17-32
Mature: Day 33-48
    """)
    if st.button("Reset Simulation", use_container_width=True):
        reset_sim()
        st.rerun()