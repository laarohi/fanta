import os
import sys
import subprocess
import pandas as pd
import numpy as np
import yaml
import streamlit as st
from pathlib import Path

APP_TITLE = "Fantacalcio Auction Model — Serie A 2025/26"

# -------------------- caching + I/O --------------------

@st.cache_data
def load_config(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

@st.cache_data
def read_csv_safe(path: str):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def try_run_pipeline(script_path: str):
    try:
        res = subprocess.run([sys.executable, script_path], capture_output=True, text=True, check=False)
        ok = res.returncode == 0
        return ok, res.stdout, res.stderr
    except Exception as e:
        return False, "", str(e)

def role_label(r):
    return {"P":"GK","D":"DEF","C":"MID","A":"FWD"}.get(str(r).upper()[:1], str(r))

def load_outputs(cfg):
    proj_path = cfg["output"]["projections_csv"]
    prices_path = cfg["output"]["prices_csv"]
    proj = read_csv_safe(proj_path)
    prices = read_csv_safe(prices_path)
    if "role" in proj.columns:
        proj["Role"] = proj["role"].map(role_label)
    if "role" in prices.columns:
        prices["Role"] = prices["role"].map(role_label)
    return proj, prices, proj_path, prices_path

def filter_table(df, roles, teams, search, min_pv):
    if df.empty: return df
    out = df.copy()
    if roles:
        out = out[out["Role"].isin(roles)]
    if teams:
        out = out[out["team"].isin(teams)]
    if search:
        s = search.lower()
        out = out[out["name"].str.lower().str.contains(s)]
    if min_pv is not None and "Pv_hat" in out.columns:
        out = out[out["Pv_hat"] >= min_pv]
    return out

def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

# -------------------- watchlist state --------------------

if "watchlist" not in st.session_state:
    st.session_state.watchlist = set()

def add_to_watchlist(names):
    for n in names:
        st.session_state.watchlist.add(n)

def remove_from_watchlist(names):
    for n in names:
        st.session_state.watchlist.discard(n)

# -------------------- points + pricing (local re-compute for tilts) --------------------

def compute_points_local(df):
    out = df.copy()

    # Skaters
    sk = out.role.isin(['D','C','A'])
    out.loc[sk, 'Pts_hat_loc'] = (
        out.loc[sk,'Pv_hat']*out.loc[sk,'Mv_hat']
        + 3*out.loc[sk,'G_hat']
        + 1*out.loc[sk,'A_hat']
        - 0.5*out.loc[sk,'YC_hat']
        - 2.0*out.loc[sk,'RC_hat']
        - 2.0*out.loc[sk,'OG_hat']
        + out.loc[sk,'Mod_share_hat'].fillna(0.0)
    )

    # Goalkeepers
    gk = out.role=='P'
    if "GC_perPv_hat" in out.columns and out["GC_perPv_hat"].notna().any():
        gcpp = out["GC_perPv_hat"].fillna( out["GC_hat"].fillna(0) / out["Pv_hat"].replace(0, np.nan) ).fillna(0)
    else:
        gcpp = out["GC_hat"].fillna(0) / out["Pv_hat"].replace(0, np.nan)
        gcpp = gcpp.fillna(0)
    cs = np.exp(-gcpp) * out.loc[gk,'Pv_hat']
    out.loc[gk,'CS_hat'] = cs
    out.loc[gk, 'Pts_hat_loc'] = (
        out.loc[gk,'Pv_hat']*out.loc[gk,'Mv_hat']
        - 1.0*out.loc[gk,'GC_hat']
        + 2.0*out.loc[gk,'CS_hat']
        + 3.0*out.loc[gk,'PKsave_hat'].fillna(0.0)
        - 0.5*out.loc[gk,'YC_hat']
        - 2.0*out.loc[gk,'RC_hat']
        - 2.0*out.loc[gk,'OG_hat']
        + out.loc[gk,'Mod_share_hat'].fillna(0.0)
    )
    return out

def replacement_cut_from_cfg(cfg):
    return cfg["params"]["replacement_cut"]

def vorp_and_prices(df, cfg):
    out = df.copy()
    cuts = replacement_cut_from_cfg(cfg)
    repl = {}
    for role, N in cuts.items():
        pool = out[out.role==role].sort_values('Pts_hat_loc', ascending=False).reset_index(drop=True)
        if len(pool)>=N:
            repl[role] = float(pool.loc[N-1,'Pts_hat_loc'])
        elif len(pool)>0:
            repl[role] = float(pool['Pts_hat_loc'].min())
        else:
            repl[role] = 0.0
    out["ReplacementPts"] = out["role"].map(repl)
    out["VORP_loc"] = (out["Pts_hat_loc"] - out["ReplacementPts"]).clip(lower=0)

    total_vorp = out["VORP_loc"].sum()
    model_price = out["VORP_loc"] * (10000.0/total_vorp if total_vorp>0 else 0.0)

    # blend with FVM
    w = cfg["params"]["pricing"]["blend_fvm_weight"]
    if "FVM" in out.columns and out["FVM"].sum() > 0:
        fvm_scaled = (out["FVM"]/out["FVM"].sum())*10000.0
    else:
        fvm_scaled = model_price.copy()
    blended = (1-w)*fvm_scaled + w*model_price

    # room bias
    bias = cfg["params"]["pricing"]["room_bias"]
    biased = blended * out["role"].map(bias).astype(float)

    # star exponent
    alpha = cfg["params"]["pricing"]["star_exponent"]
    if biased.max() > 0:
        scaled = (biased/biased.max())**alpha
        premium = scaled * biased.sum() / (scaled.sum() if scaled.sum()>0 else 1.0)
    else:
        premium = biased

    # renormalize to 10k
    final = premium * (10000.0/premium.sum() if premium.sum()>0 else 0.0)

    out["Price_local"] = final
    return out

# -------------------- app UI --------------------

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.subheader("Config & Build")
    # Use a portable default path
    default_cfg = "config_csv.yaml"
    cfg_path = st.text_input("Path to config_csv.yaml", value=default_cfg)
    run_build = st.button("Run pipeline (recompute projections)")
    st.caption("If your CSVs changed (rigoristi, setpieces, AFCON, starters), run this.")

    st.divider()
    st.subheader("Display Filters")
    role_filter = st.multiselect("Roles", options=["GK","DEF","MID","FWD"], default=[])
    team_filter = []
    search_text = st.text_input("Search player")
    min_pv = st.number_input("Min Pv̂ (appearances)", min_value=0.0, value=0.0, step=1.0)

    st.divider()
    st.subheader("Role multipliers (view-only)")
    col1, col2 = st.columns(2)
    with col1:
        bias_GK = st.number_input("GK ×", value=1.00, step=0.05, format="%.2f")
        bias_DEF = st.number_input("DEF ×", value=1.00, step=0.05, format="%.2f")
    with col2:
        bias_MID = st.number_input("MID ×", value=1.00, step=0.05, format="%.2f")
        bias_FWD = st.number_input("FWD ×", value=1.00, step=0.05, format="%.2f")
    bias_mult = {"GK":bias_GK, "DEF":bias_DEF, "MID":bias_MID, "FWD":bias_FWD}

# Load config + outputs
cfg = load_config(cfg_path)
proj, prices, proj_path, prices_path = load_outputs(cfg)

# Team selector
if not prices.empty and "team" in prices.columns:
    with st.sidebar:
        team_options = sorted(prices["team"].dropna().unique().tolist())
        team_filter = st.multiselect("Teams", options=team_options, default=[])

# Run pipeline
if run_build:
    script_path = "build_model_csv.py"
    ok, out, err = try_run_pipeline(script_path)
    with st.expander("Build log", expanded=True):
        st.code(out or "(no stdout)")
        if err:
            st.code(err)
    if ok:
        st.success("Pipeline completed. Reloading tables...")
        proj, prices, proj_path, prices_path = load_outputs(cfg)
    else:
        st.error("Pipeline failed. See logs above.")

# Tilts uploader
st.sidebar.divider()
st.sidebar.subheader("What-if tilts (advanced)")
st.sidebar.caption("Upload CSV: name,team,delta_Mv,delta_G,delta_A,cap_Pv,price_mult")
tilts_file = st.sidebar.file_uploader("Upload tilts.csv", type=["csv"])
use_tilts = st.sidebar.checkbox("Apply tilts and recompute prices", value=False)

# Prepare a working projection DataFrame (prefer projections.csv)
base = proj.copy() if not proj.empty else prices.copy()
if base.empty and not prices.empty:
    base = prices.copy()

# Ensure required columns exist
for col in ["Pv_hat","Mv_hat","G_hat","A_hat","YC_hat","RC_hat","OG_hat","GC_hat","PKsave_hat","Mod_share_hat","role","name","team"]:
    if col not in base.columns:
        base[col] = 0.0 if col not in ["role","name","team"] else base.get(col, "")

# Apply tilts
adj = base.copy()
if tilts_file is not None:
    tilts = pd.read_csv(tilts_file)
    def norm(s): return str(s).strip().lower()
    tilts["_key"] = tilts["name"].map(norm) + "||" + tilts["team"].map(norm)
    adj["_key"] = adj["name"].map(norm) + "||" + adj["team"].map(norm)
    adj = adj.merge(tilts, on="_key", how="left", suffixes=("","_tilt"))
    for c_df, c_t in [("Mv_hat","delta_Mv"), ("G_hat","delta_G"), ("A_hat","delta_A")]:
        if c_t in adj.columns:
            adj[c_df] = adj[c_df] + adj[c_t].fillna(0.0)
    if "cap_Pv" in adj.columns:
        adj["Pv_hat"] = np.minimum(adj["Pv_hat"], adj["cap_Pv"].fillna(np.inf))
    if "price_mult" not in adj.columns:
        adj["price_mult"] = 1.0
else:
    adj["price_mult"] = 1.0

# Recompute points & prices if requested
if use_tilts:
    pts_df = compute_points_local(adj)
    priced = vorp_and_prices(pts_df, cfg)
    priced["Role"] = priced["role"].map(role_label)
    priced["StopPrice"] = priced["Price_local"]
    # Apply post-hoc role multipliers
    for k, letter in {"GK":"P","DEF":"D","MID":"C","FWD":"A"}.items():
        mask = priced["role"]==letter
        priced.loc[mask, "StopPrice"] = priced.loc[mask, "StopPrice"] * float(bias_mult[k])
    # Per-player price multiplier from tilts
    if "price_mult" in priced.columns:
        priced["StopPrice"] = priced["StopPrice"] * priced["price_mult"].fillna(1.0)
    working_prices = priced
else:
    working_prices = prices.copy()
    if not working_prices.empty:
        working_prices["Role"] = working_prices["role"].map(role_label)
        working_prices["StopPrice"] = working_prices["Price_final"]
        for k, letter in {"GK":"P","DEF":"D","MID":"C","FWD":"A"}.items():
            mask = working_prices["role"]==letter
            working_prices.loc[mask, "StopPrice"] = working_prices.loc[mask, "StopPrice"] * float(bias_mult[k])

# ------------ tabs ------------

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["🏷️ Stop Prices","📊 Projections","🧮 Planner","🎯 Targets by Round","⭐ Watchlist","ℹ️ About"]
)

with tab1:
    st.subheader("Auction stop prices")
    if working_prices.empty:
        st.warning("No prices found. Press *Run pipeline* or check the config paths.")
    else:
        filt = filter_table(working_prices, role_filter, team_filter, search_text, min_pv)
        if "FVM" in filt.columns:
            filt["Delta_vs_FVM"] = filt["StopPrice"] - filt["FVM"].fillna(0.0)
        for c in ["StopPrice","FVM","Delta_vs_FVM","Pts_hat","VORP","Pv_hat","Mv_hat"]:
            if c in filt.columns:
                filt[c] = filt[c].astype(float).round(2)
        show_cols = [c for c in ["name","team","Role","StopPrice","VORP","Pts_hat","FVM","Delta_vs_FVM","Pv_hat","Mv_hat","AFCON_RISK","START_TIER"] if c in filt.columns]
        st.dataframe(filt[show_cols], use_container_width=True, hide_index=True)
        st.caption(f"Rows: {len(filt)} (from {len(working_prices)} total)")

        # Downloads / Save
        st.download_button("Download filtered prices", data=df_to_csv_bytes(filt[show_cols]), file_name="stop_prices_filtered.csv", mime="text/csv")
        if st.button("💾 Save full adjusted prices to outputs/stop_prices_adjusted.csv"):
            Path("outputs").mkdir(parents=True, exist_ok=True)
            cols_save = ["name","team","role","Role","StopPrice","VORP","Pts_hat","FVM","Pv_hat","Mv_hat","AFCON_RISK","START_TIER"]
            cols_save = [c for c in cols_save if c in working_prices.columns or c in ["StopPrice","Role"]]
            to_save = working_prices.copy()
            if "Price_final" in to_save.columns and "StopPrice" not in to_save.columns:
                to_save["StopPrice"] = to_save["Price_final"]
            to_save = to_save[[c for c in cols_save if c in to_save.columns]]
            to_save.to_csv("outputs/stop_prices_adjusted.csv", index=False)
            st.success("Saved: outputs/stop_prices_adjusted.csv")
            st.download_button("Download saved file", data=df_to_csv_bytes(to_save), file_name="stop_prices_adjusted.csv", mime="text/csv")

        # Add to watchlist (manual)
        st.markdown("**Add to watchlist**")
        choices = filt["name"].unique().tolist()
        pick = st.multiselect("Select players to star", options=choices, key="wl_add")
        if st.button("➕ Add selected"):
            add_to_watchlist(pick)
            st.success(f"Added {len(pick)} players to watchlist.")

        # Quick star top N per role
        st.markdown("**Quick star: Top N per role (from current filter/view)**")
        colN1, colN2, colN3, colN4 = st.columns(4)
        with colN1:
            n_gk = st.number_input("Top GK", min_value=0, value=0, step=1)
        with colN2:
            n_def = st.number_input("Top DEF", min_value=0, value=0, step=1)
        with colN3:
            n_mid = st.number_input("Top MID", min_value=0, value=0, step=1)
        with colN4:
            n_fwd = st.number_input("Top FWD", min_value=0, value=0, step=1)
        if st.button("⭐ Star top picks"):
            new_stars = []
            for r, n in [("P", n_gk), ("D", n_def), ("C", n_mid), ("A", n_fwd)]:
                if n > 0:
                    topn = filt[filt["role"]==r].sort_values("StopPrice", ascending=False).head(int(n))["name"].tolist()
                    new_stars.extend(topn)
            add_to_watchlist(new_stars)
            st.success(f"Starred {len(new_stars)} players across roles.")

        # Summary chart
        if "StopPrice" in filt.columns:
            role_sum = filt.groupby("Role")["StopPrice"].sum().reindex(["GK","DEF","MID","FWD"]).fillna(0)
            st.bar_chart(role_sum)

with tab2:
    st.subheader("Full projections (base file)")
    if base.empty:
        st.warning("No projections found. Press *Run pipeline* or check the config paths.")
    else:
        default_cols = ["name","team","Role","Pts_hat","VORP","Pv_hat","Mv_hat","G_hat","A_hat","AFCON_RISK","START_TIER","PK_goals_hat","SP_assists_hat","Mod_share_hat"]
        cols = st.multiselect("Columns to show", options=list(base.columns), default=[c for c in default_cols if c in base.columns])
        f2 = filter_table(base, role_filter, team_filter, search_text, min_pv)
        if cols: f2 = f2[cols]
        st.dataframe(f2, use_container_width=True, hide_index=True)
        st.caption(f"Rows: {len(f2)} (from {len(base)} total)")
        st.download_button("Download filtered projections", data=df_to_csv_bytes(f2), file_name="projections_filtered.csv", mime="text/csv")

with tab3:
    st.subheader("Budget planner")
    if working_prices.empty:
        st.warning("No prices available.")
    else:
        c1, c2, c3, c4, c5 = st.columns([1,1,1,1,2])
        with c1:
            budget = st.number_input("Room budget", value=10000, step=100)
        with c2:
            bGK = st.number_input("GK #", value=3, step=1)
        with c3:
            bDEF = st.number_input("DEF #", value=8, step=1)
        with c4:
            bMID = st.number_input("MID #", value=8, step=1)
        with c5:
            bFWD = st.number_input("FWD #", value=6, step=1)

        def compute_budget_plan(df, budget, target_counts):
            plan = []
            total_weight = 0.0
            weights = {}
            for r, k in target_counts.items():
                pool = df[df["role"]==r].sort_values("StopPrice", ascending=False).head(int(k))
                w = pool["StopPrice"].sum()
                weights[r] = w
                total_weight += w
            for r, w in weights.items():
                share = (w / total_weight) if total_weight > 0 else 0.0
                plan.append({"Role": role_label(r), "Target #": int(target_counts[r]), "Budget": budget*share})
            return pd.DataFrame(plan)

        plan_df = compute_budget_plan(working_prices, budget, {"P":bGK,"D":bDEF,"C":bMID,"A":bFWD})
        plan_df["Budget"] = plan_df["Budget"].round(2)
        st.dataframe(plan_df, use_container_width=True, hide_index=True)
        st.download_button("Download budget plan", data=df_to_csv_bytes(plan_df), file_name="budget_plan.csv", mime="text/csv")
        st.markdown("**Tip:** Use the role multipliers or What-if tilts to simulate market heat and convictions.")

with tab4:
    st.subheader("Targets by Round")
    if working_prices.empty:
        st.warning("No prices available.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            budget = st.number_input("Room budget", value=10000, step=100, key="round_budget")
        with c2:
            anchor_mult = st.number_input("Round 1 ceiling ×", value=1.15, step=0.05, format="%.2f")
            starter_mult = st.number_input("Round 2 ceiling ×", value=1.05, step=0.05, format="%.2f")
        with c3:
            bench_mult = st.number_input("Round 3+ ceiling ×", value=0.95, step=0.05, format="%.2f")
        # desired counts
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            bGK = st.number_input("GK target #", value=3, step=1, key="round_gk")
        with c2:
            bDEF = st.number_input("DEF target #", value=8, step=1, key="round_def")
        with c3:
            bMID = st.number_input("MID target #", value=8, step=1, key="round_mid")
        with c4:
            bFWD = st.number_input("FWD target #", value=6, step=1, key="round_fwd")

        # Tier shares (percentages of each role's target count)
        st.markdown("**Tier split per role (as % of targets)**")
        t1, t2, t3 = st.columns(3)
        with t1:
            p_anchor = st.slider("Round 1 share %", 0, 100, 20)
        with t2:
            p_starter = st.slider("Round 2 share %", 0, 100, 40)
        with t3:
            p_bench = 100 - p_anchor - p_starter
            st.metric("Round 3+ share %", p_bench)

        target_counts = {"P":bGK,"D":bDEF,"C":bMID,"A":bFWD}

        def role_budgets(df, budget, target_counts):
            weights = {}
            total_weight=0.0
            for r,k in target_counts.items():
                pool = df[df["role"]==r].sort_values("StopPrice", ascending=False).head(int(k))
                w = pool["StopPrice"].sum()
                weights[r]=w; total_weight+=w
            budgets = {r: (budget*(w/total_weight) if total_weight>0 else 0.0) for r,w in weights.items()}
            return budgets

        rb = role_budgets(working_prices, budget, target_counts)

        rows=[]
        for r, k in target_counts.items():
            k = int(k)
            if k<=0: continue
            pool = working_prices[working_prices["role"]==r].sort_values("StopPrice", ascending=False).head(k).copy()
            n_anchor = max(1, int(round(p_anchor/100.0*k))) if k>0 else 0
            n_starter = max(0, int(round(p_starter/100.0*k)))
            if n_anchor + n_starter > k:
                n_starter = max(0, k - n_anchor)
            n_bench = max(0, k - n_anchor - n_starter)

            pool["Round"] = ["R1"]*n_anchor + ["R2"]*n_starter + ["R3+"]*n_bench
            pool["CeilingMult"] = pool["Round"].map({"R1":anchor_mult,"R2":starter_mult,"R3+":bench_mult})
            pool["Ceiling"] = pool["StopPrice"] * pool["CeilingMult"]

            # normalize to fit role budget
            role_sum = pool["Ceiling"].sum()
            role_budget = rb.get(r, 0.0)
            if role_sum > 0 and role_budget > 0:
                scale = role_budget / role_sum
                pool["Ceiling"] = pool["Ceiling"] * scale

            pool["RoleLabel"] = pool["role"].map(role_label)
            rows.append(pool[["RoleLabel","name","team","Round","StopPrice","Ceiling"]])

        if rows:
            targets = pd.concat(rows, ignore_index=True)
            targets["StopPrice"] = targets["StopPrice"].round(2)
            targets["Ceiling"] = targets["Ceiling"].round(2)
            targets = targets.rename(columns={"RoleLabel":"Role"})
            st.dataframe(targets, use_container_width=True, hide_index=True)
            st.download_button("Download targets by round", data=df_to_csv_bytes(targets), file_name="targets_by_round.csv", mime="text/csv")

            if st.button("💾 Save targets to outputs/targets_by_round.csv"):
                Path("outputs").mkdir(parents=True, exist_ok=True)
                targets.to_csv("outputs/targets_by_round.csv", index=False)
                st.success("Saved: outputs/targets_by_round.csv")
        else:
            st.info("Adjust targets or ensure prices are loaded.")

with tab5:
    st.subheader("⭐ Watchlist")
    if not st.session_state.watchlist:
        st.info("Your watchlist is empty. Add players from the Stop Prices tab.")
    else:
        wl = pd.DataFrame({"name": sorted(list(st.session_state.watchlist))})
        join = working_prices.merge(wl, on="name", how="inner")
        keep = [c for c in ["name","team","Role","StopPrice","VORP","Pts_hat","Pv_hat","Mv_hat","AFCON_RISK","START_TIER"] if c in join.columns]
        st.dataframe(join[keep], use_container_width=True, hide_index=True)
        st.download_button("Download watchlist", data=df_to_csv_bytes(join[keep]), file_name="watchlist.csv", mime="text/csv")
        rem = st.multiselect("Remove players", options=sorted(list(st.session_state.watchlist)))
        if st.button("🗑️ Remove selected"):
            remove_from_watchlist(rem)
            st.success(f"Removed {len(rem)} players.")

with tab6:
    st.markdown("""
**About**  
- Reads `config_csv.yaml` and displays outputs from:
  - `outputs/stop_prices.csv`
  - `outputs/projections.csv`
- **Run pipeline** triggers `build_model_csv.py`.
- **What-if tilts:** upload a CSV with columns:
  - `name,team,delta_Mv,delta_G,delta_A,cap_Pv,price_mult`
  - Deltas add to existing projections; `cap_Pv` caps expected appearances; `price_mult` scales the displayed StopPrice.
- **Watchlist:** star players, export or prune as you go.
- **Save adjusted prices:** write the currently computed StopPrice table to `outputs/stop_prices_adjusted.csv`.
- **Targets by Round:** compute per-role ceilings by round (R1/R2/R3+), normalized to fit role budgets.
""")
