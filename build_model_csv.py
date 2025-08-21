
import re, math, yaml, pandas as pd, numpy as np
from pathlib import Path
import unicodedata

# ----------------------------- utils -----------------------------

def ascii_fold(s: str) -> str:
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")

def norm_name(s: str) -> str:
    s = ascii_fold(str(s)).lower()
    s = re.sub(r"[^a-z0-9\s/,-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def role_letter(x):
    return str(x).strip().upper()[0]

def recency_blend(values, weights):
    vals = [v for v in values if pd.notna(v)]
    if not vals: return np.nan
    wts = weights[:len(vals)]
    wts = np.array(wts) / np.sum(wts)
    return float(np.dot(vals, wts))

def ensure_cols(df, cols):
    for c in cols:
        if c not in df.columns: df[c] = 0
    return df

def load_aliases(path="alias.yaml"):
    p = Path(path)
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text()) or {}
    return {norm_name(k): norm_name(v) for k, v in data.items() if v}


def apply_aliases(df, aliases):
    if not aliases:
        return df
    df['player_norm'] = df['player_norm'].map(lambda s: aliases.get(s, s))
    return df

# ------------------------ ingest: roster -------------------------

def load_roster(path):
    df = pd.read_excel(path, sheet_name="Tutti", header=1)
    colmap = {"Nome":"player", "Squadra":"team", "R":"role", "FVM":"FVM"}
    df = df.rename(columns={k:v for k,v in colmap.items() if k in df.columns})
    df['role'] = df['role'].map(role_letter)
    df['player_norm'] = df['player'].map(norm_name).str.replace("/", " ", regex=False)
    df['team_norm'] = df['team'].map(norm_name)
    return df[['player','player_norm','team','team_norm','role','FVM']].copy()

# ------------------------ ingest: history ------------------------

HIST_COLMAP = {
    'Id':'Id','R':'role','Nome':'player','Squadra':'team','Pv':'Pv',
    'Mv':'Mv','Fm':'Fm','Gf':'G','Ass':'A','Amm':'YC','Esp':'RC','Au':'OG',
    'Gs':'GC','Rc':'PK_att','R+':'PK_scored','R-':'PK_missed','Rp':'PK_saved'
}

def load_history(paths):
    frames=[]
    for p in paths:
        df = pd.read_excel(p, sheet_name="Tutti", header=1)
        season = Path(p).stem
        df = df.rename(columns={k:v for k,v in HIST_COLMAP.items() if k in df.columns})
        df['season'] = season
        df['role'] = df['role'].map(role_letter)
        df['player_norm'] = df['player'].map(norm_name).str.replace("/", " ", regex=False)
        df['team_norm'] = df['team'].map(norm_name)
        frames.append(df)
    hist = pd.concat(frames, ignore_index=True)
    need = ['Pv','Mv','Fm','G','A','YC','RC','OG','GC','PK_att','PK_scored','PK_missed','PK_saved']
    hist = ensure_cols(hist, need)
    return hist

# ------------------------ priors (role×team means) ------------------------

def build_priors(hist):
    grp = hist.groupby(['role','team_norm'])
    pri = grp.agg({
        'Mv':'mean','G':'sum','A':'sum','YC':'sum','RC':'sum','OG':'sum',
        'Pv':'sum','GC':'sum','PK_att':'sum','PK_scored':'sum','PK_saved':'sum'
    }).reset_index()
    for col in ['G','A','YC','RC','OG','GC','PK_att','PK_scored','PK_saved']:
        pri[col+'_perPv'] = pri[col]/pri['Pv'].clip(lower=1)
    pri = pri.rename(columns={'Mv':'Mv_mean'})
    return pri

def prior_lookup(priors, role, team_norm, col, fallback_role=True):
    sub = priors[(priors.role==role)&(priors.team_norm==team_norm)]
    if len(sub)==0 and fallback_role:
        sub = priors[priors.role==role]
    if len(sub)==0:
        return np.nan
    if col=='Mv':
        return float(sub['Mv_mean'].mean())
    if col.endswith('_perPv'):
        return float(sub[col].mean())
    return float('nan')

# ------------------------ CSV ingestion (new schemas) ------------------------

def load_rigoristi_csv(path, aliases):
    # name, team, priority (1..3)
    df = pd.read_csv(path)
    df['team_norm'] = df['team'].map(norm_name)
    df['player_norm'] = df['name'].map(norm_name).str.replace("/", " ", regex=False)
    df = apply_aliases(df, aliases)
    # keep only valid priority
    df = df[df['priority'].isin([1,2,3])].copy()
    return df[['team_norm','player_norm','priority']]

def load_setpieces_csv(path, aliases):
    # name, team, role ('freekick' or 'corner')
    df = pd.read_csv(path)
    df['team_norm'] = df['team'].map(norm_name)
    df['player_norm'] = df['name'].map(norm_name).str.replace("/", " ", regex=False)
    df = apply_aliases(df, aliases)
    df['role_sp'] = df['role'].astype(str).str.lower().str.strip()
    df = df[df['role_sp'].isin(['freekick','corner'])].drop_duplicates()
    # compute equal shares per team-role (no primary/secondary info in schema)
    shares = []
    for (t, r), g in df.groupby(['team_norm','role_sp']):
        n = len(g)
        sh = 1.0 / n if n>0 else 0.0
        for _,row in g.iterrows():
            shares.append({'team_norm': t, 'player_norm': row['player_norm'], 'category': r, 'share': sh})
    return pd.DataFrame(shares)

def load_formations_starters_csv(path, aliases):
    # Team Name, Player Name, Position, Ballotaggio
    df = pd.read_csv(path)
    df = df.rename(columns={'Team Name':'team','Player Name':'player','Position':'role','Ballotaggio':'ballotaggio'})
    df['team_norm'] = df['team'].map(norm_name)
    df['player_norm'] = df['player'].map(norm_name).str.replace("/", " ", regex=False)
    df = apply_aliases(df, aliases)
    df['role'] = df['role'].astype(str).str.upper().str[0]
    df['ballotaggio'] = df['ballotaggio'].astype(str).str.strip().str.lower().isin(['yes','y','true','1'])
    return df[['team_norm','player_norm','ballotaggio']]

def load_afcon_csv(path, aliases):
    # name, team
    df = pd.read_csv(path)
    df['team_norm'] = df['team'].map(norm_name)
    df['player_norm'] = df['name'].map(norm_name).str.replace("/", " ", regex=False)
    df = apply_aliases(df, aliases)
    return df[['team_norm','player_norm']].drop_duplicates()

# ------------------------ appearances from formations + AFCON ------------------------

def project_pv(roster, starters_csv, afcon_csv, params):
    sp = params['start_probs']; miss = params['afcon_default_miss']
    merged = roster.merge(starters_csv, on=['team_norm','player_norm'], how='left')
    def p_start_row(row):
        if pd.isna(row['ballotaggio']):
            return sp['backup']
        return sp['battle'] if bool(row['ballotaggio']) else sp['starter']
    merged['p_start'] = merged.apply(p_start_row, axis=1)
    merged['Pv_hat'] = 38 * merged['p_start']
    if not afcon_csv.empty:
        afcon_set = set(map(tuple, afcon_csv[['team_norm','player_norm']].values))
        merged['AFCON_RISK'] = merged.apply(lambda r: int((r['team_norm'], r['player_norm']) in afcon_set), axis=1)
        merged.loc[merged['AFCON_RISK']==1, 'Pv_hat'] = (merged.loc[merged['AFCON_RISK']==1, 'Pv_hat'] - miss).clip(lower=0)
    else:
        merged['AFCON_RISK'] = 0
    merged['START_TIER'] = merged['p_start'].map(lambda p: 'starter' if p>=sp['starter'] else ('battle' if p>=sp['battle'] else 'backup'))
    return merged

# ------------------------ player-level projections ------------------------

def recency_player_stat(hist, player_norm, team_norm, role, col, weights, priors):
    h = hist[(hist.player_norm==player_norm)&(hist.team_norm==team_norm)&(hist.role==role)].copy()
    if h.empty:
        if col=='Mv':
            return prior_lookup(priors, role, team_norm, 'Mv')
        else:
            return prior_lookup(priors, role, team_norm, f"{col}_perPv")
    h = h.groupby('season').agg({col:'sum','Pv':'sum','Mv':'mean'}).sort_index(ascending=False).reset_index()
    if col=='Mv':
        vals = h['Mv'].tolist()
        return recency_blend(vals, weights[:len(vals)])
    else:
        rates = (h[col] / h['Pv'].clip(lower=1)).tolist()
        r_base = recency_blend(rates, weights[:len(rates)])
        prior = prior_lookup(priors, role, team_norm, f"{col}_perPv")
        n = float(h['Pv'].sum())
        lam = min(1.0, n/20.0)
        if pd.isna(r_base): r_base = prior
        return lam*r_base + (1-lam)*prior

def add_mv_and_events(roster, hist, priors, params):
    W = params['recency_weights']
    cols = ['G','A','YC','RC','OG']
    mvh=[]; rates={c:[] for c in cols}; gc_rate=[]; pksave_rate=[]
    for _,r in roster.iterrows():
        role=r['role']; team=r['team_norm']; nm=r['player_norm']
        mvh.append(recency_player_stat(hist, nm, team, role, 'Mv', W, priors))
        for c in cols:
            rates[c].append(recency_player_stat(hist, nm, team, role, c, W, priors))
        if role=='P':
            gc_rate.append(recency_player_stat(hist, nm, team, role, 'GC', W, priors))
            pksave_rate.append(recency_player_stat(hist, nm, team, role, 'PK_saved', W, priors))
        else:
            gc_rate.append(np.nan); pksave_rate.append(np.nan)
    out = roster.copy()
    out['Mv_hat']=mvh
    for c in cols: out[c+'_perPv_hat']=rates[c]
    out['GC_perPv_hat']=gc_rate
    out['PKsave_perPv_hat']=pksave_rate
    for c in cols: out[c+'_hat']=out[c+'_perPv_hat']*out['Pv_hat']
    out['GC_hat']=out['GC_perPv_hat']*out['Pv_hat']
    out['PKsave_hat']=out['PKsave_perPv_hat']*out['Pv_hat']
    return out

# ------------------------ apply PK & set pieces via new CSVs ------------------------

def apply_pk(roster, hist, rigoristi_csv, params):
    # team PK volume EV
    W = params['recency_weights']
    team_rc = (hist.groupby(['team_norm','season'])['PK_att'].sum()
               .reset_index().sort_values('season', ascending=False))
    team_rc_ev = team_rc.groupby('team_norm')['PK_att'].apply(lambda s: recency_blend(s.tolist(), W[:len(s)])).to_dict()
    # conversion
    tot_rc = hist['PK_att'].sum(); tot_scored = hist['PK_scored'].sum()
    p_conv = params['pk_conv_default']
    if tot_rc>0: p_conv = float(tot_scored/tot_rc)
    # shares from priority
    pr_shares = {1: params['pk_shares'][0], 2: params['pk_shares'][1], 3: params['pk_shares'][2]}
    share_map = {}  # (team_norm, player_norm) -> share
    for _,r in rigoristi_csv.iterrows():
        share_map[(r['team_norm'], r['player_norm'])] = pr_shares.get(int(r['priority']), 0.0)
    PK_taken=[]; PK_goals=[]
    for _,r in roster.iterrows():
        team = r['team_norm']; nm = r['player_norm']
        team_pk = team_rc_ev.get(team, 0.0)
        share = share_map.get((team, nm), 0.0)
        pk_taken = team_pk * share * (r['Pv_hat']/38.0)
        PK_taken.append(pk_taken)
        PK_goals.append(pk_taken * p_conv)
    out = roster.copy()
    out['PK_taken_hat']=PK_taken
    out['PK_goals_hat']=PK_goals
    out['G_hat'] = out['G_hat'] + np.maximum(0, out['PK_goals_hat'] - 0.0)
    return out

def apply_setpieces(roster, hist, sp_long, params):
    # team assists EV * fraction -> set-piece assist pool
    W = params['recency_weights']; frac = params['sp_assist_fraction']
    team_ass = (hist.groupby(['team_norm','season'])['A'].sum()
                .reset_index().sort_values('season', ascending=False))
    team_ass_ev = team_ass.groupby('team_norm')['A'].apply(lambda s: recency_blend(s.tolist(), W[:len(s)])).to_dict()
    # share map (equal shares within each team-role)
    share_map = {}  # (team_norm, player_norm, 'freekick'|'corner') -> share
    for _,r in sp_long.iterrows():
        share_map[(r['team_norm'], r['player_norm'], r['category'])] = float(r['share'])
    sp_extra=[]
    for _,r in roster.iterrows():
        team=r['team_norm']; nm=r['player_norm']
        team_sp_ass = team_ass_ev.get(team, 0.0)*frac
        share_fk = share_map.get((team, nm, 'freekick'), 0.0)
        share_ck = share_map.get((team, nm, 'corner'), 0.0)
        share = max(share_fk, share_ck)  # conservative to avoid double-counting within the same pool
        sp_a = team_sp_ass * share * (r['Pv_hat']/38.0)
        sp_extra.append(sp_a)
    out = roster.copy()
    out['SP_assists_hat']=sp_extra
    out['A_hat'] = out['A_hat'] + out['SP_assists_hat']
    return out

# -------------------- Defense modifier --------------------

def modifier_ev_per_week(mu_team, bands):
    for b in bands:
        if mu_team <= b['max']: return b['ev']
    return bands[-1]['ev']

def apply_modifier(roster, params):
    bands = params['modifier_bands']; p4 = params['four_back_freq']
    out = roster.copy()
    out['Mod_share_hat'] = 0.0
    for team, g in out.groupby('team_norm'):
        gk = g[g.role=='P'].sort_values('Pv_hat', ascending=False).head(1)
        defs = g[g.role=='D'].sort_values('Pv_hat', ascending=False).head(3)
        if len(gk)==0 or len(defs)<3: continue
        mu_team = pd.concat([gk[['Mv_hat']], defs[['Mv_hat']]])['Mv_hat'].mean()
        ev_week = modifier_ev_per_week(mu_team, bands)
        season_team_bonus = 38 * p4 * ev_week
        alloc = [(gk.index, 0.40), (defs.index[:1], 0.20), (defs.index[1:2], 0.20), (defs.index[2:3], 0.20)]
        for idxs, w in alloc:
            out.loc[idxs, 'Mod_share_hat'] += season_team_bonus * w
    return out

# -------------------- Points, Replacement, VORP, Pricing --------------------

def compute_points(roster):
    out = roster.copy()
    sk = out.role.isin(['D','C','A'])
    out.loc[sk, 'Pts_hat'] = (
        out.loc[sk,'Pv_hat']*out.loc[sk,'Mv_hat']
        + 3*out.loc[sk,'G_hat']
        + 1*out.loc[sk,'A_hat']
        - 0.5*out.loc[sk,'YC_hat']
        - 2.0*out.loc[sk,'RC_hat']
        - 2.0*out.loc[sk,'OG_hat']
        + out.loc[sk,'Mod_share_hat']
    )
    gk = out.role=='P'
    cs = np.exp(-out.loc[gk,'GC_perPv_hat'].fillna(0.0)) * out.loc[gk,'Pv_hat']
    out.loc[gk,'CS_hat']=cs
    out.loc[gk, 'Pts_hat'] = (
        out.loc[gk,'Pv_hat']*out.loc[gk,'Mv_hat']
        - 1.0*out.loc[gk,'GC_hat']
        + 2.0*out.loc[gk,'CS_hat']
        + 3.0*out.loc[gk,'PKsave_hat']
        - 0.5*out.loc[gk,'YC_hat']
        - 2.0*out.loc[gk,'RC_hat']
        - 2.0*out.loc[gk,'OG_hat']
        + out.loc[gk,'Mod_share_hat']
    )
    return out

def replacement_points(df, params):
    cuts = params['replacement_cut']
    repl={}
    for role, N in cuts.items():
        pool = df[df.role==role].sort_values('Pts_hat', ascending=False).reset_index(drop=True)
        if len(pool)>=N:
            repl[role] = float(pool.loc[N-1, 'Pts_hat'])
        else:
            repl[role] = float(pool['Pts_hat'].min()) if len(pool)>0 else 0.0
    return repl

def compute_vorp_and_price(df, repl, roster_fvm, params):
    out = df.copy()
    out['ReplacementPts'] = out['role'].map(repl)
    out['VORP'] = out['Pts_hat'] - out['ReplacementPts']
    out.loc[out['VORP']<0, 'VORP']=0.0

    total_vorp = out['VORP'].sum()
    k = 10000.0 / total_vorp if total_vorp>0 else 0.0
    model_price = out['VORP']*k

    fvm = roster_fvm['FVM'].fillna(0).astype(float)
    fvm_scaled = (fvm / fvm.sum()) * 10000.0 if fvm.sum()>0 else fvm
    w = params['pricing']['blend_fvm_weight']
    blended = (1-w)*fvm_scaled + w*model_price

    bias = params['pricing']['room_bias']
    biased = blended * out['role'].map(bias).astype(float)

    alpha = params['pricing']['star_exponent']
    biased_norm = biased / biased.max() if biased.max()>0 else biased
    premium = np.power(biased_norm, alpha) * biased.sum() / (biased_norm.sum() if biased_norm.sum()>0 else 1)

    final = premium * (10000.0 / premium.sum()) if premium.sum()>0 else premium

    out['ModelPrice_raw']=model_price
    out['Price_final']=final
    return out

# ----------------------------- main -----------------------------

def main():
    cfg = yaml.safe_load(Path("config_csv.yaml").read_text())

    aliases = load_aliases("alias.yaml")
    roster = load_roster(cfg['paths']['roster_xlsx'])
    hist = load_history(cfg['paths']['history_xlsx'])
    pri = build_priors(hist)

    # Load CSVs (new schemas)
    rigoristi = load_rigoristi_csv(cfg['paths']['csv_rigoristi'], aliases)
    setpieces_long = load_setpieces_csv(cfg['paths']['csv_setpieces'], aliases)
    starters = load_formations_starters_csv(cfg['paths']['csv_formations_starters'], aliases)
    afcon = load_afcon_csv(cfg['paths']['csv_afcon'], aliases)

    # Appearances from formations + AFCON
    proj = project_pv(roster, starters, afcon, cfg['params'])

    # Mv + events
    proj = add_mv_and_events(proj, hist, pri, cfg['params'])

    # Add PK & Set pieces
    proj = apply_pk(proj, hist, rigoristi, cfg['params'])
    proj = apply_setpieces(proj, hist, setpieces_long, cfg['params'])

    # Defense modifier
    proj = apply_modifier(proj, cfg['params'])

    # Points
    proj = compute_points(proj)

    # Replacement & VORP & Prices
    repl = replacement_points(proj, cfg['params'])
    prices = compute_vorp_and_price(proj, repl, roster[['FVM']], cfg['params'])

    # Flags (light)
    prices['LOW_SAMPLE'] = prices['Pv_hat'] < 10
    prices['NEW_SERIEA'] = 0
    prices['PROMOTED_TEAM'] = 0

    # Save
    Path("outputs").mkdir(parents=True, exist_ok=True)
    prices.to_csv(cfg['output']['projections_csv'], index=False)
    cols = ['player','team','role','Pts_hat','VORP','Price_final','FVM',
            'Pv_hat','Mv_hat','G_hat','A_hat','YC_hat','RC_hat','OG_hat','AFCON_RISK','START_TIER','Mod_share_hat']
    prices[cols].sort_values(['role','Price_final'], ascending=[True,False]).to_csv(cfg['output']['prices_csv'], index=False)

    print("Done. Projections →", cfg['output']['projections_csv'])
    print("Stop prices →", cfg['output']['prices_csv'])

if __name__ == "__main__":
    main()
