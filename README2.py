
# Fantacalcio Auction Model (Serie A 2025/26)

A lean-but-serious projection and pricing engine for **Fantacalcio (classic)** with **modificatore difesa** and a **+2 GK clean-sheet bonus**. It ingests your roster + historical stats + four auxiliary CSVs (penalty takers, set-piece takers, expected starters/ballottaggi, AFCON risks), then produces:
- **Season projections** per player (appearances, mediavoto, goals, assists, cards, GK stats, modifier share).
- **VORP** (Value Over Replacement Player) on a **single-tier** baseline.
- **Auction stop-prices** calibrated to a **10,000-credit room**, with bias knobs to mirror room tendencies (overpaying GKs/FWDs, underpaying MIDs).

> Status: internal-use ready. One run script (`build_model_csv.py`) + one YAML config (`config_csv.yaml`).

## Project Layout

.
├── app.py                         # Streamlit UI
├── build_model_csv.py             # main modeling script
├── config_csv.yaml                # config & knobs (paths/weights)
├── README.md
├── requirements.txt
├── inputs/                        # you maintain these CSVs / Excels
│   ├── Quotazioni_Fantacalcio_Stagione_2025_26.xlsx
│   ├── Statistiche_Fantacalcio_Stagione_2024_25.xlsx   # add older seasons too
│   ├── rigoristi.csv              # name,team,priority
│   ├── setpieces.csv              # name,team,role (freekick|corner)
│   ├── formations_starters.csv    # Team Name,Player Name,Position,Ballotaggio (Yes/No)
│   └── afcon.csv                  # name,team
└── outputs/
├── projections.csv
└── stop_prices.csv

## Quick Start

```bash
pip install -r requirements.txt
python build_model_csv.py
streamlit run app.py

	•	Config: edit config_csv.yaml with your local paths (or switch to relative, e.g., ./inputs/...).
	•	Outputs: check ./outputs/stop_prices.csv first — this is your auction sheet.

Inputs & Schemas

Roster (Quotazioni … 2025/26.xlsx) — sheet Tutti (header row index 1)
	•	Nome (name), Squadra (team), R (role in {P,D,C,A}), FVM (lega price).

Historical Stats (Statistiche … .xlsx) — add as many seasons as you like
	•	Pv (appearances with rating), Mv (media voto), Fm (ignored),
Gf=G, Ass=A, Amm=YC, Esp=RC, Au=OG, Gs=GC (GK),
Rc=PK_att, R+=PK_scored, R-=PK_missed, Rp=PK_saved.

Aux CSVs
	•	rigoristi.csv: name,team,priority (1→primary, 2, 3). Shares default to [0.7,0.2,0.1].
	•	setpieces.csv: name,team,role with role∈{freekick,corner}. Equal shares within team-role.
	•	formations_starters.csv: Team Name,Player Name,Position,Ballotaggio (Yes/No). Drives Pv̂.
	•	afcon.csv: name,team. Flags AFCON and subtracts default 3 matches (configurable).

Configuration (config_csv.yaml)

paths:
  roster_xlsx: "./inputs/Quotazioni_Fantacalcio_Stagione_2025_26.xlsx"
  history_xlsx: ["./inputs/Statistiche_Fantacalcio_Stagione_2024_25.xlsx"]
  csv_rigoristi: "./inputs/rigoristi.csv"
  csv_setpieces: "./inputs/setpieces.csv"
  csv_formations_starters: "./inputs/formations_starters.csv"
  csv_afcon: "./inputs/afcon.csv"

params:
  recency_weights: [0.6, 0.3, 0.1]
  start_probs: {starter: 0.85, battle: 0.65, battle3: 0.55, backup: 0.35}
  four_back_freq: 0.85
  afcon_default_miss: 3
  pk_shares: [0.7, 0.2, 0.1]
  pk_conv_default: 0.80
  sp_assist_fraction: 0.33
  modifier_bands:
    - {max: 6.25, ev: 0.5}
    - {max: 6.50, ev: 1.0}
    - {max: 6.75, ev: 1.5}
    - {max: 7.00, ev: 2.0}
    - {max: 7.25, ev: 2.8}
    - {max: 7.50, ev: 3.7}
    - {max: 10.0, ev: 4.5}
  replacement_cut: {P: 10, D: 43, C: 38, A: 28}
  pricing:
    blend_fvm_weight: 0.30
    room_bias: {P: 1.15, D: 1.00, C: 0.90, A: 1.12}
    star_exponent: 1.05

output:
  projections_csv: "./outputs/projections.csv"
  prices_csv: "./outputs/stop_prices.csv"

Tune after a dry run: start_probs, afcon_default_miss, sp_assist_fraction (0.28–0.40), replacement_cut, room_bias.

Full Pipeline
	1.	Pv̂ from starters/ballottaggi + AFCON adjustment.
	2.	Mv̂ from recency-blended historical Mv with shrinkage to role×team priors when samples are thin.
	3.	Events (G, A, YC, RC, OG, GK: GC, PK saves): per-appearance rates blended by recency; multiply by Pv̂.
	4.	Penalties: team PK attempts EV × player share (from rigoristi) × conversion. Adds to goals.
	5.	Set-pieces: team assist EV × global set-piece fraction × player FK/CK share. Adds to assists.
	6.	Modificatore difesa EV: estimate team weekly bonus from GK+top-3 DEF Mv̂; allocate 40/20/20/20 to GK/DEFs.
	7.	Fantasy Points by role.
	8.	Replacement per role (single-tier cut).
	9.	VORP = Points – Replacement.
	10.	Prices: scale to 10k, blend with FVM, apply room biases + star convexity, renormalize.

Streamlit App (app.py)

Tabs:
	•	🏷️ Stop Prices: filter/search; delta vs FVM; save adjusted prices; Quick Star Top N by role; watchlist add; bar chart by role.
	•	📊 Projections: pick columns, export.
	•	🧮 Planner: allocate budget per role using top-K valuation.
	•	🎯 Targets by Round: choose round multipliers and tier shares; normalize to role budgets; export/save.
	•	⭐ Watchlist: list, export, remove.
	•	ℹ️ About: docs & tips.

What-if tilts uploader (name,team,delta_Mv,delta_G,delta_A,cap_Pv,price_mult) recomputes points → replacement → prices without rerunning the pipeline.

Scoring Rules
	•	Skaters (D/C/A)
Pts = Pv̂×Mv̂ + 3×Ĝ + 1×Â – 0.5×YĈ – 2×RĈ – 2×OĜ + Mod_share
	•	Goalkeepers (P)
Pts = Pv̂×Mv̂ – 1×GĈ + 2×CŜ + 3×PKsavê – 0.5×YĈ – 2×RĈ – 2×OĜ + Mod_share
with CŜ ≈ Pv̂ × e^(−GC_perPv̂).

Outputs
	•	outputs/projections.csv: id/role/team + Pv̂, Mv̂, events, extras (PK/SP bumps, Mod share), points, VORP.
	•	outputs/stop_prices.csv: name,team,role, Pts_hat, VORP, Price_final, FVM (+ context).

Personal Tilts (optional)

tilts.csv example:

name,team,delta_Mv,delta_G,delta_A,cap_Pv,price_mult
Federico Dimarco,Inter,0.10,,0.5,,
Ademola Lookman,Atalanta,,2.0,,34,
Alex Meret,Napoli,,,,,0.9

Troubleshooting
	•	Paths: ensure config_csv.yaml paths are valid. Prefer relative paths (./inputs/...).
	•	Empty prices: check that build_model_csv.py produced outputs/stop_prices.csv.
	•	Weird PK/SP bumps: review rigoristi.csv / setpieces.csv.
	•	Role counts / replacement: tweak replacement_cut to your league size/benches.

Happy drafting! ⚽️💸
