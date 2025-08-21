# Fantacalcio Auction Model (Serie A 2025/26)

A lean-but-serious projection and pricing engine for **Fantacalcio (classic)** with **modificatore difesa** and a **+2 GK clean-sheet bonus**. It ingests your roster + historical stats + four auxiliary CSVs (penalty takers, set-piece takers, expected starters/ballottaggi, AFCON risks), then produces:
- **Season projections** per player (appearances, mediavoto, goals, assists, cards, GK stats, modifier share).
- **VORP** (Value Over Replacement Player) on a **single-tier** baseline.
- **Auction stop-prices** calibrated to a **10,000-credit room**, with bias knobs to mirror league tendencies (overpaying GKs/ATs, underpaying CMs, etc.).

> **Status:** Ready for internal use. Everything lives in a single run script (`build_model_csv.py`) with one YAML config (`config_csv.yaml`). Run it in two stages (`--stage projection` then `--stage points`) if you want to tweak projections by hand.


## Table of Contents

1. [Project Layout](#project-layout)  
2. [Quick Start](#quick-start)  
3. [Inputs & Schemas](#inputs--schemas)  
4. [Configuration (config_csv.yaml)](#configuration-config_csvyaml)  
5. [Full Pipeline (Step-by-Step)](#full-pipeline-step-by-step)  
6. [Scoring Rules Used](#scoring-rules-used)  
7. [Methodology & Formulas](#methodology--formulas)  
   - [Appearances (Pv̂)](#appearances-pv̂)  
   - [Mediavoto (Mv̂)](#mediavoto-mv̂)  
   - [Events (G, A, YC, RC, OG, GK-GC, PK saves)](#events-g-a-yc-rc-og-gk-gc-pk-saves)  
   - [Penalty Goals](#penalty-goals)  
   - [Set-Piece Assists](#set-piece-assists)  
   - [Defense Modifier EV](#defense-modifier-ev)  
   - [Fantasy Points](#fantasy-points)  
   - [VORP & Replacement Cuts](#vorp--replacement-cuts)  
   - [Stop-Prices / Auction Strategy](#stop-prices--auction-strategy)  
8. [Outputs](#outputs)  
9. [Personal Tilts (Optional)](#personal-tilts-optional)  
10. [Quality Checks & Diagnostics](#quality-checks--diagnostics)  
11. [Updating Data Next Week/Season](#updating-data-next-weekseason)  
12. [Troubleshooting](#troubleshooting)  
13. [FAQ](#faq)  


---

## Project Layout

```
/mnt/data/
  ├── build_model_csv.py          # main pipeline script (read-only here; copy locally as needed)
  ├── config_csv.yaml             # config & knobs
  ├── Quotazioni_Fantacalcio_Stagione_2025_26.xlsx   # roster (source of truth for players)
  ├── Statistiche_Fantacalcio_Stagione_2024_25.xlsx  # history (add more seasons if you want)
  └── parsed_v3/                  # auxiliary CSVs you maintain (extracted from PDFs or by hand)
      ├── rigoristi.csv           # name, team, priority (1..3)
      ├── setpieces.csv           # name, team, role ("freekick"|"corner")
      ├── afcon.csv               # name, team
  └── parsed_v2/
      └── formations_starters.csv # Team Name, Player Name, Position, Ballotaggio (Yes/No)
./outputs/
  ├── projections.csv             # full per-player projections
  └── stop_prices.csv             # role-sorted stop-price sheet for the auction
```

> You can maintain `parsed_v2/parsed_v3` by hand during mercato or re-run your extractor to refresh them. The modeling script **does not** parse PDFs anymore—it trusts these CSVs.


## Quick Start

```bash
# 1) Confirm file paths in config
cat /mnt/data/config_csv.yaml

# 2) Build projections
python /mnt/data/build_model_csv.py --stage projection

# 3) (optional) tweak `outputs/projections.csv`

# 4) Compute points & prices
python /mnt/data/build_model_csv.py --stage points

# 5) Inspect outputs
ls -lah ./outputs
```

- Look at **`outputs/stop_prices.csv`** first—that’s your auction sheet.
- If something looks off (e.g., all PKs zero for a team), check the associated CSV (e.g., `rigoristi.csv`) for that team.


## Inputs & Schemas

**1) Roster (Quotazioni … 2025/26.xlsx)** — sheet `Tutti` (header row at index 1)  
Used as player universe + FVM (lega’s expected auction price). Columns consumed:
- `Nome` → `name` (string)  
- `Squadra` → `team` (string)  
- `R` → `role` (one of **P/D/C/A**)  
- `FVM` (float)  

**2) Historical Stats (Statistiche … 2024/25.xlsx, add more years)**  
Columns consumed (Italian headers → internal names):
- `Pv` (Partite a voto) → appearances with a rating  
- `Mv` (Media Voto) → base media
- `Fm` (Fanta Media) → **ignored** for modeling
- `Gf`→`G`, `Ass`→`A`, `Amm`→`YC`, `Esp`→`RC`, `Au`→`OG`, `Gs`→`GC` (GK goals conceded),  
  `Rc`→`PK_att`, `R+`→`PK_scored`, `R-`→`PK_missed`, `Rp`→`PK_saved`.

**3) `parsed_v3/rigoristi.csv`**  
```
name,team,priority
Federico Chiesa,Juventus,1
...
```
- `priority` ∈ {1,2,3}. Shares map to `[0.7, 0.2, 0.1]` by default (configurable).

**4) `parsed_v3/setpieces.csv`**  
```
name,team,role
Hakan Calhanoglu,Inter,freekick
Federico Dimarco,Inter,corner
...
```
- Equal shares are assigned **within each team+role** because schema carries no explicit order.

**5) `parsed_v2/formations_starters.csv`**  
```
Team Name,Player Name,Position,Ballotaggio
Inter,Nicolo Barella,C,No
Lazio,Taty Castellanos,A,Yes
...
```
- Drives **Pv̂** using starter vs. ballottaggio vs. backup probabilities.

**6) `parsed_v3/afcon.csv`**  
```
name,team
Ademola Lookman,Atalanta
...
```
- Anyone listed gets **AFCON_RISK = 1** and loses a default **3 matches** (configurable).


## Configuration (`config_csv.yaml`)

Key sections:

```yaml
paths:
  roster_xlsx: ".../Quotazioni_...2025_26.xlsx"
  history_xlsx: [".../Statistiche_...2024_25.xlsx"]   # add older seasons here
  csv_rigoristi: ".../parsed_v3/rigoristi.csv"
  csv_setpieces: ".../parsed_v3/setpieces.csv"
  csv_formations_starters: ".../parsed_v2/formations_starters.csv"
  csv_afcon: ".../parsed_v3/afcon.csv"

params:
  recency_weights: [0.6, 0.3, 0.1]      # newest → oldest
  start_probs: {starter: 0.85, battle: 0.65, battle3: 0.55, backup: 0.35}
  four_back_freq: 0.85
  afcon_default_miss: 3

  pk_shares: [0.7, 0.2, 0.1]
  pk_conv_default: 0.80

  sp_assist_fraction: 0.33              # share of team assists from set pieces

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
```

**Knobs to revisit after a dry run:**
- `start_probs` (your league’s rotation reality), `afcon_default_miss`, `sp_assist_fraction` (try 0.28–0.40), `replacement_cut` (size of your league & benches), `room_bias` to reflect your room’s tendencies.


## Full Pipeline (Step-by-Step)

```
Roster + History
   │
   ├─► Starters (XI) & Ballottaggi (CSV)
   │      └─► Pv̂ from start_probs (starter/battle/backup)
   │
   ├─► AFCON (CSV)  ──► subtract afcon_default_miss appearances
   │
   ├─► Recency-blended Mv̂ & event rates (G, A, YC, RC, OG, GK: GC, PK saves)
   │      └─► shrinkage toward role×team priors when Pv is small
   │
   ├─► Penalties: team PK volume (recency) × player share × p_conv
   ├─► Set pieces: team assists (recency) × sp_assist_fraction × player share
   │
   ├─► Modificatore difesa EV: (GK+3 top DEF Mv̂) → band EV → allocate 40/20/20/20
   │
   ├─► Fantasy Points by role (per season)
   ├─► Replacement baselines (single-tier cut per role)
   ├─► VORP = Points – Replacement
   └─► Price curve: scale VORP to 10k, blend FVM, apply room biases, star premium, renormalize
```


## Scoring Rules Used

- **Skaters (D/C/A):**  
  `Pts = Pv̂×Mv̂ + 3×Ĝ + 1×Â – 0.5×YĈ – 2×RĈ – 2×OĜ + Mod_share`
- **Goalkeeper (P):**  
  `Pts = Pv̂×Mv̂ – 1×GĈ + 2×CŜ + 3×PKsavê – 0.5×YĈ – 2×RĈ – 2×OĜ + Mod_share`

Where:
- `CŜ` is expected clean sheets (Poisson: `CŜ ≈ Pv̂ × e^(–GC_perPv̂)`).  
- `Mod_share` is each player’s allocated slice from the team’s defensive modifier EV.


## Methodology & Formulas

### Appearances (Pv̂)
- From `formations_starters.csv`:
  - **starter:** `p_start = start_probs.starter` (default 0.85)  
  - **ballottaggio:** `p_start = start_probs.battle` (default 0.65)  
  - **not listed:** `p_start = start_probs.backup` (default 0.35)  
- Season appearances: `Pv̂ = 38 × p_start`.  
- **AFCON**: if in `afcon.csv`, subtract `afcon_default_miss` (default 3).

### Mediavoto (Mv̂)
- For a player: recency-blend mean `Mv` from history across seasons using `recency_weights` (newest→oldest).  
- If sample is thin, shrink toward **role×team** mean prior.

### Events (G, A, YC, RC, OG, GK: GC, PK saves)
- Convert each event to **per-appearance rates** by season, recency-blend rates.  
- If sample is thin, shrink toward **role×team** rate priors.  
- Multiply by `Pv̂` to get season totals.  
- GK-specific: `GC_perPv̂` and `PKsave_perPv̂` handled similarly.

### Penalty Goals
- Team PK attempts EV per season via recency blend: `PK_team_EV`.  
- Player PK share from `rigoristi.csv` priority (1→0.7, 2→0.2, 3→0.1 by default).  
- Conversion `p_conv` = league historical ratio (fallback `pk_conv_default`).  
- **Add** `PK_goalŝ = PK_team_EV × share × (Pv̂/38) × p_conv` to the player’s `Ĝ`.  
  (We **add** as a bump. If your historical baseline already includes PKs, this is still acceptable since it’s a blended-forward EV; you can reduce `pk_shares` if you fear double-counting.)

### Set-Piece Assists
- Team assists EV via recency blend, then take a global fraction `sp_assist_fraction` to represent set-piece share.  
- From `setpieces.csv`, compute **equal shares** within each team & role (`freekick`/`corner`).  
- For each player, we take the **max(FK_share, CK_share)** (conservative vs. double counting) and add:  
  `SP_Â = team_assists_EV × sp_assist_fraction × share × (Pv̂/38)` to `Â`.

### Defense Modifier EV
- For each team, estimate **μ_team = mean(Mv̂)** of **top GK + top 3 DEF** by `Pv̂`.  
- Map μ_team to **EV per week** via `modifier_bands` (piecewise linear table).  
- Season team bonus: `EV_team_season = 38 × four_back_freq × EV_per_week`.  
- Allocate: **40%** to GK, **20%** to each of the top-3 DEF. Others: 0.  
  (This approximates that you’ll typically field GK+3 DEF from the same team when chasing modifier.)

### Fantasy Points
See [Scoring Rules Used](#scoring-rules-used).

### VORP & Replacement Cuts
- For each role, sort players by projected points and take the **N-th** player as replacement (`replacement_cut` per role).  
- `VORP = Points – Replacement` (floored at 0).

### Stop-Prices / Auction Strategy
1. Scale VORP so the **sum of prices = 10,000 credits**.  
2. Blend with **FVM** (lega’s reference price) using `blend_fvm_weight` (default 0.30).  
3. Apply **room_bias** (e.g., `P:1.15`, `A:1.12`, `C:0.90`) to reflect league behavior.  
4. Apply a gentle **star premium** (`star_exponent`: 1.05) to slightly convexify the top end.  
5. Renormalize back to 10,000.  
6. Output **`stop_prices.csv`** by role, descending price.


## Outputs

**`outputs/projections.csv`** — exhaustive columns (highlights):  
- Identification: `name, team, role`  
- Availability & ratings: `Pv_hat, Mv_hat, START_TIER, AFCON_RISK`  
- Events: `G_hat, A_hat, YC_hat, RC_hat, OG_hat, (GK:) GC_hat, PKsave_hat, CS_hat`  
- Extras: `PK_taken_hat, PK_goals_hat, SP_assists_hat, Mod_share_hat`  
- Value: `Pts_hat, VORP, ModelPrice_raw, Price_final`

**`outputs/stop_prices.csv`** — auction sheet:  
- `name, team, role, Pts_hat, VORP, Price_final, FVM` (+ a few context fields).  
- Sorted by role then `Price_final` desc.


## Personal Tilts (Optional)

You can add a small adjustment file that reflects your convictions (e.g., raise Mv̂ for a player, bump goals, etc.). Suggested schema:

```
# tilts.csv
name,team,delta_Mv,delta_G,delta_A,cap_Pv
Federico Dimarco,Inter,0.10,0.0,1.0,
Ademola Lookman,Atalanta,,2.0,,34
```

- Empty cells mean “no change”.  
- `cap_Pv` (optional) caps appearances if you foresee injuries/rotations.  
**Integration:** easiest is to merge `tilts.csv` on `(name,team)` after `add_mv_and_events` and before points; then add deltas and apply caps. (I can wire this in on request.)


## Quality Checks & Diagnostics

- **Sanity of inputs:** Check the four CSVs per team after mercato.  
- **Coverage flags:** `LOW_SAMPLE` if `Pv̂ < 10`. Consider downgrading those prices a bit.  
- **Role outliers:** GK without set PK saves? DEF with unexpected goals? Often a CSV typo.  
- **Room bias audit:** After a mock auction, tweak `room_bias` and `blend_fvm_weight`.


## Updating Data Next Week/Season

- **Add seasons:** Drop more historical Excel files into `paths.history_xlsx`—the model will auto-blend them by recency.  
- **Edit CSVs:** If coaches reshuffle set pieces or PKs, edit the corresponding CSV rows.  
- **Re-run.** No other code changes needed.


## Troubleshooting

- **File not found / wrong sheet name:** Confirm `config_csv.yaml` paths; roster must use sheet `Tutti` with `header=1`.  
- **Garbage names (encoding):** Names are normalized (ASCII-folded + lowercased). Keep CSV names as in the roster to maximize matches.  
- **Zero prices:** Happens if VORP sums to 0 (e.g., all projections equal to replacement). Check replacement cuts & points logic.  
- **No PK/set-piece bumps:** Ensure `rigoristi.csv` / `setpieces.csv` have rows for that team; schema is strict.  


## FAQ

**Why single-tier VORP?**  
Simplicity at the table. Multi-tiered baselines add noise and are rarely worth the extra complexity for a 10-team league.

**Why use Mv̂ (not Fanta Media) as core rating?**  
You asked for analytic separation: Mv̂ captures pure performance; we then layer event EVs (G, A, cards) explicitly.

**How accurate is the defense modifier EV?**  
It’s an expectation over all weeks. Exact weekly lineup interactions will vary; we model the *team stack* effect and allocate shares to the most-used GK/DEF.

**Do I need all four CSVs?**  
Formations/starter CSV strongly improves Pv̂; the other three add smart bumps. You can run without them, but projections will be flatter.

**Can I reuse this for other leagues or scoring?**  
Yes—swap roster/history inputs and tweak scoring weights, modifier bands, and the 38-round constant.

---

Happy drafting—may your midfielders be underpriced and your rivals chase shiny GKs. ⚽️💸
