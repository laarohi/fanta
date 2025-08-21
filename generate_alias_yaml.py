"""Generate alias mappings between CSV names and roster names.

This helper scans the CSV inputs referenced in ``config_csv.yaml`` and
attempts to match each player name to the corresponding name found in the
roster Excel file.  A YAML file named ``alias.yaml`` is produced mapping the
original CSV name to the roster name.  Review and adjust this mapping before
running ``build_model_csv.py``.
"""

import difflib
import re
import unicodedata
from pathlib import Path

import pandas as pd
import yaml


def ascii_fold(s: str) -> str:
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")


def norm_name(s: str) -> str:
    s = ascii_fold(str(s)).lower()
    s = re.sub(r"[^a-z0-9\s/,-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_roster(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Tutti", header=1)
    df = df.rename(columns={"Nome": "player", "Squadra": "team"})
    df['player_norm'] = df['player'].map(norm_name).str.replace("/", " ", regex=False)
    df['team_norm'] = df['team'].map(norm_name)
    return df[['player', 'player_norm', 'team_norm']]


def collect_csv_names(paths: dict) -> pd.DataFrame:
    frames = []
    for key, p in paths.items():
        fp = Path(p)
        if not fp.exists():
            continue
        df = pd.read_csv(fp)
        if key == 'starters':
            df = df.rename(columns={'Team Name': 'team', 'Player Name': 'player'})
        name_col = 'name' if 'name' in df.columns else 'player'
        df['team_norm'] = df['team'].map(norm_name)
        df['player_norm'] = df[name_col].map(norm_name).str.replace("/", " ", regex=False)
        df['csv_name'] = df[name_col]
        frames.append(df[['csv_name', 'player_norm', 'team_norm']])
    if frames:
        return pd.concat(frames).drop_duplicates()
    return pd.DataFrame(columns=['csv_name', 'player_norm', 'team_norm'])


def best_match(name_norm: str, team_norm: str, roster: pd.DataFrame) -> str | None:
    team_pool = roster[roster.team_norm == team_norm]
    candidates = team_pool['player_norm'].tolist()
    match = difflib.get_close_matches(name_norm, candidates, n=1, cutoff=0.8)
    if match:
        return team_pool[team_pool.player_norm == match[0]].iloc[0]['player']
    global_match = difflib.get_close_matches(name_norm, roster['player_norm'], n=1, cutoff=0.8)
    if global_match:
        return roster[roster.player_norm == global_match[0]].iloc[0]['player']
    return None


def main():
    cfg = yaml.safe_load(Path('config_csv.yaml').read_text())
    roster = load_roster(cfg['paths']['roster_xlsx'])
    csv_paths = {
        'rigoristi': cfg['paths']['csv_rigoristi'],
        'setpieces': cfg['paths']['csv_setpieces'],
        'starters': cfg['paths']['csv_formations_starters'],
        'afcon': cfg['paths']['csv_afcon'],
    }
    names = collect_csv_names(csv_paths)

    mapping = {}
    for row in names.itertuples(index=False):
        match = best_match(row.player_norm, row.team_norm, roster)
        mapping[row.csv_name] = match

    with open('alias.yaml', 'w', encoding='utf-8') as fh:
        yaml.safe_dump(mapping, fh, allow_unicode=True, sort_keys=True)

    print(f'alias.yaml written with {len(mapping)} entries')


if __name__ == '__main__':
    main()

