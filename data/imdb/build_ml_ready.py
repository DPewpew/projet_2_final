# data/imdb/build_ml_ready.py
# ============================================================
# Script OFFLINE (à lancer en local)
# - Lit: data/imdb/out/films/part_*.csv
# - Lit: data/imdb/out/credits/part_*.csv (chunks)
# - Produit:
#    1) data/imdb/out/ml_ready/films_ml/part_*.csv
#    2) data/imdb/out/ml_ready/person_index.csv
#
# Objectif:
# - Ajouter dans films_ml :
#     - director_name : str
#     - cast_top      : "Actor1, Actor2, Actor3"
# - Construire person_index (primaryName -> tconst) pour la recherche "personne"
#
# Notes:
# - On ne charge PAS tout credits en RAM.
# - On limite cast_top à 3 noms / film.
# ============================================================

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

# ============================================================
# CONFIG
# ============================================================

# Taille d'écriture en parts pour films_ml (pour rester sous 100MB sur GitHub)
FILMS_ML_PART_ROWS = 50_000

# Chunk de lecture credits
CREDITS_CHUNK_SIZE = 250_000

# Catégories credits utilisées
CAST_CATEGORIES = {"actor", "actress"}
DIRECTOR_CATEGORY = "director"

# Pour person_index (qui doit permettre de retrouver des films d'une personne)
PERSON_CATEGORIES = {"actor", "actress", "director"}

# ============================================================
# PATHS (relatifs au repo)
# ============================================================

ROOT = Path(__file__).resolve().parents[2]  # -> racine du projet (projet_2_final/)
IMDB_DIR = ROOT / "data" / "imdb"
OUT_DIR = IMDB_DIR / "out"

FILMS_DIR = OUT_DIR / "films"
CREDITS_DIR = OUT_DIR / "credits"

ML_READY_DIR = OUT_DIR / "ml_ready"
FILMS_ML_DIR = ML_READY_DIR / "films_ml"
PERSON_INDEX_PATH = ML_READY_DIR / "person_index.csv"

# ============================================================
# HELPERS
# ============================================================

def ensure_dir(p: Path) -> None:
    """Crée le dossier s'il n'existe pas."""
    p.mkdir(parents=True, exist_ok=True)

def list_parts(folder: Path) -> List[Path]:
    """Liste les fichiers part_*.csv dans un dossier, triés."""
    if not folder.exists():
        raise FileNotFoundError(f"Dossier introuvable: {folder}")
    parts = sorted(folder.glob("part_*.csv"))
    if not parts:
        raise FileNotFoundError(f"Aucun part_*.csv trouvé dans: {folder}")
    return parts

def clear_parts(folder: Path) -> None:
    """Supprime les part_*.csv existants dans un dossier."""
    if not folder.exists():
        return
    for f in folder.glob("part_*.csv"):
        f.unlink()

def write_parts_from_df(df: pd.DataFrame, out_dir: Path, part_rows: int) -> None:
    """Écrit df en plusieurs part_XXX.csv."""
    ensure_dir(out_dir)
    clear_parts(out_dir)

    n = len(df)
    part_idx = 1
    for start in range(0, n, part_rows):
        end = min(start + part_rows, n)
        out_path = out_dir / f"part_{part_idx:03d}.csv"
        df.iloc[start:end].to_csv(out_path, index=False)
        print(f"[WRITE] {out_path} rows={end-start:,}")
        part_idx += 1

# ============================================================
# BUILD
# ============================================================

def load_films() -> pd.DataFrame:
    """Charge et concatène films/out/part_*.csv (100k lignes => ok RAM)."""
    parts = list_parts(FILMS_DIR)
    dfs = []
    for p in parts:
        dfs.append(pd.read_csv(p, low_memory=False))
        print(f"[FILMS] loaded {p.name} rows={len(dfs[-1]):,}")
    films = pd.concat(dfs, ignore_index=True)
    print(f"[FILMS] total rows={len(films):,} cols={len(films.columns)}")
    return films

def build_cast_and_director_and_person_index(
    tconst_allowed: Set[str],
) -> Tuple[Dict[str, str], Dict[str, str], pd.DataFrame]:
    """
    Parcourt credits/out/part_*.csv en chunks et construit:
    - director_map: tconst -> director_name (1 seul, first seen)
    - cast_map:     tconst -> "A, B, C" (max 3 noms uniques)
    - person_index: dataframe (primaryName, tconst) pour PERSON_CATEGORIES
    """
    director_map: Dict[str, str] = {}
    cast_map_names: Dict[str, List[str]] = {}

    # Pour dédupliquer person_index sans exploser: set de paires (name,tconst)
    seen_pairs: Set[Tuple[str, str]] = set()
    person_rows: List[Tuple[str, str]] = []

    credit_parts = list_parts(CREDITS_DIR)

    for part in credit_parts:
        print(f"[CREDITS] reading {part.name} ...")
        reader = pd.read_csv(part, chunksize=CREDITS_CHUNK_SIZE, low_memory=False)

        for chunk_idx, chunk in enumerate(reader, start=1):
            # garde uniquement films de notre base locale
            chunk["tconst"] = chunk["tconst"].astype(str)
            chunk = chunk[chunk["tconst"].isin(tconst_allowed)]
            if chunk.empty:
                continue

            # Sécurise colonnes attendues
            # (tu les as: tconst, category, primaryName)
            if "category" not in chunk.columns or "primaryName" not in chunk.columns:
                raise ValueError("Colonnes attendues absentes dans credits: category / primaryName")

            chunk["category"] = chunk["category"].astype(str)
            chunk["primaryName"] = chunk["primaryName"].astype(str)

            # -------- DIRECTOR --------
            dmask = chunk["category"].eq(DIRECTOR_CATEGORY)
            if dmask.any():
                dsub = chunk.loc[dmask, ["tconst", "primaryName"]]
                for t, name in zip(dsub["tconst"].values, dsub["primaryName"].values):
                    if t not in director_map and name and name != "nan":
                        director_map[t] = name

            # -------- CAST TOP 3 (actor/actress) --------
            cmask = chunk["category"].isin(CAST_CATEGORIES)
            if cmask.any():
                csub = chunk.loc[cmask, ["tconst", "primaryName"]]
                for t, name in zip(csub["tconst"].values, csub["primaryName"].values):
                    if not name or name == "nan":
                        continue
                    lst = cast_map_names.get(t)
                    if lst is None:
                        cast_map_names[t] = [name]
                    else:
                        if name not in lst and len(lst) < 3:
                            lst.append(name)

            # -------- PERSON INDEX (actor/actress/director) --------
            pmask = chunk["category"].isin(PERSON_CATEGORIES)
            if pmask.any():
                psub = chunk.loc[pmask, ["primaryName", "tconst"]].dropna()
                # dédup au fil de l'eau
                for name, t in zip(psub["primaryName"].values, psub["tconst"].values):
                    name = str(name).strip()
                    t = str(t).strip()
                    if not name or name == "nan" or not t or t == "nan":
                        continue
                    key = (name, t)
                    if key not in seen_pairs:
                        seen_pairs.add(key)
                        person_rows.append(key)

            if chunk_idx % 5 == 0:
                print(f"  [part {part.name}] chunk={chunk_idx} | directors={len(director_map):,} | cast={len(cast_map_names):,} | person_pairs={len(person_rows):,}")

    # cast_map final en string
    cast_map: Dict[str, str] = {}
    for t, names in cast_map_names.items():
        cast_map[t] = ", ".join(names[:3])

    person_index = pd.DataFrame(person_rows, columns=["primaryName", "tconst"])
    print(f"[PERSON_INDEX] rows={len(person_index):,}")

    return director_map, cast_map, person_index

def main() -> None:
    ensure_dir(ML_READY_DIR)
    ensure_dir(FILMS_ML_DIR)

    # 1) films
    films = load_films()
    if "tconst" not in films.columns:
        raise ValueError("Colonne tconst absente dans films")
    films["tconst"] = films["tconst"].astype(str)
    tconst_allowed = set(films["tconst"].dropna().unique().tolist())

    # 2) credits -> director / cast_top / person_index
    director_map, cast_map, person_index = build_cast_and_director_and_person_index(tconst_allowed)

    # 3) enrich films_ml
    films["director_name"] = films["tconst"].map(director_map)
    films["cast_top"] = films["tconst"].map(cast_map)

    # (optionnel) remplace NaN par ""
    films["director_name"] = films["director_name"].fillna("")
    films["cast_top"] = films["cast_top"].fillna("")

    # 4) write outputs
    print("[WRITE] films_ml parts ...")
    write_parts_from_df(films, FILMS_ML_DIR, FILMS_ML_PART_ROWS)

    print("[WRITE] person_index.csv ...")
    person_index.to_csv(PERSON_INDEX_PATH, index=False)

    print("[DONE] Outputs:")
    print(" -", FILMS_ML_DIR)
    print(" -", PERSON_INDEX_PATH)

if __name__ == "__main__":
    main()
