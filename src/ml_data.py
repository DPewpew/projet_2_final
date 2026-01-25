"""
ml_data.py
----------
Chargement des données ML-ready (films_ml + person_index).

Responsabilité UNIQUE :
- Charger et concaténer les fichiers CSV découpés (part_XXX.csv)
- Retourner des DataFrames propres et prêts pour le moteur de reco

AUCUN ML ici.
AUCUN affichage Streamlit ici.
"""

from pathlib import Path
import pandas as pd
import streamlit as st

# ============================================================
# Chemins (relatifs à la racine du projet)
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]  # projet_2_final/
DATA_DIR = BASE_DIR / "data" / "imdb" / "out" / "ml_ready"

FILMS_ML_DIR = DATA_DIR / "films_ml"
PERSON_INDEX_PATH = DATA_DIR / "person_index.csv"


# ============================================================
# Utils
# ============================================================

def _load_csv_parts(folder: Path) -> pd.DataFrame:
    """
    Charge et concatène tous les fichiers part_*.csv d'un dossier.

    Parameters
    ----------
    folder : Path
        Dossier contenant les part_XXX.csv

    Returns
    -------
    pd.DataFrame
    """
    if not folder.exists():
        raise FileNotFoundError(f"Dossier introuvable : {folder}")

    parts = sorted(folder.glob("part_*.csv"))
    if not parts:
        raise FileNotFoundError(f"Aucun fichier part_*.csv dans {folder}")

    dfs = []
    for p in parts:
        df = pd.read_csv(p, low_memory=False)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


# ============================================================
# Chargement ML-ready
# ============================================================

@st.cache_data(show_spinner="Chargement des films (ML-ready)...")
def load_films_ml() -> pd.DataFrame:
    """
    Charge tous les films ML-ready (concat des parts).

    Returns
    -------
    pd.DataFrame
        1 ligne = 1 film
    """
    df_films = _load_csv_parts(FILMS_ML_DIR)

    # Sécurité minimale
    if "tconst" not in df_films.columns:
        raise ValueError("Colonne 'tconst' absente de films_ml")

    return df_films


@st.cache_data(show_spinner="Chargement de l'index personnes...")
def load_person_index() -> pd.DataFrame:
    if not PERSON_INDEX_PATH.exists():
        raise FileNotFoundError(f"Fichier introuvable : {PERSON_INDEX_PATH}")

    df_person = pd.read_csv(PERSON_INDEX_PATH, low_memory=False)

    required = {"primaryName", "tconst"}
    missing = required - set(df_person.columns)

    if missing:
        raise ValueError(
            f"Colonnes manquantes dans person_index : {missing} | "
            f"Colonnes trouvées : {list(df_person.columns)}"
        )

    return df_person



# ============================================================
# Chargement combiné (optionnel mais pratique)
# ============================================================

@st.cache_data(show_spinner="Chargement des données ML...")
def load_ml_data():
    """
    Charge films + personnes en une seule fois.

    Returns
    -------
    dict
        {
            "films": DataFrame,
            "persons": DataFrame
        }
    """
    return {
        "films": load_films_ml(),
        "persons": load_person_index()
    }
