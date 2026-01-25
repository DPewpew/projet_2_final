# src/reco_engine.py
# ============================================================
# reco_engine.py
# --------------
# Moteur de recommandation + ranking homepage.
#
# Objectifs:
# 1) Ranking homepage : Top N par genre (simple, robuste, rapide)
# 2) Reco film -> films similaires (content-based via TF-IDF + KNN cosine)
# 3) Reco personne -> films (via person_index + ranking / similarité)
#
# Conçu pour fonctionner avec tes CSV "ml_ready":
# - films_ml: data/imdb/out/ml_ready/films_ml/part_*.csv
# - person_index: data/imdb/out/ml_ready/person_index.csv
#
# IMPORTANT:
# - Aucun téléchargement ici
# - Aucun traitement lourd de credits ici
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# ============================================================
# Helpers texte / score
# ============================================================

def _safe_str(x: object) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    s = str(x).strip()
    if s in {"", "nan", "None", "\\N"}:
        return ""
    return s


def build_soup_row(
    genres: str,
    director_name: str,
    cast_top: str,
    overview: str,
) -> str:
    """
    Soup textuelle pour content-based.
    Pondération simple:
    - genres x3
    - director x2
    - cast x1
    - overview x1
    """
    g = _safe_str(genres)
    d = _safe_str(director_name)
    c = _safe_str(cast_top)
    o = _safe_str(overview)
    return " ".join([g, g, g, d, d, c, o]).strip()


def _norm01(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    mn = s.min()
    mx = s.max()
    if pd.isna(mn) or pd.isna(mx) or mn == mx:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn)


def compute_rank_score(df: pd.DataFrame) -> pd.Series:
    """
    Score de ranking (homepage):
    score = 0.45*norm(popularity) + 0.45*norm(vote_average) + 0.10*log1p(vote_count)

    Fallbacks:
    - vote_average <- averageRating (IMDb) si absent
    - vote_count   <- numVotes (IMDb) si absent
    """
    popularity = _norm01(df.get("popularity", pd.Series([np.nan] * len(df))))
    vote_avg = df.get("vote_average")
    if vote_avg is None:
        vote_avg = df.get("averageRating")
    vote_avg = _norm01(vote_avg if vote_avg is not None else pd.Series([np.nan] * len(df)))

    vote_cnt = df.get("vote_count")
    if vote_cnt is None:
        vote_cnt = df.get("numVotes")
    vote_cnt = pd.to_numeric(vote_cnt if vote_cnt is not None else 0, errors="coerce").fillna(0)

    return 0.45 * popularity + 0.45 * vote_avg + 0.10 * np.log1p(vote_cnt)


# ============================================================
# Ranking homepage
# ============================================================

def top_by_genre(
    films_ml: pd.DataFrame,
    genre: str,
    top_n: int = 10,
    min_year: Optional[int] = None,
) -> pd.DataFrame:
    """
    Top N films d'un genre (homepage).
    """
    g = _safe_str(genre).lower()
    if not g:
        raise ValueError("genre vide")

    df = films_ml.copy()

    # filtre genre
    df = df[df["genres"].astype(str).str.lower().str.contains(g, na=False)].copy()

    # filtre année optionnel
    if min_year is not None and "startYear" in df.columns:
        df["startYear"] = pd.to_numeric(df["startYear"], errors="coerce")
        df = df[df["startYear"] >= min_year]

    df["score"] = compute_rank_score(df)
    df = df.sort_values("score", ascending=False).head(top_n)

    # colonnes utiles pour affichage vignette
    keep_cols = [c for c in ["tconst", "primaryTitle", "poster_path", "startYear", "genres", "score"] if c in df.columns]
    return df[keep_cols].reset_index(drop=True)


# ============================================================
# Content-based recommender (TF-IDF + KNN cosine)
# ============================================================

@dataclass
class RecoConfig:
    max_features: int = 50_000
    min_df: int = 2
    ngram_range: Tuple[int, int] = (1, 2)


class ContentKNNRecommender:
    """
    Reco film -> films similaires.

    - fit() : construit soup + TF-IDF + index KNN
    - recommend_by_tconst() : renvoie top N similaires
    """

    def __init__(self, cfg: Optional[RecoConfig] = None):
        self.cfg = cfg or RecoConfig()
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.knn: Optional[NearestNeighbors] = None
        self.X = None
        self.films_ml: Optional[pd.DataFrame] = None
        self.tconst_to_idx: Dict[str, int] = {}

    def fit(self, films_ml: pd.DataFrame) -> "ContentKNNRecommender":
        df = films_ml.copy()

        for col in ["tconst", "genres", "director_name", "cast_top", "overview"]:
            if col not in df.columns:
                df[col] = ""

        df["tconst"] = df["tconst"].astype(str).str.strip()

        df["soup"] = df.apply(
            lambda r: build_soup_row(
                r.get("genres", ""),
                r.get("director_name", ""),
                r.get("cast_top", ""),
                r.get("overview", ""),
            ),
            axis=1,
        )

        self.vectorizer = TfidfVectorizer(
            max_features=self.cfg.max_features,
            min_df=self.cfg.min_df,
            ngram_range=self.cfg.ngram_range,
            stop_words=None,
        )
        self.X = self.vectorizer.fit_transform(df["soup"].fillna(""))

        # KNN cosine (metric="cosine") -> distances = cosine distance
        self.knn = NearestNeighbors(n_neighbors=50, metric="cosine", algorithm="brute")
        self.knn.fit(self.X)

        self.films_ml = df
        self.tconst_to_idx = {t: i for i, t in enumerate(df["tconst"].tolist())}
        return self

    def recommend_by_tconst(self, tconst: str, top_n: int = 10) -> pd.DataFrame:
        if self.films_ml is None or self.knn is None or self.X is None:
            raise RuntimeError("Modèle non initialisé. Appelle fit() d'abord.")

        t = str(tconst).strip()
        if t not in self.tconst_to_idx:
            raise KeyError(f"Film introuvable dans la base (tconst): {t}")

        idx = self.tconst_to_idx[t]
        distances, indices = self.knn.kneighbors(self.X[idx], n_neighbors=top_n + 1)

        idxs = indices.ravel().tolist()
        dists = distances.ravel().tolist()

        # On retire le film lui-même (souvent en 1ère position distance=0)
        out_rows = []
        for i, d in zip(idxs, dists):
            if i == idx:
                continue
            out_rows.append((i, d))
            if len(out_rows) >= top_n:
                break

        out_idx = [i for i, _ in out_rows]
        out = self.films_ml.iloc[out_idx].copy()
        out["distance"] = [d for _, d in out_rows]
        out["similarity"] = 1 - out["distance"]

        keep_cols = [c for c in ["tconst", "primaryTitle", "poster_path", "startYear", "genres", "similarity"] if c in out.columns]
        return out[keep_cols].reset_index(drop=True)


# ============================================================
# Cache Streamlit du modèle (important pour fluidité)
# ============================================================

@st.cache_resource(show_spinner="Initialisation du moteur de recommandation...")
def get_recommender(films_ml: pd.DataFrame, cfg: Optional[RecoConfig] = None) -> ContentKNNRecommender:
    """
    Construit et met en cache le recommender.
    À appeler une fois au runtime Streamlit.
    """
    rec = ContentKNNRecommender(cfg=cfg)
    rec.fit(films_ml)
    return rec


# ============================================================
# Recherche personne -> films (via person_index)
# ============================================================

def _standardize_person_index(df_person: pd.DataFrame) -> pd.DataFrame:
    """
    Rend person_index compatible quel que soit ton schéma exact.

    Schémas possibles rencontrés:
    - (primaryName, tconst)  -> ton build actuel
    - (nconst, primaryName, tconst) -> éventuel autre version

    Retourne un DF avec colonnes:
    - primaryName
    - tconst
    """
    cols = set(df_person.columns)

    if {"primaryName", "tconst"}.issubset(cols):
        out = df_person[["primaryName", "tconst"]].copy()
    elif {"nconst", "primaryName", "tconst"}.issubset(cols):
        out = df_person[["primaryName", "tconst"]].copy()
    else:
        raise ValueError(
            "person_index.csv doit contenir au minimum "
            "('primaryName','tconst') ou ('nconst','primaryName','tconst'). "
            f"Colonnes trouvées: {sorted(cols)}"
        )

    out["primaryName"] = out["primaryName"].astype(str).str.strip()
    out["tconst"] = out["tconst"].astype(str).str.strip()
    out = out[(out["primaryName"] != "") & (out["tconst"] != "")]
    return out.drop_duplicates()


def search_person_names(df_person: pd.DataFrame, query: str, limit: int = 15) -> List[str]:
    """
    Retourne une liste de noms de personnes correspondant à la recherche (contains).
    """
    q = _safe_str(query).lower()
    if not q:
        return []

    pi = _standardize_person_index(df_person)
    names = (
        pi.loc[pi["primaryName"].str.lower().str.contains(q, na=False), "primaryName"]
        .dropna()
        .drop_duplicates()
        .head(limit)
        .tolist()
    )
    return names


def recommend_from_person(
    films_ml: pd.DataFrame,
    person_index: pd.DataFrame,
    person_name: str,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Reco "personne" -> films, format vignette.

    Stratégie simple (efficace):
    - récupérer les tconst liés à la personne
    - filtrer films_ml sur ces tconst
    - classer par ranking score
    """
    name = _safe_str(person_name)
    if not name:
        return pd.DataFrame()

    pi = _standardize_person_index(person_index)
    tconsts = pi.loc[pi["primaryName"].eq(name), "tconst"].dropna().unique().tolist()
    if not tconsts:
        return pd.DataFrame()

    df = films_ml[films_ml["tconst"].astype(str).isin(tconsts)].copy()
    if df.empty:
        return pd.DataFrame()

    df["score"] = compute_rank_score(df)
    df = df.sort_values("score", ascending=False).head(top_n)

    keep_cols = [c for c in ["tconst", "primaryTitle", "poster_path", "startYear", "genres", "score"] if c in df.columns]
    return df[keep_cols].reset_index(drop=True)
