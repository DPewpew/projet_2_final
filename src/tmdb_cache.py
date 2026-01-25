# src/tmdb_cache.py
# ============================================================
# tmdb_cache.py
# -------------
# Objectif:
# - Récupérer via TMDB la liste des films "en cours" + "à venir"
# - Mapper TMDB -> IMDb (imdb_id) -> tconst (ttxxxx)
# - Retourner un set de tconst utilisable pour filtrer les recommandations
#
# Contraintes Streamlit.io :
# - Ne PAS spammer l'API : cache + TTL
# - Ne PAS stocker la clé dans le code : utiliser st.secrets["TMDB_API_KEY"]
#
# Région / Langue :
# - region = "FR"
# - language = "fr-FR"
#
# Endpoints utilisés :
# - /movie/now_playing (liste)
# - /movie/upcoming (liste)
# - /movie/{movie_id}/external_ids (pour imdb_id)
#
# IMPORTANT :
# - On limite le nombre de "details calls" (external_ids) pour éviter quota + latence
# - On garde uniquement les films qui existent déjà dans ta DB locale (match tconst)
#   => c'est volontaire dans Solution A (pas d'injection live dans le modèle).
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import requests
import streamlit as st


# ============================================================
# Configuration
# ============================================================

TMDB_BASE_URL = "https://api.themoviedb.org/3"


@dataclass(frozen=True)
class TmdbConfig:
    region: str = "FR"
    language: str = "fr-FR"

    # On prend les 2 listes demandées
    include_now_playing: bool = True
    include_upcoming: bool = True

    # Limites pour maîtriser le temps de chargement / quota
    max_pages_now: int = 2         # 2 pages * 20 results/page = ~40 films
    max_pages_upcoming: int = 2    # ~40 films
    max_external_id_calls: int = 80  # limite globale d'appels external_ids

    # Cache TTL (en secondes)
    ttl_seconds: int = 24 * 3600


# ============================================================
# Helpers HTTP
# ============================================================

def _get_tmdb_key() -> str:
    """
    Récupère la clé TMDB depuis Streamlit secrets.
    - Local : .streamlit/secrets.toml
    - Cloud : Settings -> Secrets
    """
    if "TMDB_API_KEY" not in st.secrets:
        raise ValueError(
            "Clé TMDB manquante. Ajoute TMDB_API_KEY dans .streamlit/secrets.toml "
            "et dans les Secrets Streamlit Cloud."
        )
    return str(st.secrets["TMDB_API_KEY"]).strip()


def _tmdb_get(path: str, params: Optional[Dict] = None) -> Dict:
    """
    GET helper avec gestion d'erreurs minimaliste.
    """
    api_key = _get_tmdb_key()

    if params is None:
        params = {}

    params = dict(params)
    params["api_key"] = api_key

    url = f"{TMDB_BASE_URL}{path}"
    resp = requests.get(url, params=params, timeout=20)

    if resp.status_code != 200:
        raise RuntimeError(
            f"TMDB API error {resp.status_code} on {path} | "
            f"response: {resp.text[:200]}"
        )

    return resp.json()


# ============================================================
# Récupération listes TMDB
# ============================================================

def _fetch_list(endpoint: str, cfg: TmdbConfig, max_pages: int) -> List[Dict]:
    """
    Récupère une liste (now_playing/upcoming) sur N pages.
    Retourne une liste de dict TMDB (au moins id, title, etc.)
    """
    results: List[Dict] = []
    for page in range(1, max_pages + 1):
        data = _tmdb_get(
            endpoint,
            params={
                "region": cfg.region,
                "language": cfg.language,
                "page": page,
            },
        )
        page_results = data.get("results") or []
        results.extend(page_results)

    # dédup sur TMDB movie id
    seen = set()
    deduped = []
    for r in results:
        mid = r.get("id")
        if mid and mid not in seen:
            seen.add(mid)
            deduped.append(r)

    return deduped


def _fetch_external_ids(movie_id: int, cfg: TmdbConfig) -> Dict:
    """
    Récupère les external IDs (notamment imdb_id) pour un TMDB movie_id.
    """
    return _tmdb_get(
        f"/movie/{movie_id}/external_ids",
        params={"language": cfg.language},
    )


# ============================================================
# Construction candidate_set
# ============================================================

@st.cache_data(show_spinner="TMDB: récupération films en cours/à venir...", ttl=24 * 3600)
def get_candidate_imdb_ids(cfg_dict: Optional[Dict] = None) -> Set[str]:
    """
    Retourne un set de imdb_id (format ttXXXXXXX) issu de now_playing+upcoming.
    Cache TTL 24h par défaut.

    cfg_dict permet de sérialiser le dataclass dans le cache Streamlit:
    - ex: {"region":"FR","language":"fr-FR",...}
    """
    cfg = TmdbConfig(**(cfg_dict or {}))

    movies: List[Dict] = []

    if cfg.include_now_playing:
        movies.extend(_fetch_list("/movie/now_playing", cfg, cfg.max_pages_now))

    if cfg.include_upcoming:
        movies.extend(_fetch_list("/movie/upcoming", cfg, cfg.max_pages_upcoming))

    # dédup movie id
    seen = set()
    unique_movies = []
    for m in movies:
        mid = m.get("id")
        if mid and mid not in seen:
            seen.add(mid)
            unique_movies.append(m)

    # external_ids calls (limités)
    imdb_ids: Set[str] = set()
    calls = 0

    for m in unique_movies:
        if calls >= cfg.max_external_id_calls:
            break

        mid = m.get("id")
        if not mid:
            continue

        ext = _fetch_external_ids(int(mid), cfg)
        calls += 1

        imdb_id = ext.get("imdb_id")
        if imdb_id and isinstance(imdb_id, str) and imdb_id.startswith("tt"):
            imdb_ids.add(imdb_id.strip())

    return imdb_ids


def build_candidate_tconst_set(
    films_tconst_set: Set[str],
    cfg: Optional[TmdbConfig] = None,
) -> Set[str]:
    """
    Construit le set final des tconst filtrables.
    - films_tconst_set: set de tconst existants dans ta DB locale (films_ml)
    - retourne l'intersection avec les imdb_id récupérés (qui sont déjà des ttXXXX = tconst)

    NOTE:
    - IMDb id dans TMDB est déjà au format "tt....", identique à tconst.
    - Donc mapping = simple intersection.
    """
    if cfg is None:
        cfg = TmdbConfig()

    imdb_ids = get_candidate_imdb_ids(cfg_dict=cfg.__dict__)
    # Intersection: on garde seulement ce qui existe dans ta DB locale
    return set(imdb_ids).intersection(films_tconst_set)

# ============================================================
# Recherche TMDB (fallback quand film absent de la DB locale)
# ============================================================

@st.cache_data(show_spinner=False, ttl=24 * 3600)
def tmdb_search_movie(query: str, region: str = "FR", language: str = "fr-FR", max_results: int = 10) -> List[Dict]:
    """
    Recherche un film sur TMDB par titre.
    Retourne une liste de résultats TMDB (dicts), limités à max_results.
    """
    q = (query or "").strip()
    if not q:
        return []

    data = _tmdb_get(
        "/search/movie",
        params={
            "query": q,
            "region": region,
            "language": language,
            "include_adult": "false",
            "page": 1,
        },
    )

    results = data.get("results") or []
    return results[:max_results]


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def tmdb_movie_details(tmdb_movie_id: int, language: str = "fr-FR") -> Dict:
    """
    Récupère les détails d'un film TMDB (poster, budget, vote_average, etc.)
    """
    if not tmdb_movie_id:
        return {}
    return _tmdb_get(f"/movie/{int(tmdb_movie_id)}", params={"language": language})


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def tmdb_movie_recommendations(tmdb_movie_id: int, region: str = "FR", language: str = "fr-FR", max_pages: int = 1) -> List[Dict]:
    """
    Récupère les recommandations TMDB pour un film (liste de films TMDB).
    """
    if not tmdb_movie_id:
        return []

    out: List[Dict] = []
    for page in range(1, max_pages + 1):
        data = _tmdb_get(
            f"/movie/{int(tmdb_movie_id)}/recommendations",
            params={"region": region, "language": language, "page": page},
        )
        out.extend(data.get("results") or [])

    # dédup par id
    seen = set()
    dedup = []
    for r in out:
        mid = r.get("id")
        if mid and mid not in seen:
            seen.add(mid)
            dedup.append(r)
    return dedup


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def tmdb_movie_imdb_id(tmdb_movie_id: int, language: str = "fr-FR") -> Optional[str]:
    """
    Récupère imdb_id (ttxxxx) via external_ids.
    """
    if not tmdb_movie_id:
        return None
    ext = _tmdb_get(f"/movie/{int(tmdb_movie_id)}/external_ids", params={"language": language})
    imdb_id = ext.get("imdb_id")
    if isinstance(imdb_id, str) and imdb_id.startswith("tt"):
        return imdb_id.strip()
    return None


def tmdb_results_to_tconst_list(
    tmdb_movies: List[Dict],
    language: str = "fr-FR",
    max_external_id_calls: int = 60,
    ) -> List[str]:
    """
    Convertit une liste de films TMDB (dicts avec 'id') vers une liste de tconst (imdb_id)
    en appelant external_ids (limité).
    """
    if not tmdb_movies:
        return []

    tconsts: List[str] = []
    calls = 0

    for m in tmdb_movies:
        if calls >= max_external_id_calls:
            break

        mid = m.get("id")
        if not mid:
            continue

        imdb_id = tmdb_movie_imdb_id(int(mid), language=language)
        calls += 1

        if imdb_id:
            tconsts.append(imdb_id)

    # dédup en conservant l’ordre
    seen = set()
    out = []
    for t in tconsts:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

@st.cache_data(show_spinner=False, ttl=24 * 3600)
def tmdb_overview_from_tconst(tconst: str, language: str = "fr-FR") -> Optional[str]:
    """
    Récupère le synopsis (overview) TMDB à partir d'un tconst IMDb (ex: tt0035423).
    Utilise /find/{external_id} avec external_source=imdb_id.
    Cache 24h.
    """
    if not tconst:
        return None

    tconst = str(tconst).strip()
    if not tconst.startswith("tt"):
        return None

    # 1) Find TMDB movie à partir du tconst
    data = _tmdb_get(
        f"/find/{tconst}",
        params={"external_source": "imdb_id", "language": language},
    )

    movie_results = data.get("movie_results") or []
    if not movie_results:
        return None

    tmdb_id = movie_results[0].get("id")
    if not tmdb_id:
        return None

    # 2) Détails film -> overview
    details = _tmdb_get(f"/movie/{int(tmdb_id)}", params={"language": language})
    ov = details.get("overview")

    if isinstance(ov, str) and ov.strip():
        return ov.strip()

    return None