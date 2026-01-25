"""
src/site_app.py
---------------
UI Streamlit — Site démo + recommandations

V2 (Solution A intégrée) :
- Homepage : Top 10 par genre (ranking local) + priorité si possible aux films TMDB "en cours / à venir"
- Recherche film : affiche film + Top 10 reco filtrées uniquement sur films "en cours / à venir"
- Recherche personne : affiche Top 10 films liés à la personne, filtrés "en cours / à venir"

IMPORTANT :
- Pas d'injection live dans le modèle.
- On filtre les recommandations par un "candidate_tconst_set" calculé via TMDB (now_playing + upcoming),
  puis intersecté avec les tconst présents dans ta DB locale.
- Cache Streamlit obligatoire : TMDB est déjà caché dans tmdb_cache.py (ttl 24h).
"""

from __future__ import annotations

from typing import Optional, Set

import pandas as pd
import streamlit as st

from src.ml_data import load_films_ml, load_person_index
from src.reco_engine import (
    get_recommender,
    top_by_genre,
    search_person_names,
    recommend_from_person,
)
from src.tmdb_cache import (
    TmdbConfig,
    build_candidate_tconst_set,
    tmdb_search_movie,
    tmdb_movie_details,
    tmdb_movie_recommendations,
    tmdb_results_to_tconst_list,
    tmdb_overview_from_tconst,
)


# ============================================================
# Init (cache)
# ============================================================

@st.cache_resource(show_spinner="Initialisation du site...")
def _init_site():
    """
    Charge les DataFrames + initialise le moteur TF-IDF/KNN (cache_resource).
    """
    films = load_films_ml()
    persons = load_person_index()
    recommender = get_recommender(films)
    return films, persons, recommender


@st.cache_data(show_spinner="Préparation des films en cours / à venir (TMDB)...", ttl=24 * 3600)
def _get_candidate_set(films_tconst: pd.Series) -> Set[str]:
    """
    Construit le set final de tconst "en cours / à venir" utilisable pour filtrer les reco.
    Cache 24h (en plus du cache interne TMDB).
    """
    films_tconst_set = set(films_tconst.dropna().astype(str).tolist())

    cfg = TmdbConfig(
        region="FR",
        language="fr-FR",
        include_now_playing=True,
        include_upcoming=True,
        max_pages_now=2,
        max_pages_upcoming=2,
        max_external_id_calls=80,
        ttl_seconds=24 * 3600,
    )

    return build_candidate_tconst_set(films_tconst_set, cfg)


# ============================================================
# Helpers UI
# ============================================================

def _poster_url(poster_path: Optional[str]) -> Optional[str]:
    if poster_path and isinstance(poster_path, str) and poster_path.strip():
        return f"https://image.tmdb.org/t/p/w342{poster_path}"
    return None


def render_vignettes(df: pd.DataFrame, title: str):
    """
    Affiche une grille de vignettes (poster + titre) + bouton Voir (sélection film).
    """
    if title:
        st.subheader(title)

    if df is None or df.empty:
        st.info("Aucun résultat.")
        return

    cols = st.columns(5)
    for i, (_, row) in enumerate(df.iterrows()):
        col = cols[i % 5]
        with col:
            tconst = str(row.get("tconst", ""))
            poster = _poster_url(row.get("poster_path"))
            if poster:
                st.image(poster, use_container_width=True)

            st.markdown(f"**{row.get('primaryTitle', 'Titre inconnu')}**")
            if "startYear" in row and pd.notna(row["startYear"]):
                try:
                    st.caption(f"{int(float(row['startYear']))}")
                except Exception:
                    pass

            # Bouton "Voir" -> ouvre la fiche film (dans l'app, via session_state)
            if tconst:
                if st.button("Voir", key=f"see_{tconst}_{i}"):
                    st.session_state["selected_tconst"] = tconst



def _filter_on_candidates(df: pd.DataFrame, candidate_set: Set[str], top_n: int = 10) -> pd.DataFrame:
    """
    Filtre strict : si candidate_set est vide => retourne vide (pas de reco hors "en cours/à venir").
    """
    if df is None or df.empty:
        return pd.DataFrame()

    if not candidate_set:
        return pd.DataFrame()  # IMPORTANT : pas de reco si aucun match TMDB↔local

    out = df[df["tconst"].astype(str).isin(candidate_set)].copy()
    return out.head(top_n)



def _fallback_candidates_by_genre(
    films: pd.DataFrame,
    candidate_set: Set[str],
    genre: Optional[str],
    top_n: int,
) -> pd.DataFrame:
    """
    Fallback pour compléter une liste de reco filtrée candidates :
    - si genre connu : top_by_genre(...) puis filtrage candidates
    - sinon : ranking générique sur candidates (vote_count / popularity)
    """
    if not candidate_set:
        return pd.DataFrame()

    candidates_df = films[films["tconst"].astype(str).isin(candidate_set)].copy()
    if candidates_df.empty:
        return pd.DataFrame()

    if genre:
        df_top = top_by_genre(films, genre, top_n=top_n * 3, min_year=1980)  # large puis on filtre
        df_top = _filter_on_candidates(df_top, candidate_set, top_n=top_n)
        if not df_top.empty:
            return df_top

    # fallback générique
    # On privilégie vote_count puis popularity si dispo
    sort_cols = []
    if "vote_count" in candidates_df.columns:
        sort_cols.append("vote_count")
    if "popularity" in candidates_df.columns:
        sort_cols.append("popularity")
    if sort_cols:
        candidates_df = candidates_df.sort_values(sort_cols, ascending=False, na_position="last")
    return candidates_df.head(top_n)


def _dedup_by_tconst(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    return df.drop_duplicates(subset=["tconst"], keep="first")


# ============================================================
# Homepage — Ranking hybride
# ============================================================

def homepage_ui(films: pd.DataFrame, candidate_set: Set[str]):
    st.markdown("### 🎞️ Sélection par genre (Top 10)")

    # liste des genres
    genres = (
        films["genres"]
        .dropna()
        .astype(str)
        .str.split(",")
        .explode()
        .str.strip()
        .unique()
    )
    genres = sorted([g for g in genres if g])

    selected_genre = st.selectbox("Genre", genres, index=0, key="home_genre_select")

    # 1) Top local
    local_top = top_by_genre(films, selected_genre, top_n=10, min_year=1980)

    # 2) Top candidates (même ranking mais filtré candidates)
    candidates_top = _filter_on_candidates(
        top_by_genre(films, selected_genre, top_n=30, min_year=1980),
        candidate_set,
        top_n=10,
    )

    # 3) Fusion : priorité aux candidates (si présentes), puis complétion local
    merged = pd.concat([candidates_top, local_top], ignore_index=True)
    merged = _dedup_by_tconst(merged).head(10)

    # UI : label si candidates actives
    if candidate_set:
        st.caption("Priorité aux films *en cours / à venir* (TMDB), sinon complétion par le ranking local.")
    else:
        st.caption("TMDB indisponible ou aucun match local : ranking 100% local.")

    render_vignettes(merged, title=f"Top {selected_genre}")


# ============================================================
# Recherche FILM
# ============================================================

def film_search_ui(films: pd.DataFrame, recommender, candidate_set: Set[str]):
    st.markdown("### 🔎 Recherche par film")

    query = st.text_input("Titre du film", key="film_search_input")
    if not query:
        return

    q = query.lower()

    results = films[
        films["primaryTitle"].astype(str).str.lower().str.contains(q, na=False)
    ].head(30)

    if results.empty:
        st.warning("Film non trouvé dans la base locale. Recherche sur TMDB...")

        # 1) Search TMDB
        tmdb_hits = tmdb_search_movie(query, region="FR", language="fr-FR", max_results=10)
        if not tmdb_hits:
            st.info("Aucun résultat TMDB.")
            return

        # 2) Choix du bon film TMDB
        labels = []
        for h in tmdb_hits:
            title = h.get("title") or h.get("original_title") or "Titre inconnu"
            rd = h.get("release_date") or ""
            yr = rd[:4] if isinstance(rd, str) and len(rd) >= 4 else "----"
            labels.append(f"{title} ({yr})")

        pick = st.selectbox("Résultats TMDB", labels, key="tmdb_pick_select")
        pick_idx = labels.index(pick)
        picked = tmdb_hits[pick_idx]
        tmdb_id = int(picked.get("id"))

        # 3) Affichage fiche TMDB sélectionnée
        st.markdown("### 🎬 Film (TMDB)")
        details = tmdb_movie_details(tmdb_id, language="fr-FR")
        _render_tmdb_selected_movie(details)

        # 4) Reco TMDB/recommendations
        st.markdown("### ✨ Recommandations (en cours / à venir)")
        reco_tmdb = tmdb_movie_recommendations(tmdb_id, region="FR", language="fr-FR", max_pages=1)

        # Convert reco TMDB -> tconst (imdb_id)
        reco_tconst = tmdb_results_to_tconst_list(reco_tmdb, language="fr-FR", max_external_id_calls=60)

        # IMPORTANT : filtre strict sur candidate_set (en cours/à venir)
        reco_tconst = [t for t in reco_tconst if t in candidate_set]

        if not reco_tconst:
            st.info(
                "Aucune recommandation TMDB ne matche des films 'en cours / à venir' "
                "présents dans la DB locale."
            )
            return

        # 5) Afficher uniquement les films matchés dans ta DB locale (vignettes)
        reco_local = _get_local_by_tconst(films, reco_tconst)  # DB locale uniquement
        reco_local = reco_local.head(10)

        if reco_local.empty:
            st.info(
                "Recommandations matchées (tconst), mais aucune ligne trouvée dans films_ml "
                "(incohérence de données)."
            )
            return

        render_vignettes(reco_local, title="")
        return


    # Sélection film
    titles = results["primaryTitle"].tolist()
    selected_title = st.selectbox("Résultats", titles, key="film_select")

    film_row = results[results["primaryTitle"] == selected_title].iloc[0]
    tconst = str(film_row["tconst"])

    # Film sélectionné
    st.markdown("### 🎬 Film sélectionné")
    render_vignettes(pd.DataFrame([film_row]), title="")

    # Reco brutes (KNN)
    raw_reco = recommender.recommend_by_tconst(tconst, top_n=50)

    # Reco filtrées strict "en cours / à venir"
    st.markdown("### ✨ Recommandations (en cours / à venir)")
    reco_candidates = _filter_on_candidates(raw_reco, candidate_set, top_n=10)

    if reco_candidates.empty:
        st.info(
            "Aucune recommandation ne matche les films 'en cours / à venir' "
            "(TMDB ↔ DB locale)."
        )
        return

    render_vignettes(reco_candidates, title="")


# ============================================================
# Recherche PERSONNE
# ============================================================

def person_search_ui(films: pd.DataFrame, persons: pd.DataFrame, candidate_set: Set[str]):
    st.markdown("### 🔎 Recherche par personne")

    query = st.text_input("Nom (acteur / réalisateur)", key="person_search_input")

    if not query:
        return

    matches = search_person_names(persons, query)

    if not matches:
        st.warning("Aucune personne trouvée.")
        return

    selected_name = st.selectbox("Personnes", matches, key="person_select")

    st.markdown(f"### 🎭 Films liés à **{selected_name}** (en cours / à venir)")

    reco_df = recommend_from_person(
        films_ml=films,
        person_index=persons,
        person_name=selected_name,
        top_n=50,
    )

    reco_candidates = _filter_on_candidates(reco_df, candidate_set, top_n=10)

    if reco_candidates.empty:
        st.info(
            "Aucun film lié à cette personne ne matche les films 'en cours / à venir' "
            "(TMDB ↔ DB locale)."
        )
    else:
        render_vignettes(reco_candidates, title="")



# ============================================================
# Entrée principale
# ============================================================

def render_site():
    films, persons, recommender = _init_site()

    # Candidate set TMDB (tconst "en cours/à venir" matchés local)
    candidate_set = _get_candidate_set(films["tconst"])

    st.title("🎥 Cinéma — Démo & Recommandations")
    st.caption("Recherche locale + recommandations | Filtrage 'en cours / à venir' via TMDB")

    # ============================================================
    # Détails film (affiché si une vignette a été cliquée)
    # ============================================================
    selected = st.session_state.get("selected_tconst")

    if selected:
        row = films[films["tconst"].astype(str) == str(selected)]
        if not row.empty:
            r = row.iloc[0]

            st.markdown("---")
            st.subheader("🎬 Détails du film")

            c1, c2 = st.columns([1, 3])

            with c1:
                poster = _poster_url(r.get("poster_path"))
                if poster:
                    st.image(poster, use_container_width=True)

            with c2:
                # Titre
                st.markdown(f"## {r.get('primaryTitle', 'Titre inconnu')}")

                # Année
                start_year = r.get("startYear")
                if pd.notna(start_year):
                    try:
                        st.write(f"**Année :** {int(float(start_year))}")
                    except Exception:
                        pass

                # Réalisateur / casting (depuis films_ml)
                director = (r.get("director_name") or "").strip()
                cast_top = (r.get("cast_top") or "").strip()

                if director:
                    st.write(f"**Réalisateur :** {director}")
                if cast_top:
                    st.write(f"**Acteurs :** {cast_top}")

                # Note
                vote = r.get("vote_average")
                if pd.notna(vote):
                    try:
                        st.write(f"**Note :** {float(vote):.1f}/10")
                    except Exception:
                        pass

                # Budget
                budget = r.get("budget")
                if pd.notna(budget):
                    try:
                        st.write(f"**Budget :** {int(float(budget)):,} $".replace(",", " "))
                    except Exception:
                        pass

                # ============================================================
                # Synopsis : local si dispo, sinon TMDB via tconst
                # ============================================================
                local_overview = (r.get("overview") or "").strip()

                st.markdown("### Synopsis")
                if local_overview:
                    st.write(local_overview)
                else:
                    ov = tmdb_overview_from_tconst(str(selected), language="fr-FR")
                    if ov:
                        st.write(ov)
                    else:
                        st.info("Synopsis indisponible (local + TMDB).")

            # Actions
            cbtn1, cbtn2 = st.columns([1, 5])
            with cbtn1:
                if st.button("Fermer la fiche", key="close_movie_details"):
                    st.session_state.pop("selected_tconst", None)
                    st.rerun()

            st.markdown("---")

    # ============================================================
    # Tabs
    # ============================================================
    tabs = st.tabs(["Accueil", "Recherche film", "Recherche personne"])

    with tabs[0]:
        homepage_ui(films, candidate_set)

    with tabs[1]:
        film_search_ui(films, recommender, candidate_set)

    with tabs[2]:
        person_search_ui(films, persons, candidate_set)


def _dominant_genre_from_df(df: pd.DataFrame) -> Optional[str]:
    """
    Déduit le genre dominant d'une liste de films (colonne 'genres' format 'A,B,C').
    Retourne le genre le plus fréquent, ou None.
    """
    if df is None or df.empty or "genres" not in df.columns:
        return None

    g = (
        df["genres"]
        .dropna()
        .astype(str)
        .str.split(",")
        .explode()
        .str.strip()
    )
    g = g[g.ne("")]

    if g.empty:
        return None

    return g.value_counts().index[0]


def _get_local_by_tconst(films: pd.DataFrame, tconst_list: list[str]) -> pd.DataFrame:
    """
    Retourne les lignes films_ml correspondant à une liste de tconst, dans le même ordre.
    """
    if not tconst_list:
        return pd.DataFrame()

    tmp = films[films["tconst"].astype(str).isin([str(x) for x in tconst_list])].copy()
    if tmp.empty:
        return tmp

    # réordonner selon tconst_list
    order = {t: i for i, t in enumerate(tconst_list)}
    tmp["_ord"] = tmp["tconst"].astype(str).map(order)
    tmp = tmp.sort_values("_ord", ascending=True).drop(columns=["_ord"])
    return tmp


def _render_tmdb_selected_movie(details: dict):
    """
    Affiche la fiche du film TMDB sélectionné (minimal, comme ton design).
    """
    if not details:
        st.info("Impossible de récupérer les détails TMDB.")
        return

    title = details.get("title") or details.get("original_title") or "Titre inconnu"
    release_date = details.get("release_date") or ""
    year = release_date[:4] if isinstance(release_date, str) and len(release_date) >= 4 else None
    vote_avg = details.get("vote_average")
    poster_path = details.get("poster_path")
    poster = _poster_url(poster_path)

    c1, c2 = st.columns([1, 3])
    with c1:
        if poster:
            st.image(poster, use_container_width=True)
    with c2:
        st.markdown(f"## {title}")
        if year:
            st.write(f"**Année :** {year}")
        if vote_avg is not None:
            try:
                st.write(f"**Note :** {float(vote_avg):.1f}/10")
            except Exception:
                pass
