# app.py
# ============================================================
# Streamlit HUB (app principale)
# - Point d'entrée pour Streamlit.io
# - Affiche un menu latéral (étude de marché / site démo)
# - Appelle des modules séparés dans src/
#
# IMPORTANT :
# - Ne lance PAS les scripts offline (traitement_db.py / build_ml_ready.py)
# - L'app consomme uniquement les CSV déjà présents dans data/
# ============================================================

import streamlit as st
from src.market_app import render_market
from src.site_app import render_site


# -----------------------------
# Config Streamlit
# -----------------------------
st.set_page_config(
    page_title="Projet 2 — Cinéma Creuse",
    layout="wide",
)

# -----------------------------
# Imports des pages (modules)
# -----------------------------
# NOTE : Ces fichiers doivent exister dans src/
# - src/market_app.py  -> render_market()
# - src/site_app.py    -> render_site()
#
# Pour l'instant, si tu n'as pas encore créé ces modules,
# tu peux laisser les imports commentés et utiliser les placeholders plus bas.
try:
    from src.market_app import render_market
except Exception:
    render_market = None

try:
    from src.site_app import render_site
except Exception:
    render_site = None


# -----------------------------
# UI — Sidebar
# -----------------------------
st.sidebar.title("Projet 2 — Hub")

section = st.sidebar.radio(
    "Navigation",
    [
        "Étude de marché",
        "Site (démo + reco)",
        "DB / Notes",
    ],
    index=0,
)

st.sidebar.markdown("---")

# Optionnel : petit indicateur de statut
st.sidebar.caption("Données ML : data/imdb/out/ml_ready/")
st.sidebar.caption("Données marché : data/INSEE/ + data/CNC/")


# -----------------------------
# Contenu principal
# -----------------------------
if section == "Étude de marché":
    st.title("Étude de marché — Creuse")
    st.caption("Visualisations INSEE + CNC (module séparé).")

    if render_market is None:
        st.warning(
            "Module `src/market_app.py` introuvable ou en erreur.\n\n"
            "Crée `src/market_app.py` avec une fonction `render_market()`."
        )
        st.code(
            "def render_market():\n"
            "    import streamlit as st\n"
            "    st.write('Market app placeholder')\n",
            language="python",
        )
    else:
        render_market()

elif section == "Site (démo + reco)":
    st.title("Site — Démo + Recommandations")
    st.caption("Recherche film/personne + recommandations + page film (module séparé).")

    if render_site is None:
        st.warning(
            "Module `src/site_app.py` introuvable ou en erreur.\n\n"
            "Crée `src/site_app.py` avec une fonction `render_site()`."
        )
        st.code(
            "def render_site():\n"
            "    import streamlit as st\n"
            "    st.write('Site app placeholder')\n",
            language="python",
        )
    else:
        render_site()

else:  # "DB / Notes"
    st.title("DB / Notes")

    st.info(
                """
            ## Documentation du pipeline (IMDb → parts → ML-ready → app Streamlit)

            Cette section documente **ce que fait chaque fichier**, **pourquoi il existe**, et **comment fonctionne le système de recommandation**.
            L’objectif est d’avoir un dépôt **reproductible** : tu peux reconstruire les CSV *ML-ready* à partir des sources IMDb/TMDB, puis déployer l’app sur **Streamlit Community Cloud** (streamlit.io) sans exécuter les scripts offline.

            ---

            # 1) Structure du projet

            ```
            projet_2_final/
            ├── app.py                      # HUB Streamlit (point d’entrée streamlit.io)
            ├── src/
            │   ├── market_app.py           # étude de marché (INSEE/CNC)
            │   ├── site_app.py             # site démo + recherche + reco
            │   ├── ml_data.py              # charge les CSV (parts) et renvoie films/persons
            │   ├── reco_engine.py          # ranking homepage + recommender KNN (cosine)
            │   └── tmdb_cache.py           # appels TMDB + cache CSV + candidate set
            │
            └── data/
                ├── INSEE/                  # csv marché
                ├── CNC/                    # csv marché
                └── imdb/
                    └── out/
                        ├── films/          # part_*.csv (100k films filtrés)
                        ├── credits/        # part_*.csv (crédits filtrés)
                        └── ml_ready/
                            ├── films_ml/   # part_*.csv (enrichi pour ML)
                            ├── person_index.csv
                            └── tmdb_cache.csv
            ```

            **Idée clé :**
            - Les gros fichiers IMDb sont **découpés en parts** pour rester < 100MB sur GitHub.
            - L’app Streamlit **ne fait que lire** ces parts et construire le modèle en mémoire (avec cache).
            - Les scripts `traitement_db.py` et `build_ml_ready.py` sont **offline** (local) et servent à générer les CSV versionnés.

            ---

            # 2) Fichiers OFFLINE (dans `data/imdb/`)

            Ces scripts se lancent en local (VS Code).  
            Ils ne doivent pas tourner sur Streamlit.io (trop lourd, trop lent, et nécessite des téléchargements).

            ## 2.1 `data/imdb/traitement_db.py`
            **Rôle :** télécharger/filtrer IMDb (TSV.gz) + fusion avec un export TMDB (csv) puis produire :
            - `data/imdb/out/films/part_*.csv`
            - `data/imdb/out/credits/part_*.csv`

            ### Pourquoi ce fichier existe ?
            IMDb est énorme. Tu ne peux pas pousser les TSV bruts sur GitHub, ni les charger sur Streamlit.io.  
            Donc tu fais un **filtrage dur** (ex: films depuis 1980, min votes, top 100k) et tu sors des CSV “propres”.

            ### Filtres appliqués (config)
            - `MIN_YEAR = 1980`
            - `MIN_VOTES = 300`
            - `TOP_N_FILMS = 100_000` (classés par `numVotes`)

            ### Output “films” (1 ligne = 1 film)
            Colonnes principales (selon ton script) :
            - `tconst`, `primaryTitle`, `startYear`, `genres`, `runtimeMinutes`, `averageRating`, `numVotes`,
            - `directors`, `writers`,
            - + champs TMDB (budget, revenue, poster_path, overview, popularity, vote_average, vote_count, etc.)

            ### Output “credits” (1 ligne = 1 personne x film)
            Colonnes principales :
            - `tconst`, `nconst`, `category`, `characters`, `primaryName`, `birthYear`, `primaryProfession`, etc.

            ### Fonctions utilitaires
            - `clear_dir(folder)` : supprime les anciens `part_*.csv`
            - `write_parts_from_df(df, out_dir, part_rows)` : découpe un DF en plusieurs `part_XXX.csv`
            - `write_chunked_parts(...)` : writer incrémental (utile pour `credits` afin d’éviter l’explosion RAM)

            ---

            ## 2.2 `data/imdb/build_ml_ready.py`
            **Rôle :** transformer tes parts “films + credits” en tables plus adaptées au ML et à l’UI :
            - `data/imdb/out/ml_ready/films_ml/part_*.csv`
            - `data/imdb/out/ml_ready/person_index.csv`

            ### Pourquoi tu as besoin d’un “ml_ready” ?
            Le fichier `credits` contient plusieurs lignes par film, donc pas pratique pour :
            - afficher vite “Réalisateur / casting principal” sur la page film,
            - créer un modèle de similarité “film ↔ film” (il faut 1 ligne par film + features texte)

            ### Ce que construit `build_ml_ready.py`

            #### A) `films_ml`
            C’est `films` + agrégations dérivées de `credits` :
            - `director_name` : nom du réalisateur (rows `category == 'director'`)
            - `cast_top` : top acteurs/actrices (ex: 5 premiers) concaténés en string
            - `soup` : “texte combiné” utilisé pour le TF‑IDF (voir reco_engine)

            #### B) `person_index.csv`
            Index “personne → films” pour la recherche acteur :
            - une ligne par `nconst` avec :
            - `primaryName`
            - `known_for_tconst` (liste/texte de tconst associés)

            ### Fonctions principales
            - `list_parts(dir)` : liste les `part_*.csv`
            - `load_films()` : concat les parts films
            - `build_cast_and_director_and_person_index(...)` :
            - lit credits
            - calcule `director_name`, `cast_top`
            - construit `person_index`
            - `write_parts_from_df(...)` : écrit `films_ml` en parts

            ---

            # 3) Fichiers APP (dans `src/`) — utilisés sur Streamlit.io

            ## 3.1 `src/ml_data.py`
            **Rôle :** charger des CSV découpés (`part_*.csv`) depuis `data/` (local repo ou raw GitHub).

            ### Fonctions
            - `_load_csv_parts(folder_or_base_url, pattern='part_*.csv')` : lit toutes les parts et concat
            - `load_films_ml(base)` : charge `data/imdb/out/ml_ready/films_ml/part_*.csv`
            - `load_person_index(base)` : charge `person_index.csv` (**doit contenir `nconst`**)
            - `load_ml_data(base)` : renvoie `(films_ml, persons_index)`

            **Pourquoi ce fichier est important ?**
            - Streamlit.io charge tout depuis le repo : il faut un loader robuste, “parts-friendly”.
            - `@st.cache_data` (dans les modules appelants) évite de recharger/concat à chaque interaction.

            ---

            ## 3.2 `src/reco_engine.py`
            **Rôle :** toute la logique de recommandation.

            Tu as 2 mécanismes complémentaires :

            ### (1) Homepage “Top par genre” (ranking)
            But : afficher un Top N stable et rapide, sans modèle complexe.

            - `compute_rank_score(...)` calcule un score avec :
            - popularité (TMDB `popularity`)
            - qualité (notes `vote_average` / `averageRating`)
            - crédibilité (volumes `vote_count` / `numVotes`)
            - `top_by_genre(films, genre, n=10)` : filtre `genres` puis trie par `rank_score`

            **Pourquoi c’est adapté à la homepage ?**
            - robuste (pas besoin d’entraînement),
            - résultat cohérent même sans interaction.

            ### (2) Reco “quand tu cherches un film” (KNN content-based)
            But : si l’utilisateur sélectionne un film, proposer des films similaires.

            #### Feature principale : un texte “soup”
            `build_soup_row(row)` assemble un texte à partir de colonnes disponibles :
            - `primaryTitle` / `originalTitle`
            - `genres`
            - `director_name`
            - `cast_top`
            - (optionnel) `original_language`, `startYear`

            Ensuite :
            - TF‑IDF (texte → vecteurs)
            - Similarité cosine (top voisins)

            Classe :
            - `ContentKNNRecommender`
            - `fit(films_ml)` construit TF‑IDF + matrice
            - `recommend_by_tconst(tconst, top_n)` renvoie les films les plus similaires

            #### Recherche acteur
            - `search_person_names(person_index, query)` : recherche
            - `recommend_from_person(...)` : récupère les films liés à l’acteur puis propose autour.

            ---

            ## 3.3 `src/tmdb_cache.py`
            **Rôle :** intégrer TMDB *sans casser le modèle local*.

            Deux usages :

            ### A) “Candidate set” (films en cours / à venir)
            - `build_candidate_tconst_set(...)` :
            - appelle TMDB (région FR, langue fr-FR)
            - récupère `now_playing` / `upcoming`
            - convertit TMDB → `imdb_id` (tconst)
            - intersecte avec ta DB locale → `candidate_set`

            ### B) Fallback “hors DB locale”
            Si un film n’existe pas en local :
            - `tmdb_search_movie(query)`
            - `tmdb_movie_details(tmdb_id)`
            - `tmdb_movie_recommendations(tmdb_id)`
            - `tmdb_results_to_tconst_list(results)`
            - `tmdb_overview_from_tconst(tconst)`

            ### Cache CSV
            `tmdb_cache.csv` évite de re-taper TMDB à chaque refresh (quotas + perf).

            ---

            ## 3.4 `src/site_app.py`
            **Rôle :** UI du “Netflix-like” : homepage + recherche + page film.

            ### Initialisation
            - `_init_site()` : charge `films_ml` + `person_index`, construit le recommender
            - `_get_candidate_set(tconst_series)` : set de tconst “en cours / à venir”

            ### UI / helpers
            - `render_vignettes(...)` : posters + titres
            - `homepage_ui(...)` : top genres (ranking) + priorisation candidate_set
            - `film_search_ui(...)` : recherche locale + reco KNN filtrées + fallback TMDB
            - `person_search_ui(...)` : recherche acteur + films liés (filtrés)
            - `render_site()` : tabs + page film (via `st.session_state['selected_tconst']`)

            ### Page film
            - affiche `director_name` + `cast_top` (depuis `films_ml`)
            - affiche synopsis :
            - local (`overview`) si dispo
            - sinon fallback TMDB (via `tmdb_overview_from_tconst`)

            ---

            ## 3.5 `src/market_app.py`
            **Rôle :** module étude de marché (INSEE/CNC) encapsulé.

            - `load_market_data()` : charge les CSV marché
            - `graph_1 ... graph_9()` : graphiques
            - `render_market()` : affichage global

            ---

            # 4) `app.py` (HUB)
            **Rôle :** navigation principale (menu latéral) + appel des 2 apps.

            **Important :** `app.py` reste léger et n’exécute jamais les scripts offline.

            ---

            # 5) Choix du ML (à présenter)

            - **Homepage** : ranking par genre (simple, stable, explicable)
            - **Recherche film** : content-based KNN (TF‑IDF + cosine) → similarité sur méta-données
            - **Contrainte métier** : filtrage “en cours / à venir” via TMDB (candidate_set)
            - **Fallback** : si film hors DB locale → reco TMDB “recommendations”

            ---
            # 6) Rebuild (repo reproductible)

            1. En local :
            - `python data/imdb/traitement_db.py`
            - `python data/imdb/build_ml_ready.py`

            2. Vérifier :
            - `data/imdb/out/films/part_*.csv`
            - `data/imdb/out/credits/part_*.csv`
            - `data/imdb/out/ml_ready/films_ml/part_*.csv`
            - `data/imdb/out/ml_ready/person_index.csv`

            3. Push GitHub (en parts) puis déploiement Streamlit.io.

            """
            )
    
    
    st.info(
                """
                ###Code sur le traitement de la db founie avec les filtres et la sortie en part csv
                
                
                
                # Filtres demandés
                MIN_YEAR = 1980
                MIN_VOTES = 300
                TOP_N_FILMS = 100_000

                # Chunk sizes (ajuste si besoin)
                CHUNK_PRINCIPALS = 500_000

                # Découpage des sorties en parts (par nombre de lignes)
                FILMS_PART_ROWS = 50_000         # 100k films -> 2 parts
                CREDITS_PART_ROWS = 250_000      # ajuste pour rester <100MB/part

                # =========================
                # SOURCES
                # =========================
                URL_NAME_BASICS      = "https://datasets.imdbws.com/name.basics.tsv.gz"
                URL_TITLE_BASICS     = "https://datasets.imdbws.com/title.basics.tsv.gz"
                URL_TITLE_CREW       = "https://datasets.imdbws.com/title.crew.tsv.gz"
                URL_TITLE_PRINCIPALS = "https://datasets.imdbws.com/title.principals.tsv.gz"
                URL_TITLE_RATINGS    = "https://datasets.imdbws.com/title.ratings.tsv.gz"

                URL_TMDB_GDRIVE      = "https://drive.google.com/uc?id=1VB5_gl1fnyBDzcIOXZ5vUSbCY68VZN1v"
                TMDB_FILE            = "tmdb_full.csv"

                # Temp file (intermédiaire) : on le garde local, on ne push pas
                PRINCIPALS_TMP = os.path.join(OUT_DIR, "principals_filtered_tmp.csv")

                # =========================
                # HELPERS
                # =========================
                def clear_dir(folder: str):
                    "Supprime tous les fichiers part_*.csv d'un dossier."
                    if not os.path.exists(folder):
                        return
                    for fn in os.listdir(folder):
                        if fn.startswith("part_") and fn.endswith(".csv"):
                            os.remove(os.path.join(folder, fn))

                def write_parts_from_df(df: pd.DataFrame, out_dir: str, part_rows: int):
                    "Écrit un DataFrame en plusieurs fichiers part_XXX.csv."
                    clear_dir(out_dir)
                    n = len(df)
                    part = 1
                    for start in range(0, n, part_rows):
                        end = min(start + part_rows, n)
                        out_path = os.path.join(out_dir, f"part_{part:03d}.csv")
                        df.iloc[start:end].to_csv(out_path, index=False)
                        print(f"  wrote {out_path} rows={end-start:,}", flush=True)
                        part += 1

                def write_chunked_parts(out_dir: str, part_rows: int):
                    ""
                    Génère un writer incrémental par parts (par lignes).
                    Usage:
                        writer = write_chunked_parts(...)
                        writer.write(df_chunk)
                        writer.close()
                    ""
                    clear_dir(out_dir)

                    class _Writer:
                        def __init__(self):
                            self.part_idx = 1
                            self.rows_in_part = 0
                            self.header_written = False
                            self.current_path = None

                        def _open_new_part(self):
                            self.current_path = os.path.join(out_dir, f"part_{self.part_idx:03d}.csv")
                            self.rows_in_part = 0
                            self.header_written = False
                            self.part_idx += 1

                        def write(self, df_chunk: pd.DataFrame):
                            if df_chunk is None or df_chunk.empty:
                                return

                            # Si pas de part ouverte, on en ouvre une
                            if self.current_path is None:
                                self._open_new_part()

                            start = 0
                            n = len(df_chunk)

                            while start < n:
                                remaining = part_rows - self.rows_in_part
                                take = min(remaining, n - start)
                                slice_df = df_chunk.iloc[start:start + take]

                                # écriture
                                slice_df.to_csv(
                                    self.current_path,
                                    index=False,
                                    mode="a",
                                    header=(not self.header_written),
                                )
                                self.header_written = True
                                self.rows_in_part += take
                                start += take

                                # Si part pleine -> nouvelle part
                                if self.rows_in_part >= part_rows:
                                    print(f"  finalized {self.current_path} rows={self.rows_in_part:,}", flush=True)
                                    self._open_new_part()

                        def close(self):
                            if self.current_path is not None and self.rows_in_part > 0:
                                print(f"  finalized {self.current_path} rows={self.rows_in_part:,}", flush=True)

                    return _Writer()

                # =========================
                # ETAPE 1 — Construire la liste des tconst (TOP 100k) en RAM
                # =========================
                print("[1/6] Loading basics + ratings (small cols)...", flush=True)

                basics = pd.read_csv(
                    URL_TITLE_BASICS,
                    sep="\t",
                    compression="gzip",
                    usecols=["tconst", "titleType", "primaryTitle", "originalTitle", "isAdult",
                            "startYear", "runtimeMinutes", "genres"],
                    low_memory=False
                )
                ratings = pd.read_csv(
                    URL_TITLE_RATINGS,
                    sep="\t",
                    compression="gzip",
                    usecols=["tconst", "averageRating", "numVotes"],
                    low_memory=False
                )

                # conversions
                basics["startYear"] = pd.to_numeric(basics["startYear"], errors="coerce")
                basics["runtimeMinutes"] = pd.to_numeric(basics["runtimeMinutes"], errors="coerce")
                basics["isAdult"] = pd.to_numeric(basics["isAdult"], errors="coerce")
                ratings["averageRating"] = pd.to_numeric(ratings["averageRating"], errors="coerce")
                ratings["numVotes"] = pd.to_numeric(ratings["numVotes"], errors="coerce")

                # filtres
                basics_f = basics[(basics["titleType"] == "movie") & (basics["startYear"] >= MIN_YEAR)].copy()
                ratings_f = ratings[ratings["numVotes"] >= MIN_VOTES].copy()

                print("[1/6] Ranking TOP films by numVotes...", flush=True)
                rank = (basics_f.merge(ratings_f, on="tconst", how="inner")
                            .sort_values("numVotes", ascending=False)
                            .head(TOP_N_FILMS))

                tconst_keep = rank["tconst"].unique()
                tconst_set = set(tconst_keep)

                print(f"[1/6] Kept tconst: {len(tconst_keep):,}", flush=True)

                del basics, ratings, basics_f, ratings_f, rank
                gc.collect()

                # =========================
                # ETAPE 2 — Construire FILMS (1 ligne par film)
                # =========================
                print("[2/6] Loading crew...", flush=True)
                crew = pd.read_csv(
                    URL_TITLE_CREW,
                    sep="\t",
                    compression="gzip",
                    usecols=["tconst", "directors", "writers"],
                    low_memory=False
                )
                crew = crew[crew["tconst"].isin(tconst_keep)]

                print("[2/6] Loading TMDB (download if missing)...", flush=True)
                if not os.path.exists(TMDB_FILE):
                    gdown.download(URL_TMDB_GDRIVE, TMDB_FILE, quiet=False)

                tmdb_raw = pd.read_csv(TMDB_FILE, low_memory=False)
                tmdb = tmdb_raw[
                    ["imdb_id", "budget", "revenue", "production_countries", "original_language",
                    "release_date", "runtime", "poster_path", "overview", "popularity",
                    "vote_average", "vote_count"]
                ].copy()

                tmdb["imdb_id"] = tmdb["imdb_id"].astype(str).str.strip()
                tmdb = tmdb[tmdb["imdb_id"].notna() & tmdb["imdb_id"].ne("") & tmdb["imdb_id"].ne("nan")]
                tmdb = tmdb.drop_duplicates(subset=["imdb_id"], keep="first")

                for col in ["budget", "revenue"]:
                    tmdb[col] = pd.to_numeric(tmdb[col], errors="coerce").replace(0, np.nan)

                tmdb["runtime"] = pd.to_numeric(tmdb["runtime"], errors="coerce")
                tmdb["vote_average"] = pd.to_numeric(tmdb["vote_average"], errors="coerce")
                tmdb["vote_count"] = pd.to_numeric(tmdb["vote_count"], errors="coerce")
                tmdb["popularity"] = pd.to_numeric(tmdb["popularity"], errors="coerce")

                print("[2/6] Reloading basics+ratings for final films table...", flush=True)
                basics2 = pd.read_csv(
                    URL_TITLE_BASICS,
                    sep="\t",
                    compression="gzip",
                    usecols=["tconst", "titleType", "primaryTitle", "originalTitle", "isAdult",
                            "startYear", "runtimeMinutes", "genres"],
                    low_memory=False
                )
                ratings2 = pd.read_csv(
                    URL_TITLE_RATINGS,
                    sep="\t",
                    compression="gzip",
                    usecols=["tconst", "averageRating", "numVotes"],
                    low_memory=False
                )

                basics2["startYear"] = pd.to_numeric(basics2["startYear"], errors="coerce")
                basics2["runtimeMinutes"] = pd.to_numeric(basics2["runtimeMinutes"], errors="coerce")
                basics2["isAdult"] = pd.to_numeric(basics2["isAdult"], errors="coerce")
                ratings2["averageRating"] = pd.to_numeric(ratings2["averageRating"], errors="coerce")
                ratings2["numVotes"] = pd.to_numeric(ratings2["numVotes"], errors="coerce")

                basics2 = basics2[basics2["tconst"].isin(tconst_keep)]
                ratings2 = ratings2[ratings2["tconst"].isin(tconst_keep)]

                print("[2/6] Building FILMS dataframe...", flush=True)
                df_films = (basics2
                            .merge(ratings2, on="tconst", how="left")
                            .merge(crew, on="tconst", how="left")
                            .merge(tmdb, left_on="tconst", right_on="imdb_id", how="left")
                            .drop(columns=["imdb_id"]))

                print(f"[2/6] FILMS built shape={df_films.shape}", flush=True)

                print("[2/6] Writing FILMS parts...", flush=True)
                write_parts_from_df(df_films, OUT_FILMS_DIR, FILMS_PART_ROWS)

                del crew, tmdb_raw, tmdb, basics2, ratings2, df_films
                gc.collect()

                # =========================
                # ETAPE 3 — Principals filtré (chunks) + nconst_needed
                # =========================
                print("[3/6] Extracting principals (chunks) for selected tconst...", flush=True)

                # reset tmp
                if os.path.exists(PRINCIPALS_TMP):
                    os.remove(PRINCIPALS_TMP)

                nconst_needed = set()
                first_write = True

                reader = pd.read_csv(
                    URL_TITLE_PRINCIPALS,
                    sep="\t",
                    compression="gzip",
                    usecols=["tconst", "nconst", "category", "job", "characters"],
                    chunksize=CHUNK_PRINCIPALS,
                    low_memory=False
                )

                for i, chunk in enumerate(reader, start=1):
                    chunk = chunk[chunk["tconst"].isin(tconst_set)]
                    if chunk.empty:
                        continue

                    nconst_needed.update(chunk["nconst"].dropna().unique().tolist())

                    chunk.to_csv(PRINCIPALS_TMP, index=False, mode="a", header=first_write)
                    first_write = False

                    print(f"  principals chunk {i}: kept rows={len(chunk):,} | nconst_needed={len(nconst_needed):,}", flush=True)
                    del chunk
                    gc.collect()

                print(f"[3/6] Principals tmp saved: {PRINCIPALS_TMP}", flush=True)
                print(f"[3/6] Unique nconst needed: {len(nconst_needed):,}", flush=True)

                # =========================
                # ETAPE 4 — Charger name.basics filtré sur nconst_needed
                # =========================
                print("[4/6] Loading name.basics (filtered)...", flush=True)
                df_names = pd.read_csv(
                    URL_NAME_BASICS,
                    sep="\t",
                    compression="gzip",
                    usecols=["nconst", "primaryName", "birthYear", "deathYear", "primaryProfession"],
                    low_memory=False
                )

                df_names = df_names[df_names["nconst"].isin(nconst_needed)].copy()
                df_names["birthYear"] = pd.to_numeric(df_names["birthYear"], errors="coerce")
                df_names["deathYear"] = pd.to_numeric(df_names["deathYear"], errors="coerce")

                print(f"[4/6] Names filtered shape={df_names.shape}", flush=True)

                # =========================
                # ETAPE 5 — Construire CREDITS en parts (join tmp + names)
                # =========================
                print("[5/6] Writing CREDITS parts...", flush=True)

                credits_writer = write_chunked_parts(OUT_CREDITS_DIR, CREDITS_PART_ROWS)

                tmp_reader = pd.read_csv(PRINCIPALS_TMP, chunksize=CHUNK_PRINCIPALS, low_memory=False)

                for j, pchunk in enumerate(tmp_reader, start=1):
                    out_chunk = pchunk.merge(df_names, on="nconst", how="left")
                    credits_writer.write(out_chunk)

                    print(f"  credits chunk {j}: processed rows={len(out_chunk):,}", flush=True)
                    del pchunk, out_chunk
                    gc.collect()

                credits_writer.close()

                # =========================
                # ETAPE 6 — Nettoyage
                # =========================
                # Optionnel : supprimer le tmp après génération
                # os.remove(PRINCIPALS_TMP)

                print("[DONE] Output folders:", flush=True)
                print(" -", OUT_FILMS_DIR, flush=True)
                print(" -", OUT_CREDITS_DIR, flush=True)
                print("[DONE] You can load all parts with glob + concat.", flush=True)
                """
    )
    
    
    
    
    st.info(
        """
        ## Système de recommandation — Détails Machine Learning

        Le projet repose sur un **système de recommandation content-based**, sans données utilisateurs
        (pas d’historique de clics, pas de notes par utilisateur).

        Le ML est volontairement **simple, explicable et robuste**, afin d’être :
        - compatible avec un déploiement Streamlit.io,
        - cohérent avec une base IMDb/TMDB statique.

        ---

        # 1) Pourquoi un modèle content-based (et pas collaboratif)

        Un système collaboratif nécessite :
        - des utilisateurs identifiés,
        - des interactions (notes, clics, historiques).

        Dans ce projet :
        - il n’y a **pas d’utilisateurs**,
        - pas de logs de consommation,
        - uniquement des métadonnées films.

        👉 Le **content-based filtering** est donc le seul choix pertinent.

        ---

        # 2) Architecture générale du ML

        Le système de reco est composé de **2 mécanismes distincts** :

        ### A) Homepage — Ranking (pas de ML lourd)
        - Objectif : afficher un Top films par genre
        - Méthode : score calculé à partir de métriques existantes
        - Avantage : rapide, stable, explicable

        ### B) Recherche film / personne — Similarité ML
        - Objectif : proposer des films similaires à un film (ou à un acteur)
        - Méthode : similarité de contenu (TF-IDF + cosine)
        - Avantage : pas besoin de variable cible

        ---

        # 3) Features utilisées pour le ML (films_ml)

        Le modèle travaille sur une table **1 ligne = 1 film** (`films_ml`).

        Les features retenues sont **exclusivement textuelles et catégorielles**,
        car elles décrivent le contenu du film.

        ### Features utilisées :

        #### 1) Genres (`genres`)
        - Ex: "Comedy,Romance"
        - Feature la plus discriminante pour la similarité
        - Permet de ne jamais recommander un film hors univers

        #### 2) Réalisateur (`director_name`)
        - Les films d’un même réalisateur partagent souvent un style
        - Très pertinent pour la recommandation qualitative

        #### 3) Casting principal (`cast_top`)
        - Top acteurs/actrices (2 à 5 max)
        - Important pour la recherche “par acteur”
        - Ajoute une dimension shéma en étoile

        #### 4) Titres (`primaryTitle`, `originalTitle`)
        - Permet de rapprocher des sagas, remakes, suites
        - Améliore la cohérence sémantique

        ---

        # 4) Construction de la feature ML principale : la "soup"

        Toutes les features sont combinées dans une **feature texte unique** appelée `soup`.

        Exemple simplifié :

            soup = "
                Kate & Leopold
                Comedy Romance Fantasy
                James Mangold
                Meg Ryan Hugh Jackman
            "

        Pourquoi cette approche ?
        - TF-IDF fonctionne très bien sur du texte libre
        - Pas besoin de normaliser chaque feature séparément
        - Méthode classique utilisée dans de nombreux systèmes de reco simples

        ---

        # 5) Modèle utilisé : TF-IDF + Similarité Cosine

        ### Étape 1 — Vectorisation (TF-IDF)
        - Chaque film est transformé en vecteur numérique
        - Les mots rares sont plus importants que les mots fréquents
        - Aucun apprentissage supervisé

        ### Étape 2 — Similarité Cosine
        - Mesure l’angle entre deux vecteurs films
        - Plus l’angle est faible → films similaires
        - Résultat : un score de similarité ∈ [0,1]

        ### Étape 3 — KNN implicite
        - Pour un film donné :
        - on calcule la similarité avec tous les autres films
        - on prend les **Top N voisins**
        - Pas besoin d’un `KNeighborsClassifier` classique
        - Plus rapide et plus contrôlable

        ---

        # 6) Recherche par film

        Workflow :
        1. L’utilisateur sélectionne un film
        2. On récupère son `tconst`
        3. On calcule les similarités avec tous les films
        4. On retourne les films les plus proches
        5. On applique un **filtre métier** (voir section suivante)

        ---

        # 7) Recherche par personne (acteur / réalisateur)

        Workflow :
        1. Recherche du nom dans `person_index`
        2. Récupération des films liés à cette personne
        3. Union des recommandations ML de ces films
        4. Suppression des doublons
        5. Filtrage métier

        Cela permet :
        - une recherche “acteur” sans modèle spécifique
        - de rester cohérent avec le même moteur ML

        ---

        # 8) Filtrage métier : films « en cours / à venir »

        Le modèle ML calcule la similarité **sur toute la base**,
        mais l’affichage final applique une contrainte métier forte :

        👉 **ne recommander que des films en salle ou à venir**.

        Cette contrainte est implémentée via :
        - appels TMDB (`now_playing`, `upcoming`)
        - conversion TMDB → IMDb (`imdb_id`)
        - création d’un `candidate_set` (ensemble de tconst autorisés)

        Le ML **ne change pas** :
        - on filtre simplement les résultats finaux.

        Avantage :
        - séparation claire ML / règles métier
        - modèle stable et réutilisable

        ---

        # 9) Pourquoi ce choix est pertinent pour un projet Data Analyst

        - Modèle explicable (pas de boîte noire)
        - Pas de sur-ingénierie
        - Performant sur 100k films
        - Déployable sur Streamlit.io
        - Aligné avec une étude de marché (logique métier)

        Ce système est volontairement **simple mais solide** :
        il montre la maîtrise du pipeline data, du feature engineering,
        et de l’intégration ML dans une application réelle.
        """
        )
    st.info(
                """
                ###Code sur le systeme de recommandation 
                
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
                    "
                    Soup textuelle pour content-based.
                    Pondération simple:
                    - genres x3
                    - director x2
                    - cast x1
                    - overview x1
                    "
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
                    "
                    Score de ranking (homepage):
                    score = 0.45*norm(popularity) + 0.45*norm(vote_average) + 0.10*log1p(vote_count)

                    Fallbacks:
                    - vote_average <- averageRating (IMDb) si absent
                    - vote_count   <- numVotes (IMDb) si absent
                    "
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
                    "
                    Top N films d'un genre (homepage).
                    "
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
                    "
                    Reco film -> films similaires.

                    - fit() : construit soup + TF-IDF + index KNN
                    - recommend_by_tconst() : renvoie top N similaires
                    "

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
                    "
                    Construit et met en cache le recommender.
                    À appeler une fois au runtime Streamlit.
                    "
                    rec = ContentKNNRecommender(cfg=cfg)
                    rec.fit(films_ml)
                    return rec


                # ============================================================
                # Recherche personne -> films (via person_index)
                # ============================================================

                def _standardize_person_index(df_person: pd.DataFrame) -> pd.DataFrame:
                    "
                    Rend person_index compatible quel que soit ton schéma exact.

                    Schémas possibles rencontrés:
                    - (primaryName, tconst)  -> ton build actuel
                    - (nconst, primaryName, tconst) -> éventuel autre version

                    Retourne un DF avec colonnes:
                    - primaryName
                    - tconst
                    "
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
                    "
                    Retourne une liste de noms de personnes correspondant à la recherche (contains).
                    "
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
                    "
                    Reco "personne" -> films, format vignette.

                    Stratégie simple (efficace):
                    - récupérer les tconst liés à la personne
                    - filtrer films_ml sur ces tconst
                    - classer par ranking score
                    "
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
                """
    )