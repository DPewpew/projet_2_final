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