import os
import gc
import pandas as pd
import numpy as np
import gdown

# =========================
# CONFIG
# =========================
OUT_DIR = OUT_DIR = os.path.join("data", "imdb", "out")
OUT_FILMS_DIR = os.path.join(OUT_DIR, "films")
OUT_CREDITS_DIR = os.path.join(OUT_DIR, "credits")
os.makedirs(OUT_FILMS_DIR, exist_ok=True)
os.makedirs(OUT_CREDITS_DIR, exist_ok=True)

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
    """Supprime tous les fichiers part_*.csv d'un dossier."""
    if not os.path.exists(folder):
        return
    for fn in os.listdir(folder):
        if fn.startswith("part_") and fn.endswith(".csv"):
            os.remove(os.path.join(folder, fn))

def write_parts_from_df(df: pd.DataFrame, out_dir: str, part_rows: int):
    """Écrit un DataFrame en plusieurs fichiers part_XXX.csv."""
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
    """
    Génère un writer incrémental par parts (par lignes).
    Usage:
        writer = write_chunked_parts(...)
        writer.write(df_chunk)
        writer.close()
    """
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