import os
import re
import json
import numpy as np
import pandas as pd
import joblib
from scipy import sparse
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# ---------- config ----------
DATA_PATH = os.environ.get(
    "MUSICOPHILE_DATA",
    r"C:\Datasets\musicophile.ai\data\spotify_dataset.csv",
)

OUT_DIR = os.environ.get("MUSICOPHILE_ARTIFACTS", os.path.join("backend", "artifacts"))
os.makedirs(OUT_DIR, exist_ok=True)

AUDIO_FEATURES = [
    "Tempo", "loudness", "Energy", "Danceability",
    "Positiveness", "Speechiness", "Liveness",
    "Acousticness", "Instrumentalness"
]
META_COLS = ["Genre", "emotion", "Explicit", "Key"]

# Speed/quality knobs
MAX_LYRIC_CHARS = 4000
TFIDF_MAX_FEATURES = 30000
TFIDF_MIN_DF = 20
SVD_DIM = 128

# Candidate retrieval size in stage1 (for reranking later)
DEFAULT_K_CANDIDATES = 2000

# ---------- utils ----------
def parse_loudness(x):
    if pd.isna(x):
        return np.nan
    s = str(x).lower().strip().replace("db", "")
    s = s.replace("−", "-").replace("–", "-").replace("—", "-").replace(",", ".")
    m = re.search(r"-?\d+(\.\d+)?", s)
    return float(m.group(0)) if m else np.nan

def clean_lyrics(s: str, max_chars: int = 4000) -> str:
    s = "" if pd.isna(s) else str(s).lower()
    s = re.sub(r"\[.*?\]", " ", s)      # remove [chorus], [verse], etc.
    s = s.replace("\n", " ")
    s = re.sub(r"[^a-z\s']", " ", s)    # keep letters/spaces/apostrophes
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_chars]

def main():
    print("Loading:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)

    # rename columns to stable names
    df = df.rename(columns={
        "Artist(s)": "artist",
        "song": "song",
        "text": "lyrics",
        "Loudness (db)": "loudness",
    })

    # required columns check
    for col in ["song", "artist", "lyrics", "loudness"] + AUDIO_FEATURES:
        if col not in df.columns and col != "loudness":
            pass

    # lyrics clean
    df["lyrics_clean"] = df["lyrics"].apply(lambda x: clean_lyrics(x, MAX_LYRIC_CHARS))

    # loudness parse
    df["loudness"] = df["loudness"].apply(parse_loudness)

    # numeric audio
    for col in AUDIO_FEATURES:
        if col == "loudness":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # keep only rows usable for recs
    df = df.dropna(subset=["song", "artist", "lyrics_clean", "Tempo", "Energy", "Danceability"]).reset_index(drop=True)

    # fill remaining audio NaNs
    for col in AUDIO_FEATURES:
        df[col] = df[col].fillna(df[col].median())

    # ---------- build audio ----------
    print("Building audio matrix...")
    scaler = StandardScaler()
    X_audio = scaler.fit_transform(df[AUDIO_FEATURES].values).astype(np.float32)
    X_audio = normalize(X_audio, norm="l2", axis=1).astype(np.float32)

    # ---------- build meta (one-hot -> SVD -> normalized) ----------
    print("Building meta one-hot...")
    X_meta_oh = pd.get_dummies(df[META_COLS].fillna("UNK"), drop_first=False)
    meta_feature_names = X_meta_oh.columns.tolist()
    X_meta_oh = X_meta_oh.values.astype(np.float32)

    print(f"Compressing meta with SVD to {SVD_DIM} dims...")
    svd = TruncatedSVD(n_components=SVD_DIM, random_state=42)
    X_meta = svd.fit_transform(X_meta_oh).astype(np.float32)
    X_meta = normalize(X_meta, norm="l2", axis=1).astype(np.float32)

    # ---------- build lyrics TF-IDF (sparse) ----------
    print("Building TF-IDF lyrics (sparse)...")
    tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=TFIDF_MIN_DF,
        max_features=TFIDF_MAX_FEATURES,
        sublinear_tf=True
    )
    X_lyrics = tfidf.fit_transform(df["lyrics_clean"])
    # (TF-IDF is already l2-normalized by default; cosine works well.)

    # ---------- stage1 vectors for FAISS ----------
    print("Building stage1 vectors (audio + meta_svd)...")
    X_stage1 = np.hstack([X_audio, X_meta]).astype(np.float32)
    X_stage1 = normalize(X_stage1, norm="l2", axis=1).astype(np.float32)

    # ---------- build FAISS index ----------
    print("Building FAISS index (HNSWFlat on L2)...")
    try:
        import faiss  # noqa
    except Exception as e:
        raise RuntimeError(
            "FAISS import failed. On Windows, use WSL2 or conda for faiss. "
            f"Original error: {e}"
        )

    import faiss
    d = X_stage1.shape[1]
    M = 32  # HNSW connectivity
    index = faiss.IndexHNSWFlat(d, M)  # L2 metric; with normalized vectors, L2 ~ cosine
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 128
    index.add(X_stage1)

    # ---------- save artifacts ----------
    print("Saving artifacts to:", OUT_DIR)

    # df (aligned with matrices)
    df.to_parquet(os.path.join(OUT_DIR, "df.parquet"), index=False)

    # models
    joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.joblib"))
    joblib.dump(tfidf, os.path.join(OUT_DIR, "tfidf.joblib"))
    joblib.dump(svd, os.path.join(OUT_DIR, "svd.joblib"))

    # matrices
    np.save(os.path.join(OUT_DIR, "X_audio.npy"), X_audio)
    np.save(os.path.join(OUT_DIR, "X_meta.npy"), X_meta)
    np.save(os.path.join(OUT_DIR, "X_stage1.npy"), X_stage1)
    sparse.save_npz(os.path.join(OUT_DIR, "X_lyrics.npz"), X_lyrics)

    # faiss index
    faiss.write_index(index, os.path.join(OUT_DIR, "faiss_stage1.index"))

    # metadata
    meta = {
        "rows": int(len(df)),
        "audio_dim": int(X_audio.shape[1]),
        "meta_dim": int(X_meta.shape[1]),
        "stage1_dim": int(X_stage1.shape[1]),
        "tfidf_max_features": TFIDF_MAX_FEATURES,
        "tfidf_min_df": TFIDF_MIN_DF,
        "svd_dim": SVD_DIM,
        "default_k_candidates": DEFAULT_K_CANDIDATES,
        "meta_onehot_dim": int(len(meta_feature_names)),
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("✅ Done. Saved:")
    for name in ["df.parquet", "X_audio.npy", "X_meta.npy", "X_stage1.npy", "X_lyrics.npz",
                 "scaler.joblib", "tfidf.joblib", "svd.joblib", "faiss_stage1.index", "meta.json"]:
        print(" -", name)

if __name__ == "__main__":
    main()
