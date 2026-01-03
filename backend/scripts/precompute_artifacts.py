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

META_SVD_DIM = 128
LYRICS_SVD_DIM = 128

# Save knobs
SAVE_X_LYRICS_NPZ = False   # HUGE. keep False for Render
SAVE_FLOAT16 = True         # saves lots of disk; fine for similarity scoring
PARQUET_COMPRESSION = "zstd"  # "zstd" (best), or "snappy"

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

def l2_normalize_dense(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return X / norms

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

    # lyrics clean
    if "lyrics" not in df.columns:
        raise ValueError("Missing lyrics column. Expected after rename: 'lyrics'")

    df["lyrics_clean"] = df["lyrics"].apply(lambda x: clean_lyrics(x, MAX_LYRIC_CHARS))

    # loudness parse
    if "loudness" in df.columns:
        df["loudness"] = df["loudness"].apply(parse_loudness)
    else:
        df["loudness"] = np.nan

    # numeric audio
    for col in AUDIO_FEATURES:
        if col == "loudness":
            continue
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    # keep only rows usable for recs
    df = df.dropna(subset=["song", "artist", "lyrics_clean", "Tempo", "Energy", "Danceability"]).reset_index(drop=True)

    # fill remaining audio NaNs
    for col in AUDIO_FEATURES:
        df[col] = df[col].fillna(df[col].median())

    # ---------- build audio ----------
    print("Building audio matrix...")
    scaler_audio = StandardScaler()
    X_audio = scaler_audio.fit_transform(df[AUDIO_FEATURES].values).astype(np.float32)
    X_audio = normalize(X_audio, norm="l2", axis=1).astype(np.float32)

    # ---------- build meta (one-hot -> SVD -> normalized) ----------
    print("Building meta one-hot...")
    X_meta_oh_df = pd.get_dummies(df[META_COLS].fillna("UNK"), drop_first=False)
    meta_feature_names = X_meta_oh_df.columns.tolist()
    X_meta_oh = X_meta_oh_df.values.astype(np.float32)

    print(f"Compressing meta with SVD to {META_SVD_DIM} dims...")
    svd_meta = TruncatedSVD(n_components=META_SVD_DIM, random_state=42)
    X_meta = svd_meta.fit_transform(X_meta_oh).astype(np.float32)
    X_meta = normalize(X_meta, norm="l2", axis=1).astype(np.float32)

    # ---------- build lyrics TF-IDF ----------
    print("Building TF-IDF lyrics...")
    tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=TFIDF_MIN_DF,
        max_features=TFIDF_MAX_FEATURES,
        sublinear_tf=True
    )
    X_tfidf = tfidf.fit_transform(df["lyrics_clean"])

    # ---------- lyrics SVD embeddings (dense + normalized) ----------
    print(f"Compressing lyrics TF-IDF with SVD to {LYRICS_SVD_DIM} dims...")
    svd_lyrics = TruncatedSVD(n_components=LYRICS_SVD_DIM, random_state=42)
    X_lyrics_svd = svd_lyrics.fit_transform(X_tfidf).astype(np.float32)
    X_lyrics_svd = l2_normalize_dense(X_lyrics_svd).astype(np.float32)

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

    # ---------- SAVE SMALL df.parquet (overwrite) ----------
    # KEY FIX: drop lyrics + lyrics_clean so df.parquet is production-safe
    print("Preparing compact df.parquet (dropping lyrics columns)...")
    drop_cols = [c for c in ["lyrics", "lyrics_clean"] if c in df.columns]
    df_compact = df.drop(columns=drop_cols)

    df_path = os.path.join(OUT_DIR, "df.parquet")
    df_compact.to_parquet(df_path, index=False, compression=PARQUET_COMPRESSION)

    # ---------- save artifacts ----------
    print("Saving artifacts to:", OUT_DIR)

    # models
    joblib.dump(scaler_audio, os.path.join(OUT_DIR, "scaler.joblib"))
    joblib.dump(tfidf, os.path.join(OUT_DIR, "tfidf.joblib"))
    joblib.dump(svd_lyrics, os.path.join(OUT_DIR, "svd.joblib"))      # lyrics SVD
    joblib.dump(svd_meta, os.path.join(OUT_DIR, "svd_meta.joblib"))   # meta SVD (new)

    # matrices (float16 where safe)
    if SAVE_FLOAT16:
        np.save(os.path.join(OUT_DIR, "X_audio.npy"), X_audio.astype(np.float16))
        np.save(os.path.join(OUT_DIR, "X_meta.npy"), X_meta.astype(np.float16))
        np.save(os.path.join(OUT_DIR, "X_lyrics_svd.npy"), X_lyrics_svd.astype(np.float16))
        np.save(os.path.join(OUT_DIR, "X_stage1.npy"), X_stage1.astype(np.float32))  # keep float32 for FAISS
    else:
        np.save(os.path.join(OUT_DIR, "X_audio.npy"), X_audio.astype(np.float32))
        np.save(os.path.join(OUT_DIR, "X_meta.npy"), X_meta.astype(np.float32))
        np.save(os.path.join(OUT_DIR, "X_lyrics_svd.npy"), X_lyrics_svd.astype(np.float32))
        np.save(os.path.join(OUT_DIR, "X_stage1.npy"), X_stage1.astype(np.float32))

    # OPTIONAL: save the huge sparse TF-IDF (not recommended on Render)
    if SAVE_X_LYRICS_NPZ:
        sparse.save_npz(os.path.join(OUT_DIR, "X_lyrics.npz"), X_tfidf)

    # faiss index
    faiss.write_index(index, os.path.join(OUT_DIR, "faiss_stage1.index"))

    # metadata
    meta = {
        "rows": int(len(df_compact)),
        "audio_dim": int(X_audio.shape[1]),
        "meta_dim": int(X_meta.shape[1]),
        "lyrics_svd_dim": int(X_lyrics_svd.shape[1]),
        "stage1_dim": int(X_stage1.shape[1]),
        "tfidf_max_features": TFIDF_MAX_FEATURES,
        "tfidf_min_df": TFIDF_MIN_DF,
        "meta_svd_dim": META_SVD_DIM,
        "lyrics_svd_dim": LYRICS_SVD_DIM,
        "default_k_candidates": DEFAULT_K_CANDIDATES,
        "meta_onehot_dim": int(len(meta_feature_names)),
        "parquet_compression": PARQUET_COMPRESSION,
        "float16_saved": bool(SAVE_FLOAT16),
        "saved_sparse_tfidf_npz": bool(SAVE_X_LYRICS_NPZ),
        "notes": "df.parquet saved without lyrics columns; use X_lyrics_svd.npy for lyrics similarity",
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("✅ Done. Saved:")
    saved = [
        "df.parquet",
        "X_audio.npy",
        "X_meta.npy",
        "X_stage1.npy",
        "X_lyrics_svd.npy",
        "scaler.joblib",
        "tfidf.joblib",
        "svd.joblib",
        "svd_meta.joblib",
        "faiss_stage1.index",
        "meta.json",
    ]
    if SAVE_X_LYRICS_NPZ:
        saved.append("X_lyrics.npz")

    for name in saved:
        p = os.path.join(OUT_DIR, name)
        if os.path.exists(p):
            mb = os.path.getsize(p) / 1024 / 1024
            print(f" - {name} ({mb:.2f} MB)")
        else:
            print(" -", name, "(missing?)")

if __name__ == "__main__":
    main()
