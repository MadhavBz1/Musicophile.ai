# backend/app.py
import os
import re
import json
import numpy as np
import pandas as pd

from fastapi import FastAPI, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import joblib

print("✅ RUNNING MUSICOPHILE BACKEND (FAISS + TFIDF/SVD LYRICS):", __file__)

# ---------------- Config ----------------
HERE = os.path.dirname(os.path.abspath(__file__))

# Prefer Render persistent disk if present, else fall back to local repo artifacts
DEFAULT_ART_DIR_RENDER = "/data/artifacts"
DEFAULT_ART_DIR_LOCAL = os.path.join(HERE, "artifacts")

ART_DIR = os.environ.get(
    "MUSICOPHILE_ARTIFACTS",
    DEFAULT_ART_DIR_RENDER if os.path.isdir(DEFAULT_ART_DIR_RENDER) else DEFAULT_ART_DIR_LOCAL,
)

ORIGINS_ENV = os.environ.get("MUSICOPHILE_ORIGINS", "")
if ORIGINS_ENV.strip():
    ALLOW_ORIGINS = [o.strip() for o in ORIGINS_ENV.split(",") if o.strip()]
else:
    ALLOW_ORIGINS = [
        "http://localhost:8081",
        "http://127.0.0.1:8081",
        "http://localhost:19006",
        "http://127.0.0.1:19006",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

app = FastAPI(title="musicophile.ai API (FAISS + TFIDF/SVD Lyrics)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Globals (loaded artifacts) ----------------
df: pd.DataFrame | None = None
X_audio: np.ndarray | None = None      # (N, da), normalized float32
X_meta: np.ndarray | None = None       # (N, dm), normalized float32
X_stage1: np.ndarray | None = None     # (N, d1), normalized float32
faiss_index = None

tfidf = None
svd = None

meta_info: dict = {}

# ---------------- Utils ----------------
def _norm(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _build_text_signals(series_norm: pd.Series, qn: str):
    q_esc = re.escape(qn)
    word_re = rf"\b{q_esc}\b"

    exact = (series_norm == qn)
    starts = series_norm.str.startswith(qn, na=False)
    word_contains = series_norm.str.contains(word_re, regex=True, na=False)
    any_contains = series_norm.str.contains(q_esc, regex=True, na=False)
    return exact, starts, word_contains, any_contains

def _l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # X: (n, d)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return X / norms

def _pick_lyrics_column(frame: pd.DataFrame) -> str | None:
    if "lyrics_clean" in frame.columns:
        return "lyrics_clean"
    if "lyrics" in frame.columns:
        return "lyrics"
    return None

def ensure_loaded():
    """
    Loads:
      - df.parquet
      - X_audio.npy, X_meta.npy, X_stage1.npy (mmap)
      - faiss_stage1.index
      - tfidf.joblib + svd.joblib  (for lyrics similarity on-the-fly)
    """
    global df, X_audio, X_meta, X_stage1, faiss_index, tfidf, svd, meta_info

    if df is not None and faiss_index is not None and tfidf is not None and svd is not None:
        return

    if not os.path.isdir(ART_DIR):
        raise FileNotFoundError(f"Artifacts directory not found: {ART_DIR}")

    meta_path = os.path.join(ART_DIR, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_info = json.load(f)

    df_path = os.path.join(ART_DIR, "df.parquet")
    if not os.path.exists(df_path):
        raise FileNotFoundError(f"Missing df.parquet at: {df_path}")
    df = pd.read_parquet(df_path)

    # Memmap numeric arrays so RAM stays low
    X_audio_path = os.path.join(ART_DIR, "X_audio.npy")
    X_meta_path = os.path.join(ART_DIR, "X_meta.npy")
    X_stage1_path = os.path.join(ART_DIR, "X_stage1.npy")

    for p in (X_audio_path, X_meta_path, X_stage1_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required artifact: {p}")

    X_audio = np.load(X_audio_path, mmap_mode="r")
    X_meta = np.load(X_meta_path, mmap_mode="r")
    X_stage1 = np.load(X_stage1_path, mmap_mode="r")

    # TF-IDF + SVD for lyrics (lightweight)
    tfidf_path = os.path.join(ART_DIR, "tfidf.joblib")
    svd_path = os.path.join(ART_DIR, "svd.joblib")
    if not os.path.exists(tfidf_path) or not os.path.exists(svd_path):
        raise FileNotFoundError(
            f"Missing tfidf/svd models. Expected:\n- {tfidf_path}\n- {svd_path}"
        )

    tfidf = joblib.load(tfidf_path)
    svd = joblib.load(svd_path)

    # FAISS index
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "FAISS failed to import. Render Linux should be OK with faiss-cpu.\n"
            f"Original error: {e}"
        )
    import faiss  # type: ignore
    index_path = os.path.join(ART_DIR, "faiss_stage1.index")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Missing faiss index at: {index_path}")
    faiss_index = faiss.read_index(index_path)

    # Validate alignment
    n = len(df)
    if X_audio.shape[0] != n or X_meta.shape[0] != n or X_stage1.shape[0] != n:
        raise RuntimeError("Artifact row mismatch: df and matrices are not aligned.")

    if _pick_lyrics_column(df) is None:
        print("⚠️ Warning: No lyrics column found (lyrics_clean/lyrics). Lyrics similarity will be 0.")

# ---------------- Schemas ----------------
class RecommendRequest(BaseModel):
    seed_id: int
    w_audio: float = 0.4
    w_lyrics: float = 0.4
    w_meta: float = 0.2
    diversify_genre: float = 0.0
    top_k: int = 10
    k_candidates: int = 2000

    artist_mode: str = "different"  # "same" or "different"

# ---------------- Startup ----------------
@app.on_event("startup")
def on_startup():
    ensure_loaded()
    print(
        f"✅ Loaded artifacts from {ART_DIR}: rows={len(df)} "
        f"stage1_dim={X_stage1.shape[1]} "
        f"audio_dim={X_audio.shape[1]} meta_dim={X_meta.shape[1]}"
    )

# ---------------- Routes ----------------
@app.get("/health")
def health():
    ready = df is not None and faiss_index is not None and tfidf is not None and svd is not None
    return {
        "ok": True,
        "ready": ready,
        "rows": 0 if df is None else int(len(df)),
        "artifacts_dir": ART_DIR,
        "meta": meta_info,
        "lyrics_mode": "tfidf+svd(on_the_fly)",
    }

@app.get("/search")
def search(
    q_title: str = Query("", min_length=0),
    q_artist: str = Query("", min_length=0),
    q: str = Query("", min_length=0),
    n: int = 20,
):
    ensure_loaded()
    if df is None:
        return {"results": []}

    qt = _norm(q_title)
    qa = _norm(q_artist)
    q_legacy = _norm(q)

    use_split = bool(qt or qa)
    if not use_split and not q_legacy:
        return {"results": []}

    song_raw = df["song"].astype(str)
    artist_raw = df["artist"].astype(str)
    song = song_raw.map(_norm)
    artist = artist_raw.map(_norm)

    if use_split:
        if qa:
            a_exact, a_starts, a_word, a_any = _build_text_signals(artist, qa)
            a_score = (
                1200 * a_exact.astype(int)
                + 700 * a_starts.astype(int)
                + 450 * a_word.astype(int)
                + 250 * a_any.astype(int)
            ).astype(int)
            gate = (a_any | a_word | a_starts | a_exact).values
        else:
            a_score = np.zeros(len(df), dtype=int)
            gate = np.ones(len(df), dtype=bool)

        if qt:
            t_exact, t_starts, t_word, t_any = _build_text_signals(song, qt)
            t_score = (
                1400 * t_exact.astype(int)
                + 650 * t_starts.astype(int)
                + 350 * t_word.astype(int)
                + 160 * t_any.astype(int)
            ).astype(int)
        else:
            t_score = np.zeros(len(df), dtype=int)

        score = (t_score + a_score).astype(int)
        score = np.where(gate, score, 0)

    else:
        a_exact, a_starts, a_word, a_any = _build_text_signals(artist, q_legacy)
        t_exact, t_starts, t_word, t_any = _build_text_signals(song, q_legacy)

        score = (
            1000 * t_exact.astype(int)
            + 450 * t_starts.astype(int)
            + 260 * t_word.astype(int)
            + 120 * t_any.astype(int)
            + 500 * a_starts.astype(int)
            + 260 * (a_word | a_any).astype(int)
        ).astype(int)

    hits = np.where(score > 0)[0]
    if hits.size == 0:
        return {"results": []}

    title_len = song_raw.str.len().fillna(10**9).values
    order = np.lexsort((title_len[hits], -score[hits]))
    top = hits[order][:n]

    out = []
    for i in top:
        out.append({
            "id": int(i),
            "song": df.at[i, "song"],
            "artist": df.at[i, "artist"],
            "genre": df.at[i, "Genre"] if "Genre" in df.columns else None,
            "emotion": df.at[i, "emotion"] if "emotion" in df.columns else None,
            "search_score": int(score[i]),
        })
    return {"results": out}

@app.post("/recommend")
def recommend(req: RecommendRequest):
    ensure_loaded()
    if df is None:
        return {"error": "dataset not loaded"}

    n = len(df)
    seed_idx = int(req.seed_id)
    if seed_idx < 0 or seed_idx >= n:
        return {"error": "seed_id out of range"}

    artist_mode = str(req.artist_mode).lower().strip()
    if artist_mode not in ("same", "different"):
        return {"error": "artist_mode must be 'same' or 'different'"}

    # normalize weights
    s = float(req.w_audio + req.w_lyrics + req.w_meta)
    if s <= 0:
        w_audio, w_lyrics, w_meta = 0.4, 0.4, 0.2
    else:
        w_audio, w_lyrics, w_meta = req.w_audio / s, req.w_lyrics / s, req.w_meta / s

    # -------- Stage 1: FAISS candidates --------
    k_cand = int(req.k_candidates)
    k_cand = max(k_cand, int(req.top_k) + 50)
    k_cand = min(k_cand, n)

    seed_vec = np.asarray(X_stage1[seed_idx], dtype=np.float32).reshape(1, -1)
    D, I = faiss_index.search(seed_vec, k_cand)
    cand = I.ravel().astype(int)

    cand = cand[(cand >= 0) & (cand < n)]
    cand = cand[cand != seed_idx]
    if cand.size == 0:
        return {"error": "no candidates found"}

    # -------- Artist filter --------
    seed_artist = str(df.at[seed_idx, "artist"])
    if artist_mode == "same":
        cand = cand[df.loc[cand, "artist"].astype(str).values == seed_artist]
    else:
        cand = cand[df.loc[cand, "artist"].astype(str).values != seed_artist]

    if cand.size == 0:
        return {"error": "no candidates after artist filter"}

    # -------- Stage 2: rerank --------
    seed_audio = np.asarray(X_audio[seed_idx], dtype=np.float32)
    seed_meta = np.asarray(X_meta[seed_idx], dtype=np.float32)

    # audio/meta are assumed normalized so dot = cosine
    audio_sim = np.asarray(X_audio[cand] @ seed_audio, dtype=np.float32)
    meta_sim = np.asarray(X_meta[cand] @ seed_meta, dtype=np.float32)

    # Lyrics similarity via TF-IDF -> SVD on-the-fly (seed + candidates only)
    lyrics_col = _pick_lyrics_column(df)
    if lyrics_col is None:
        lyrics_sim = np.zeros(cand.shape[0], dtype=np.float32)
    else:
        seed_text = "" if pd.isna(df.at[seed_idx, lyrics_col]) else str(df.at[seed_idx, lyrics_col])
        cand_texts = df.loc[cand, lyrics_col].fillna("").astype(str).tolist()

        # transform
        X_seed_tfidf = tfidf.transform([seed_text])
        X_cand_tfidf = tfidf.transform(cand_texts)

        seed_ly = svd.transform(X_seed_tfidf).astype(np.float32)   # (1, d)
        cand_ly = svd.transform(X_cand_tfidf).astype(np.float32)   # (k, d)

        # normalize then dot-product
        seed_ly = _l2_normalize_rows(seed_ly)
        cand_ly = _l2_normalize_rows(cand_ly)
        lyrics_sim = (cand_ly @ seed_ly[0]).astype(np.float32)

    score = (w_audio * audio_sim) + (w_lyrics * lyrics_sim) + (w_meta * meta_sim)

    if req.diversify_genre > 0 and "Genre" in df.columns:
        seed_genre = df.at[seed_idx, "Genre"]
        same_genre = (df.loc[cand, "Genre"].values == seed_genre).astype(np.float32)
        score = score - (float(req.diversify_genre) * same_genre)

    top_k = int(req.top_k)
    top_k = max(1, min(top_k, cand.size))

    order = np.argsort(score)[::-1][:top_k]
    top_idx = cand[order]

    seed = {
        "id": int(seed_idx),
        "song": df.at[seed_idx, "song"],
        "artist": df.at[seed_idx, "artist"],
        "genre": df.at[seed_idx, "Genre"] if "Genre" in df.columns else None,
        "emotion": df.at[seed_idx, "emotion"] if "emotion" in df.columns else None,
    }

    recs = []
    for j, i in enumerate(top_idx):
        recs.append({
            "id": int(i),
            "song": df.at[i, "song"],
            "artist": df.at[i, "artist"],
            "genre": df.at[i, "Genre"] if "Genre" in df.columns else None,
            "emotion": df.at[i, "emotion"] if "emotion" in df.columns else None,
            "score": float(score[order[j]]),
            "audio_sim": float(audio_sim[order[j]]),
            "lyric_sim": float(lyrics_sim[order[j]]),
            "meta_sim": float(meta_sim[order[j]]),
        })

    return {"seed": seed, "recommendations": recs}
