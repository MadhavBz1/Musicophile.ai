# backend/app.py
import os
import re
import json
import numpy as np
import pandas as pd

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics.pairwise import cosine_similarity  # still used as fallback (optional)
from scipy import sparse
import joblib

print("✅ RUNNING FAISS ARTIFACT BACKEND:", __file__)

# ---------------- Config ----------------
HERE = os.path.dirname(os.path.abspath(__file__))

# Render Persistent Disk: set MUSICOPHILE_ARTIFACTS=/data/artifacts
# Local default: backend/artifacts
ART_DIR = os.environ.get("MUSICOPHILE_ARTIFACTS", os.path.join(HERE, "artifacts"))

# Load at startup?
EAGER_LOAD = os.environ.get("MUSICOPHILE_EAGER_LOAD", "true").strip().lower() in ("1", "true", "yes", "y")

# CORS
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

app = FastAPI(title="musicophile.ai API (FAISS + TFIDF Lyrics)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Globals ----------------
df: pd.DataFrame | None = None
X_audio: np.ndarray | None = None      # (N, da), normalized float32
X_meta: np.ndarray | None = None       # (N, dm), normalized float32
X_stage1: np.ndarray | None = None     # (N, d1), normalized float32
faiss_index = None
meta_info: dict = {}

# Lyrics models (small, safe)
tfidf_model = None
svd_model = None
lyrics_col: str | None = None

# Optional cache for seed lyric vectors (speeds repeated requests)
_seed_lyrics_cache: dict[int, np.ndarray] = {}

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

def _missing(path: str) -> str:
    return f"Missing required artifact: {path}"

def _pick_lyrics_column(frame: pd.DataFrame) -> str | None:
    """
    Try to detect the lyrics text column name.
    Common possibilities: 'lyrics', 'Lyrics', 'lyric', 'text', etc.
    """
    candidates = ["lyrics", "Lyrics", "lyric", "Lyric", "text", "Text", "clean_lyrics", "Clean Lyrics"]
    for c in candidates:
        if c in frame.columns:
            return c

    # Heuristic: any column containing 'lyric'
    for c in frame.columns:
        if "lyric" in str(c).lower():
            return c

    return None

def _l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return X / norms

def _l2_normalize_vec(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v
    return v / n

def ensure_loaded(force: bool = False):
    global df, X_audio, X_meta, X_stage1, faiss_index, meta_info
    global tfidf_model, svd_model, lyrics_col

    if not force and df is not None and faiss_index is not None:
        return

    if not os.path.exists(ART_DIR):
        raise FileNotFoundError(f"Artifacts directory not found: {ART_DIR}")

    # Optional meta.json
    meta_path = os.path.join(ART_DIR, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_info = json.load(f)
    else:
        meta_info = {}

    # Required artifacts
    df_path = os.path.join(ART_DIR, "df.parquet")
    if not os.path.exists(df_path):
        raise FileNotFoundError(_missing(df_path))
    df = pd.read_parquet(df_path)

    # Matrices required
    audio_path = os.path.join(ART_DIR, "X_audio.npy")
    meta_mat_path = os.path.join(ART_DIR, "X_meta.npy")
    stage1_path = os.path.join(ART_DIR, "X_stage1.npy")
    faiss_path = os.path.join(ART_DIR, "faiss_stage1.index")

    for p in (audio_path, meta_mat_path, stage1_path, faiss_path):
        if not os.path.exists(p):
            raise FileNotFoundError(_missing(p))

    X_audio = np.load(audio_path, mmap_mode="r")
    X_meta = np.load(meta_mat_path, mmap_mode="r")
    X_stage1 = np.load(stage1_path, mmap_mode="r")

    # FAISS
    try:
        import faiss  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "FAISS failed to import. Ensure requirements include faiss-cpu.\n"
            f"Original error: {e}"
        )
    import faiss
    faiss_index = faiss.read_index(faiss_path)

    # Align check
    n = len(df)
    if X_audio.shape[0] != n or X_meta.shape[0] != n or X_stage1.shape[0] != n:
        raise RuntimeError("Artifact row mismatch: df and matrices are not aligned.")

    # Lyrics via TFIDF + SVD (lightweight)
    tfidf_path = os.path.join(ART_DIR, "tfidf.joblib")
    svd_path = os.path.join(ART_DIR, "svd.joblib")

    if os.path.exists(tfidf_path) and os.path.exists(svd_path):
        tfidf_model = joblib.load(tfidf_path)
        svd_model = joblib.load(svd_path)
        lyrics_col = _pick_lyrics_column(df)
        if lyrics_col is None:
            print("⚠️ tfidf/svd found, but no lyrics column detected in df; lyrics_sim will be 0.")
        else:
            print(f"✅ Lyrics enabled via TFIDF+SVD using df column: {lyrics_col}")
    else:
        tfidf_model = None
        svd_model = None
        lyrics_col = None
        print("⚠️ tfidf.joblib/svd.joblib not found; lyrics_sim will be 0.")

    # Clear cache on reload
    _seed_lyrics_cache.clear()

def lyrics_similarity(seed_idx: int, cand: np.ndarray) -> np.ndarray:
    """
    Compute lyrics similarity using tfidf -> svd -> cosine (dot product after L2 normalization).
    Returns array shape (len(cand),) float32
    """
    if df is None or tfidf_model is None or svd_model is None or lyrics_col is None:
        return np.zeros(len(cand), dtype=np.float32)

    # Seed vector cache
    if seed_idx in _seed_lyrics_cache:
        seed_vec = _seed_lyrics_cache[seed_idx]
    else:
        seed_text = str(df.at[seed_idx, lyrics_col] if lyrics_col in df.columns else "")
        seed_tfidf = tfidf_model.transform([seed_text])          # sparse
        seed_vec = svd_model.transform(seed_tfidf).astype(np.float32)  # (1, d)
        seed_vec = _l2_normalize_rows(seed_vec)[0]
        _seed_lyrics_cache[seed_idx] = seed_vec

    # Candidate vectors
    texts = df.loc[cand, lyrics_col].fillna("").astype(str).tolist()
    cand_tfidf = tfidf_model.transform(texts)  # sparse
    cand_vecs = svd_model.transform(cand_tfidf).astype(np.float32)  # (k, d)
    cand_vecs = _l2_normalize_rows(cand_vecs)

    # cosine similarity = dot product since both normalized
    sim = cand_vecs @ seed_vec
    return sim.astype(np.float32)

# ---------------- Schemas ----------------
class RecommendRequest(BaseModel):
    seed_id: int = Field(..., description="Row index of the seed song in df")
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
    if EAGER_LOAD:
        ensure_loaded()
        print(f"✅ Loaded artifacts from {ART_DIR}: rows={len(df)} stage1_dim={X_stage1.shape[1]}")
    else:
        print("ℹ️ MUSICOPHILE_EAGER_LOAD=false; artifacts will load on first request.")

# ---------------- Routes ----------------
@app.get("/health")
def health():
    ready = df is not None and faiss_index is not None
    return {
        "ok": True,
        "ready": ready,
        "rows": 0 if df is None else int(len(df)),
        "artifacts_dir": ART_DIR,
        "meta": meta_info,
        "lyrics_mode": "tfidf+svd" if (tfidf_model is not None and svd_model is not None) else "disabled",
        "lyrics_col": lyrics_col,
        "eager_load": EAGER_LOAD,
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
        raise HTTPException(status_code=500, detail="dataset not loaded")

    n_rows = len(df)
    seed_idx = int(req.seed_id)
    if seed_idx < 0 or seed_idx >= n_rows:
        raise HTTPException(status_code=400, detail="seed_id out of range")

    artist_mode = str(req.artist_mode).lower().strip()
    if artist_mode not in ("same", "different"):
        raise HTTPException(status_code=400, detail="artist_mode must be 'same' or 'different'")

    # Normalize weights
    s = float(req.w_audio + req.w_lyrics + req.w_meta)
    if s <= 0:
        w_audio, w_lyrics, w_meta = 0.4, 0.4, 0.2
    else:
        w_audio, w_lyrics, w_meta = req.w_audio / s, req.w_lyrics / s, req.w_meta / s

    # -------- Stage 1: FAISS candidates --------
    k_cand = int(req.k_candidates)
    k_cand = max(k_cand, int(req.top_k) + 50)
    k_cand = min(k_cand, n_rows)

    seed_vec = np.asarray(X_stage1[seed_idx], dtype=np.float32).reshape(1, -1)
    D, I = faiss_index.search(seed_vec, k_cand)
    cand = I.ravel().astype(int)

    cand = cand[(cand >= 0) & (cand < n_rows)]
    cand = cand[cand != seed_idx]
    if cand.size == 0:
        raise HTTPException(status_code=404, detail="no candidates found")

    # -------- Artist filter --------
    seed_artist = str(df.at[seed_idx, "artist"])
    if artist_mode == "same":
        cand = cand[df.loc[cand, "artist"].astype(str).values == seed_artist]
    else:
        cand = cand[df.loc[cand, "artist"].astype(str).values != seed_artist]

    if cand.size == 0:
        raise HTTPException(status_code=404, detail="no candidates after artist filter")

    # -------- Stage 2: rerank --------
    seed_audio = np.asarray(X_audio[seed_idx], dtype=np.float32)
    seed_meta = np.asarray(X_meta[seed_idx], dtype=np.float32)

    # since normalized, dot = cosine
    audio_sim = np.asarray(X_audio[cand] @ seed_audio, dtype=np.float32)
    meta_sim = np.asarray(X_meta[cand] @ seed_meta, dtype=np.float32)

    # Lyrics sim via TFIDF+SVD
    lyrics_sim = lyrics_similarity(seed_idx, cand)

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
