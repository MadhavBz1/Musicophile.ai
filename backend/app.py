# backend/app.py
import os
import re
import json
import numpy as np
import pandas as pd

from fastapi import FastAPI, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

print("✅ RUNNING MUSICOPHILE BACKEND (FAISS + LYRICS SVD):", __file__)

# ---------------- Config ----------------
HERE = os.path.dirname(os.path.abspath(__file__))

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

# Hard safety: keep workers/threads low (also set in Render env vars)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

app = FastAPI(title="musicophile.ai API (FAISS + Lyrics SVD)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Globals ----------------
df: pd.DataFrame | None = None

X_audio = None
X_meta = None
X_stage1 = None
X_lyrics_svd = None  # (N, dL) normalized, memmap if possible

faiss_index = None
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

def _must(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return path

# ---------------- Loading ----------------
def ensure_loaded():
    """
    Loads artifacts in a Render-friendly way:
    - df.parquet is expected to be compact already (lyrics columns dropped by precompute).
    - big matrices are loaded via mmap (no huge RAM spike).
    - lyrics similarity uses X_lyrics_svd.npy (dense) instead of X_lyrics.npz (huge).
    """
    global df, X_audio, X_meta, X_stage1, X_lyrics_svd, faiss_index, meta_info

    if df is not None and faiss_index is not None and X_stage1 is not None:
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

    # Load only columns we need (precompute should already have dropped lyrics, but keep this safe)
    wanted_cols = ["artist", "song"]
    optional_cols = ["Genre", "emotion", "Album", "Release Date"]
    cols_to_try = wanted_cols + optional_cols
    try:
        df_try = pd.read_parquet(df_path, columns=cols_to_try)
    except Exception:
        df_try = pd.read_parquet(df_path, columns=wanted_cols)

    # Ensure required cols exist
    for c in wanted_cols:
        if c not in df_try.columns:
            raise RuntimeError(f"df.parquet missing required column: {c}")

    df = df_try.reset_index(drop=True)

    # Memmap arrays (float16/float32 both fine)
    X_audio = np.load(_must(os.path.join(ART_DIR, "X_audio.npy")), mmap_mode="r")
    X_meta = np.load(_must(os.path.join(ART_DIR, "X_meta.npy")), mmap_mode="r")
    X_stage1 = np.load(_must(os.path.join(ART_DIR, "X_stage1.npy")), mmap_mode="r")

    lyr_path = os.path.join(ART_DIR, "X_lyrics_svd.npy")
    X_lyrics_svd = np.load(lyr_path, mmap_mode="r") if os.path.exists(lyr_path) else None

    # FAISS
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "FAISS failed to import. Make sure you're deploying on Linux and have faiss-cpu installed.\n"
            f"Original error: {e}"
        )

    import faiss  # type: ignore
    faiss_index = faiss.read_index(_must(os.path.join(ART_DIR, "faiss_stage1.index")))

    # Validate alignment
    n = len(df)
    if X_audio.shape[0] != n or X_meta.shape[0] != n or X_stage1.shape[0] != n:
        raise RuntimeError(
            f"Artifact row mismatch: df={n}, X_audio={X_audio.shape[0]}, X_meta={X_meta.shape[0]}, X_stage1={X_stage1.shape[0]}"
        )
    if X_lyrics_svd is not None and X_lyrics_svd.shape[0] != n:
        raise RuntimeError(
            f"Artifact row mismatch: df={n}, X_lyrics_svd={X_lyrics_svd.shape[0]}"
        )

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
        f"✅ Loaded from {ART_DIR}: rows={len(df)} "
        f"stage1_dim={int(X_stage1.shape[1])} "
        f"lyrics_svd={'yes' if X_lyrics_svd is not None else 'no'}"
    )

# ---------------- Routes ----------------
@app.get("/health")
def health():
    ensure_loaded()
    return {
        "ok": True,
        "ready": True,
        "rows": int(len(df)),
        "artifacts_dir": ART_DIR,
        "meta": meta_info,
        "lyrics_mode": "X_lyrics_svd.npy" if X_lyrics_svd is not None else "none",
        "dtypes": {
            "X_audio": str(getattr(X_audio, "dtype", None)),
            "X_meta": str(getattr(X_meta, "dtype", None)),
            "X_stage1": str(getattr(X_stage1, "dtype", None)),
            "X_lyrics_svd": str(getattr(X_lyrics_svd, "dtype", None)) if X_lyrics_svd is not None else None,
        },
    }

@app.get("/search")
def search(
    q_title: str = Query("", min_length=0),
    q_artist: str = Query("", min_length=0),
    q: str = Query("", min_length=0),
    n: int = 20,
):
    ensure_loaded()

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
    _, I = faiss_index.search(seed_vec, k_cand)
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

    # Dot products because rows are L2-normalized
    audio_sim = np.asarray(np.asarray(X_audio[cand], dtype=np.float32) @ seed_audio, dtype=np.float32)
    meta_sim = np.asarray(np.asarray(X_meta[cand], dtype=np.float32) @ seed_meta, dtype=np.float32)

    # Lyrics similarity: dense SVD embeddings (preferred)
    if X_lyrics_svd is not None:
        seed_ly = np.asarray(X_lyrics_svd[seed_idx], dtype=np.float32)
        cand_ly = np.asarray(X_lyrics_svd[cand], dtype=np.float32)
        lyrics_sim = np.asarray(cand_ly @ seed_ly, dtype=np.float32)
    else:
        lyrics_sim = np.zeros(cand.shape[0], dtype=np.float32)

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
