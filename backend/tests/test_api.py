import os
import json
import string
import pytest
from textwrap import indent
from fastapi.testclient import TestClient

from backend.app import app

client = TestClient(app)


# ---------------------------
# Pretty-print helpers
# ---------------------------

def pretty(obj) -> str:
    """Pretty-print JSON for readable pytest failure output."""
    try:
        return indent(json.dumps(obj, indent=2, ensure_ascii=False), "  ")
    except Exception:
        return indent(str(obj), "  ")


def assert_ok(r, context: str = ""):
    """Assert HTTP 200 and pretty-print response if it fails."""
    if r.status_code == 200:
        return
    try:
        body = pretty(r.json())
    except Exception:
        body = indent(r.text, "  ")

    raise AssertionError(
        f"{context}\n"
        f"Status: {r.status_code}\n"
        f"Response:\n{body}"
    )


# ---------------------------
# Helpers
# ---------------------------

def _norm(s: str) -> str:
    return (s or "").strip().casefold()

def _get_any_seed_id(query: str = "drake") -> int:
    r = client.get("/search", params={"q": query})
    assert_ok(r, context=f"GET /search?q={query!r}")

    data = r.json()
    assert "results" in data and isinstance(data["results"], list) and len(data["results"]) > 0, (
        f"No results returned for query={query!r}\nResponse:\n{pretty(data)}"
    )

    first = data["results"][0]
    assert "id" in first, (
        "Search results must include an 'id' field\n"
        f"First result:\n{pretty(first)}"
    )
    return first["id"]

def _collect_seed_ids(max_seeds: int = 200) -> list[int]:
    """
    Collect lots of unique seed_ids by searching many short queries.
    This approximates broad coverage without needing an 'all tracks' endpoint.
    """
    seeds: list[int] = []
    seen = set()

    queries = list(string.ascii_lowercase) + ["drake", "love", "the", "a", "i"]

    for q in queries:
        r = client.get("/search", params={"q": q})
        if r.status_code != 200:
            continue

        results = r.json().get("results", [])
        for item in results:
            sid = item.get("id")
            if isinstance(sid, int) and sid not in seen:
                seen.add(sid)
                seeds.append(sid)
                if len(seeds) >= max_seeds:
                    return seeds

    return seeds


# ---------------------------
# Search tests
# ---------------------------

def test_search_basic():
    r = client.get("/search", params={"q": "drake"})
    assert_ok(r, context="GET /search?q='drake'")

    data = r.json()
    assert isinstance(data, dict), f"Expected dict\nResponse:\n{pretty(data)}"
    assert "results" in data, f"Missing 'results'\nResponse:\n{pretty(data)}"
    assert isinstance(data["results"], list), f"'results' must be a list\nResponse:\n{pretty(data)}"

    if data["results"]:
        first = data["results"][0]
        assert isinstance(first, dict), f"Expected dict item\n{pretty(first)}"
        assert "artist" in first, f"Missing 'artist'\n{pretty(first)}"
        assert "id" in first, f"Missing 'id'\n{pretty(first)}"

def test_search_empty_query():
    r = client.get("/search", params={"q": ""})
    assert r.status_code in (200, 422), (
        f"Unexpected status for empty query\n"
        f"Status: {r.status_code}\n"
        f"Response:\n{pretty(r.json()) if r.headers.get('content-type','').startswith('application/json') else r.text}"
    )

def test_search_has_no_duplicate_ids_single_query():
    r = client.get("/search", params={"q": "drake"})
    assert_ok(r, context="GET /search?q='drake' (duplicate ID check)")

    results = r.json().get("results", [])
    ids = [item.get("id") for item in results]

    assert None not in ids, (
        "Every search result should have an id\n"
        f"Results:\n{pretty(results)}"
    )
    assert len(ids) == len(set(ids)), (
        "Duplicate track IDs found in search results\n"
        f"IDs: {ids}\n"
        f"Results:\n{pretty(results)}"
    )

def test_search_has_no_duplicate_artist_title_pairs_single_query():
    r = client.get("/search", params={"q": "drake"})
    assert_ok(r, context="GET /search?q='drake' (artist/title check)")

    results = r.json().get("results", [])

    pairs = []
    for item in results:
        artist = _norm(item.get("artist"))
        title = _norm(item.get("song") or item.get("title"))
        pairs.append((artist, title))

    assert len(pairs) == len(set(pairs)), (
        "Duplicate (artist, title) pairs found in search results\n"
        f"Pairs:\n  {pairs}\n"
        f"Results:\n{pretty(results)}"
    )

def test_search_no_duplicate_ids_across_many_queries():
    queries = list(string.ascii_lowercase) + ["drake", "love", "the"]
    for q in queries:
        r = client.get("/search", params={"q": q})
        assert_ok(r, context=f"GET /search?q={q!r}")

        results = r.json().get("results", [])
        ids = [item.get("id") for item in results if item.get("id") is not None]
        assert len(ids) == len(set(ids)), (
            f"Duplicate search IDs found for query={q!r}\n"
            f"Results:\n{pretty(results)}"
        )


# ---------------------------
# Recommend tests
# ---------------------------

def test_recommend_requires_seed_id():
    r = client.post("/recommend", json={"artist": "Drake", "song": "God's Plan"})
    assert r.status_code == 422, (
        f"Expected 422 when seed_id missing\n"
        f"Status: {r.status_code}\n"
        f"Response:\n{pretty(r.json()) if r.headers.get('content-type','').startswith('application/json') else r.text}"
    )
    assert "seed_id" in r.text

def test_recommend_happy_path_with_seed_id():
    seed_id = _get_any_seed_id("drake")

    r = client.post("/recommend", json={"seed_id": seed_id})
    assert_ok(r, context=f"POST /recommend seed_id={seed_id}")

    data = r.json()
    assert isinstance(data, dict), f"Expected dict\n{pretty(data)}"
    assert "seed" in data, f"Missing 'seed'\n{pretty(data)}"
    assert "recommendations" in data, f"Missing 'recommendations'\n{pretty(data)}"
    assert isinstance(data["recommendations"], list), f"'recommendations' must be list\n{pretty(data)}"

    if data["recommendations"]:
        assert isinstance(data["recommendations"][0], dict), (
            f"Expected dict items\n{pretty(data['recommendations'][0])}"
        )

def test_recommend_invalid_seed_id():
    r = client.post("/recommend", json={"seed_id": -1})
    assert r.status_code in (200, 400, 404), (
        f"Unexpected status for invalid seed_id\n"
        f"Status: {r.status_code}\n"
        f"Response:\n{pretty(r.json()) if r.headers.get('content-type','').startswith('application/json') else r.text}"
    )

def test_recommendations_have_no_duplicate_ids_and_exclude_seed_single_seed():
    seed_id = _get_any_seed_id("drake")
    r = client.post("/recommend", json={"seed_id": seed_id})
    assert_ok(r, context=f"POST /recommend seed_id={seed_id}")

    data = r.json()
    recs = data["recommendations"]

    ids = [item.get("id") for item in recs]
    assert None not in ids, (
        f"Missing id in recommendations\n"
        f"Recs:\n{pretty(recs)}"
    )
    assert seed_id not in ids, (
        f"Seed track appeared in recommendations\n"
        f"Recs:\n{pretty(recs)}"
    )
    assert len(ids) == len(set(ids)), (
        f"Duplicate track IDs in recommendations\n"
        f"IDs: {ids}\n"
        f"Recs:\n{pretty(recs)}"
    )

def test_recommendations_no_duplicates_for_many_seeds():
    """
    Broad-coverage test: run recommend for many seed IDs.
    Control runtime with env var:
      SEED_TEST_COUNT=100  (default)
    """
    max_seeds = int(os.getenv("SEED_TEST_COUNT", "100"))
    seed_ids = _collect_seed_ids(max_seeds=max_seeds)

    assert len(seed_ids) > 0, (
        "No seed_ids collected from /search\n"
        f"SEED_TEST_COUNT={max_seeds}"
    )

    for seed_id in seed_ids:
        r = client.post("/recommend", json={"seed_id": seed_id})
        assert_ok(r, context=f"POST /recommend seed_id={seed_id}")

        data = r.json()
        recs = data["recommendations"]

        ids = [item.get("id") for item in recs]
        assert None not in ids, (
            f"Missing id in recs for seed_id={seed_id}\n"
            f"Recs:\n{pretty(recs)}"
        )
        assert seed_id not in ids, (
            f"Seed appeared in its own recs for seed_id={seed_id}\n"
            f"Recs:\n{pretty(recs)}"
        )
        assert len(ids) == len(set(ids)), (
            f"Duplicate recommendation IDs for seed_id={seed_id}\n"
            f"IDs: {ids}\n"
            f"Recs:\n{pretty(recs)}"
        )
