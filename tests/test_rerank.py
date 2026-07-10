"""
Tests for geometric verification (search_verified) in vlad_matcher.

Uses synthetic images and monkeypatched reference fetching so no network,
vocabulary, or vectors repository is needed.
"""

import cv2
import numpy as np
import pytest

import vlad_matcher
from vlad_matcher import VLADCardSearch


def make_card(seed):
    """Generate a synthetic textured card image (deterministic per seed)."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (558, 400, 3), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (9, 9), 0)
    # Add some strong structure so SIFT finds stable keypoints
    for _ in range(30):
        pt1 = tuple(rng.integers(0, 380, 2).tolist())
        pt2 = (pt1[0] + int(rng.integers(10, 60)), pt1[1] + int(rng.integers(10, 60)))
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.rectangle(img, pt1, pt2, color, -1)
    return img


@pytest.fixture
def matcher():
    """A VLADCardSearch that skips __init__ side effects (no repo sync/load)."""
    m = VLADCardSearch.__new__(VLADCardSearch)
    m.sift = cv2.SIFT_create()
    m._ref_features = {}
    return m


def test_extract_match_features(matcher):
    pts, des = matcher._extract_match_features(make_card(1))
    assert pts.shape[0] == des.shape[0] > 4
    assert des.shape[1] == 128
    # RootSIFT descriptors are unit-ish after L1+sqrt; all non-negative
    assert (des >= 0).all()


def test_ransac_prefers_true_card(matcher, monkeypatch):
    """The card the query actually shows must win the re-rank even when a
    lookalike edges it on VLAD similarity."""
    true_img = make_card(42)
    other_img = make_card(43)
    # Query: the true card, slightly perspective-warped (an imperfect dewarp)
    h, w = true_img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[8, 4], [w - 4, 8], [w - 8, h - 4], [4, h - 8]])
    query = cv2.warpPerspective(true_img, cv2.getPerspectiveTransform(src, dst), (w, h))

    refs = {'111': true_img, '222': other_img}
    monkeypatch.setattr(matcher, '_fetch_reference_image', lambda card_id: refs[card_id])
    # VLAD says the wrong card is (marginally) better
    monkeypatch.setattr(
        VLADCardSearch, 'search',
        lambda self, image, top_k=5: [('222', 0.31), ('111', 0.30)],
    )

    results = matcher.search_verified(query, top_k=2)
    assert [r[0] for r in results] == ['111', '222']
    assert results[0][2] > results[1][2]  # true card has more inliers
    assert results[0][2] > 50  # a real match produces a strong inlier count


def test_search_verified_falls_back_to_vlad_order(matcher, monkeypatch):
    """If no reference images can be fetched, VLAD ordering is preserved."""
    monkeypatch.setattr(matcher, '_fetch_reference_image', lambda card_id: None)
    monkeypatch.setattr(
        VLADCardSearch, 'search',
        lambda self, image, top_k=5: [('1', 0.5), ('2', 0.4), ('3', 0.3)],
    )
    results = matcher.search_verified(make_card(7), top_k=3)
    assert [r[0] for r in results] == ['1', '2', '3']
    assert all(r[2] == 0 for r in results)


def test_search_verified_empty_search(matcher, monkeypatch):
    monkeypatch.setattr(VLADCardSearch, 'search', lambda self, image, top_k=5: [])
    assert matcher.search_verified(make_card(9)) == []


def test_chunked_search_matches_full_promote(matcher, monkeypatch):
    """The chunked float32 dot in search() must equal the single full-matrix
    promote it replaced, across chunk boundaries."""
    rng = np.random.default_rng(0)
    n, dim = 5000, 64  # n deliberately not a multiple of the chunk size
    db = rng.normal(size=(n, dim)).astype(np.float32)
    db /= np.linalg.norm(db, axis=1, keepdims=True)
    matcher.database = {str(i): db[i].astype(np.float16) for i in range(n)}
    matcher._rebuild_search_cache()

    query_vlad = db[7].astype(np.float16)
    monkeypatch.setattr(matcher, 'encode_vlad', lambda image, max_features=None: query_vlad)

    results = matcher.search(np.zeros((10, 10, 3), dtype=np.uint8), top_k=50)

    expected = np.dot(matcher._db_array.astype(np.float32), query_vlad.astype(np.float32))
    order = np.argsort(expected)[-50:][::-1]
    assert [r[0] for r in results] == [str(matcher._db_ids[i]) for i in order]
    np.testing.assert_array_equal(
        np.array([r[1] for r in results], dtype=np.float32), expected[order]
    )


def test_reference_features_cached_on_disk(matcher, monkeypatch, tmp_path):
    """Second lookup must hit the npz cache, not re-download."""
    monkeypatch.setattr(vlad_matcher.settings, 'ref_image_cache_path', str(tmp_path))
    calls = []

    def fake_fetch(card_id):
        calls.append(card_id)
        return make_card(5)

    monkeypatch.setattr(matcher, '_fetch_reference_image', fake_fetch)
    pts1, des1 = matcher._reference_features('999')
    matcher._ref_features.clear()  # drop memory cache, keep disk cache
    pts2, des2 = matcher._reference_features('999')

    assert calls == ['999']
    np.testing.assert_allclose(pts1, pts2)
    # fp16 round-trip on disk
    np.testing.assert_allclose(des1, des2, atol=1e-3)
