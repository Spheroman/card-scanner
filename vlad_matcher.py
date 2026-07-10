"""
Handles the VLAD matcher for the card scanner.
"""

import cv2
import numpy as np
import pickle
import os
import time
import git
import asyncio
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from config import settings
from logging_config import get_logger

logger = get_logger(__name__)


class VLADCardSearch:
    """
    Minimal VLAD card search - load pre-built database and search only.
    No training or database building - just fast inference.
    """

    def __init__(self, vocab_path=None, db_path=None, normalize_size=(400, 557)):
        """
        Initialize searcher with paths to vocabulary and database.
        If paths are not provided, it will use the vectors repository.

        Uses RootSIFT features with a VLAD codebook (default K=256). The
        encoding here MUST match the database generation script (vlad.py) in
        the vectors repository so query and database vectors are comparable.
        """
        self.sift = cv2.SIFT_create()
        self.centers = None
        self.k = None
        self.database = {}
        # Image normalization size (width, height). Must match the size the
        # database vectors were generated with (standard card ratio 400x557).
        self.normalize_size = normalize_size
        # Cached arrays for fast vectorized search (rebuilt on load/reload).
        self._db_array = None
        self._db_ids = None
        self.update_task = None
        self.repo_path = Path(settings.vectors_repo_path)
        # In-memory cache of reference SIFT features for geometric
        # verification: card_id -> (points Nx2 float32, RootSIFT descriptors
        # Nx128 float32), or None for cards whose image couldn't be fetched.
        self._ref_features = {}

        # Sync repository and set default paths
        self._sync_repository()

        self.vocab_path = vocab_path or str(self.repo_path / 'vlad_vocab.npz')
        self.db_path = db_path or str(self.repo_path / 'vectors')

        # Auto-load on initialization
        self.load_vocabulary()
        self.load_database()
    
    @retry(
        stop=stop_after_attempt(settings.retry_attempts),
        wait=wait_exponential(min=settings.retry_min_wait, max=settings.retry_max_wait),
        retry=retry_if_exception_type((git.GitCommandError, git.InvalidGitRepositoryError)),
        before_sleep=before_sleep_log(logger, "WARNING"),
        reraise=True,
    )
    def _git_clone(self):
        """Clone the vectors repository with retry logic."""
        logger.info(f"Cloning vectors repository from {settings.vectors_repo_url}...")
        git.Repo.clone_from(
            settings.vectors_repo_url,
            self.repo_path,
            depth=1  # Shallow clone for faster downloads
        )

    @retry(
        stop=stop_after_attempt(settings.retry_attempts),
        wait=wait_exponential(min=settings.retry_min_wait, max=settings.retry_max_wait),
        retry=retry_if_exception_type((git.GitCommandError,)),
        before_sleep=before_sleep_log(logger, "WARNING"),
        reraise=True,
    )
    def _git_pull(self):
        """Pull updates from vectors repository with retry logic."""
        logger.info("Checking for updates in vectors repository...")
        repo = git.Repo(self.repo_path)
        origin = repo.remotes.origin
        origin.pull()

    def _sync_repository(self, force=False):
        """Clone or pull the vectors repository."""
        sync_interval = 86400  # 24 hours in seconds
        try:
            if not self.repo_path.exists():
                self._git_clone()
                self._update_sync_timestamp()
            else:
                last_sync = self._get_last_sync_timestamp()
                if force or (time.time() - last_sync > sync_interval):
                    self._git_pull()
                    self._update_sync_timestamp()
                else:
                    logger.debug("Vectors repository is up to date (synced within 24h).")
        except Exception as e:
            logger.warning(f"Failed to sync vectors repository: {e}")

    def _get_last_sync_timestamp(self):
        sync_file = self.repo_path / '.last_sync'
        if sync_file.exists():
            try:
                return float(sync_file.read_text())
            except ValueError:
                return 0
        return 0

    def _update_sync_timestamp(self):
        sync_file = self.repo_path / '.last_sync'
        sync_file.write_text(str(time.time()))

    def _rebuild_search_cache(self):
        """Rebuild cached arrays (stacked vectors + ids) for fast searching."""
        if not self.database:
            self._db_array = None
            self._db_ids = None
            return
        self._db_ids = np.array(list(self.database.keys()))
        # Store as float16 for memory efficiency; promoted to float32 at search.
        self._db_array = np.vstack([np.asarray(v) for v in self.database.values()]).astype(np.float16)
        logger.info(f"Search cache built: {len(self._db_ids)} cards, shape {self._db_array.shape}")

    def reload_database(self):
        """Reload database into a new dictionary and swap to prevent race conditions."""
        db_path = Path(self.db_path)
        new_database = {}
        if db_path.is_dir():
            logger.info(f"Reloading split databases from {db_path}...")
            pkl_files = list(db_path.rglob("*.pkl"))
            total_loaded = 0
            for pkl_file in pkl_files:
                try:
                    with open(pkl_file, 'rb') as f:
                        data = pickle.load(f)
                        new_database.update(data)
                        total_loaded += len(data)
                except Exception as e:
                    logger.warning(f"Failed to reload {pkl_file}: {e}")
            logger.info(f"Reloaded {total_loaded} cards from {len(pkl_files)} files")
        else:
            with open(self.db_path, 'rb') as f:
                new_database = pickle.load(f)
            logger.info(f"Reloaded {len(new_database)} cards")

        self.database = new_database
        self._rebuild_search_cache()

    def sync_and_reload(self):
        """Sync with remote repository and reload database."""
        logger.info("Starting manual sync and reload...")
        self._sync_repository(force=True)
        self.reload_database()
        logger.info("Sync and reload complete.")

    async def scheduled_update(self):
        """Run updates at scheduled time every 24 hours."""
        while True:
            now = datetime.now()
            update_time = settings.vectors_update_time
            target = datetime.combine(now.date(), update_time)

            # If target time has passed today, schedule for tomorrow
            if now.time() > update_time:
                target = target + timedelta(days=1)

            sleep_seconds = (target - now).total_seconds()
            logger.info(f"Matcher background update scheduled in {sleep_seconds/3600:.2f} hours")

            await asyncio.sleep(sleep_seconds)
            logger.info("Running scheduled matcher update...")
            # Run blocking I/O in a thread to keep event loop free
            await asyncio.to_thread(self.sync_and_reload)

    def start_scheduled_updates(self):
        """Start the scheduled update background task."""
        if self.update_task is None or self.update_task.done():
            self.update_task = asyncio.create_task(self.scheduled_update())
            logger.info("Matcher scheduled updates started")

    def load_vocabulary(self):
        """Load vocabulary from disk."""
        if not os.path.exists(self.vocab_path):
            raise FileNotFoundError(f"Vocabulary not found at {self.vocab_path}")
        data = np.load(self.vocab_path, allow_pickle=True)
        self.centers = data['centers'].astype(np.float32)
        self.k = int(data['k'])

        # Guard against a codebook/encoder mismatch. The query encoder uses SIFT
        # (128-D descriptors), so the VLAD centers must be 128-D too. A stale
        # vectors clone — e.g. an old ORB codebook (32-D) left behind when a
        # sync pull silently failed — would otherwise load here and produce
        # dimension errors or garbage matches with no obvious cause. Fail loudly.
        expected_dim = self.sift.descriptorSize()
        actual_dim = self.centers.shape[1]
        if actual_dim != expected_dim:
            raise ValueError(
                f"Codebook/encoder mismatch: vocabulary centers are {actual_dim}-D "
                f"but the SIFT encoder produces {expected_dim}-D descriptors. The "
                f"vectors repository at {self.repo_path} is likely stale (a sync "
                f"pull may have failed) — delete it to force a fresh clone, or "
                f"regenerate the codebook. Vocab: {self.vocab_path}"
            )

        # Use the normalize_size the vocabulary was trained with, if present,
        # so encoding matches database generation exactly.
        if 'normalize_size' in data:
            saved_size = data['normalize_size']
            if saved_size is None or (isinstance(saved_size, np.ndarray) and saved_size.shape == ()):
                saved_size = None
            else:
                saved_size = tuple(int(x) for x in saved_size)
            if saved_size != self.normalize_size:
                logger.warning(
                    f"Vocabulary trained with normalize_size={saved_size}, "
                    f"overriding current setting {self.normalize_size}"
                )
                self.normalize_size = saved_size

        logger.info(f"Vocabulary loaded: K={self.k}, dim={self.centers.shape[1]}, normalize={self.normalize_size}")

    def load_database(self):
        """Load database from disk (supports single .pkl or directory of .pkl files)."""
        db_path = Path(self.db_path)
        if db_path.is_dir():
            logger.info(f"Loading split databases from {db_path}...")
            pkl_files = list(db_path.rglob("*.pkl"))
            total_loaded = 0
            for pkl_file in pkl_files:
                try:
                    with open(pkl_file, 'rb') as f:
                        data = pickle.load(f)
                        self.database.update(data)
                        total_loaded += len(data)
                except Exception as e:
                    logger.warning(f"Failed to load {pkl_file}: {e}")
            logger.info(f"Total database loaded: {total_loaded} cards from {len(pkl_files)} files")
        elif db_path.exists():
            with open(self.db_path, 'rb') as f:
                self.database = pickle.load(f)
            logger.info(f"Database loaded: {len(self.database)} cards")
        else:
            logger.warning(f"Database not found at {self.db_path}, starting with empty database")
            self.database = {}

        self._rebuild_search_cache()
    
    def encode_vlad(self, image, max_features=None):
        """
        Encode an image to a VLAD vector using RootSIFT features.

        This must match the database generation algorithm (vlad.py) exactly:
        grayscale -> resize to normalize_size -> SIFT -> RootSIFT (L1 + sqrt)
        -> VLAD aggregation -> power norm -> L2 norm -> float16.

        Args:
            image: cv2 image (numpy array, BGR or Grayscale)
            max_features: Optional cap on SIFT features (keeps strongest);
                          None uses all features (best accuracy).

        Returns:
            VLAD vector (normalized, float16)
        """
        if image is None:
            raise ValueError("Invalid image provided")

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            img = image

        # Normalize image size to match database generation
        if self.normalize_size is not None:
            img = cv2.resize(img, self.normalize_size, interpolation=cv2.INTER_AREA)

        # Extract SIFT descriptors
        kp, des = self.sift.detectAndCompute(img, None)

        if des is None:
            # Return zero vector if no features
            return np.zeros(self.centers.shape[0] * self.centers.shape[1], dtype=np.float16)

        # Optionally keep only the strongest features (for real-time speed)
        if max_features and len(des) > max_features:
            responses = np.array([k.response for k in kp])
            top_indices = np.argsort(responses)[-max_features:]
            des = des[top_indices]

        # RootSIFT normalization: L1 normalize then element-wise sqrt
        des = des.astype(np.float32)
        des /= (np.linalg.norm(des, ord=1, axis=1, keepdims=True) + 1e-7)
        np.sqrt(des, out=des)

        K, D = self.centers.shape
        vlad_vector = np.zeros((K, D), dtype=np.float32)

        # Vectorized nearest-center assignment and residual aggregation
        distances = np.sqrt(np.sum((des[:, None, :] - self.centers[None, :, :]) ** 2, axis=2))
        assignments = np.argmin(distances, axis=1)
        for k in range(K):
            mask = (assignments == k)
            if np.any(mask):
                vlad_vector[k] = np.sum(des[mask] - self.centers[k], axis=0)

        # Flatten and normalize
        vlad_flat = vlad_vector.flatten()
        vlad_flat = np.sign(vlad_flat) * np.sqrt(np.abs(vlad_flat))  # Power norm

        norm = np.linalg.norm(vlad_flat)
        if norm > 0:
            vlad_flat = vlad_flat / norm  # L2 norm

        return vlad_flat.astype(np.float16)
    
    def search(self, query_image, top_k=5):
        """
        Search for most similar cards.
        
        Args:
            query_image: cv2 image (numpy array)
            top_k: Number of results to return
            
        Returns:
            List of (card_id, similarity_score) tuples, sorted by similarity
        """
        if not self.database:
            return []

        # Ensure the search cache is built
        if self._db_array is None:
            self._rebuild_search_cache()
            if self._db_array is None:
                return []

        # Encode query
        query_vlad = self.encode_vlad(query_image)

        # Cosine similarity via dot product (vectors are L2-normalized).
        # Computed in float32, chunked: promoting the whole float16 matrix at
        # once materializes a 2x full-database copy (gigabytes) per query,
        # which spikes past the deploy VM's memory. Doing the math natively in
        # float16 is not an option either: numpy has no BLAS path for it
        # (slower) and its accumulation error (~1e-4) is significant relative
        # to real match margins (~2e-2).
        query_f32 = query_vlad.astype(np.float32)
        similarities = np.empty(self._db_array.shape[0], dtype=np.float32)
        chunk = 2048
        for i in range(0, self._db_array.shape[0], chunk):
            similarities[i:i + chunk] = self._db_array[i:i + chunk].astype(np.float32) @ query_f32

        # Select top-k
        top_k = min(top_k, len(similarities))
        if top_k < 50:
            top_indices = np.argsort(similarities)[-top_k:][::-1]
        else:
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        results = [(str(self._db_ids[i]), float(similarities[i])) for i in top_indices]
        return results
    
    @staticmethod
    def _rootsift(des):
        """RootSIFT-transform raw SIFT descriptors (L1 normalize + sqrt)."""
        des = des.astype(np.float32)
        des /= (np.linalg.norm(des, ord=1, axis=1, keepdims=True) + 1e-7)
        np.sqrt(des, out=des)
        return des

    # Canonical size for geometric matching; matches the dewarped crop
    # produced by scanner.py (400 x 400*88/63).
    _MATCH_SIZE = (400, 558)

    def _extract_match_features(self, image):
        """Grayscale, resize to canonical card size, extract RootSIFT features.

        Returns (points Nx2 float32, descriptors Nx128 float32) or None if
        too few features for a homography.
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, self._MATCH_SIZE, interpolation=cv2.INTER_AREA)
        kp, des = self.sift.detectAndCompute(image, None)
        if des is None or len(des) < 4:
            return None
        pts = np.float32([k.pt for k in kp])
        return pts, self._rootsift(des)

    def _fetch_reference_image(self, card_id):
        """Download a card's reference image (disk-cached). Returns cv2 image or None."""
        cache_dir = Path(settings.ref_image_cache_path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        image_path = cache_dir / f"{card_id}.jpg"
        if not image_path.exists():
            for template in settings.ref_image_url_templates:
                url = template.format(card_id=card_id)
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "card-scanner"})
                    with urllib.request.urlopen(req, timeout=settings.http_timeout) as resp:
                        data = resp.read()
                    image_path.write_bytes(data)
                    break
                except Exception as e:
                    logger.debug(f"Reference image fetch failed for {card_id} at {url}: {e}")
            else:
                logger.warning(f"No reference image available for card {card_id}")
                return None
        image = cv2.imread(str(image_path))
        if image is None:
            # Corrupt cache entry; remove so a later call can re-fetch.
            image_path.unlink(missing_ok=True)
        return image

    def _reference_features(self, card_id):
        """Get (points, descriptors) for a card's reference image, or None.

        Cached in memory and on disk (npz next to the cached image), so each
        card pays the download + SIFT cost once.
        """
        if card_id in self._ref_features:
            return self._ref_features[card_id]

        features = None
        npz_path = Path(settings.ref_image_cache_path) / f"{card_id}_sift.npz"
        if npz_path.exists():
            try:
                data = np.load(npz_path)
                features = (data['pts'].astype(np.float32), data['des'].astype(np.float32))
            except Exception as e:
                logger.warning(f"Failed to load cached features for {card_id}: {e}")
                npz_path.unlink(missing_ok=True)

        if features is None:
            image = self._fetch_reference_image(card_id)
            if image is not None:
                features = self._extract_match_features(image)
                if features is not None:
                    # float16 halves disk usage; promoted back on load.
                    npz_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(npz_path, pts=features[0], des=features[1].astype(np.float16))
                    features = (features[0], features[1])

        self._ref_features[card_id] = features
        return features

    def _ransac_inliers(self, query_pts, query_des, ref_pts, ref_des):
        """Count RANSAC homography inliers between query and reference features."""
        matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(query_des, ref_des, k=2)
        good = [
            m for m, n in (pair for pair in matches if len(pair) == 2)
            if m.distance < settings.rerank_lowe_ratio * n.distance
        ]
        if len(good) < 4:
            return len(good)
        src = query_pts[[m.queryIdx for m in good]].reshape(-1, 1, 2)
        dst = ref_pts[[m.trainIdx for m in good]].reshape(-1, 1, 2)
        _, inlier_mask = cv2.findHomography(
            src, dst, cv2.RANSAC, settings.rerank_ransac_reproj
        )
        return int(inlier_mask.sum()) if inlier_mask is not None else 0

    def search_verified(self, query_image, top_k=5, rerank_k=None):
        """
        Search with geometric verification: VLAD retrieves candidates, then
        SIFT + RANSAC homography inlier counts re-rank them.

        Global VLAD vectors can't separate same-art printings and get fragile
        on degraded photos (margins of ~0.02); the inlier count is a much
        stronger signal and doubles as a confidence score (tens of inliers or
        fewer means "not sure").

        Args:
            query_image: cv2 image (numpy array)
            top_k: Number of results to return
            rerank_k: How many VLAD candidates to verify
                      (default settings.rerank_candidates)

        Returns:
            List of (card_id, similarity, inliers) tuples sorted by inliers
            (similarity as tie-break). Falls back to plain VLAD order (all
            inliers 0) if the query has too few features or no reference
            images could be fetched.
        """
        rerank_k = rerank_k or settings.rerank_candidates
        candidates = self.search(query_image, top_k=max(rerank_k, top_k))
        if not candidates:
            return []

        query_features = self._extract_match_features(query_image)
        results = []
        for card_id, similarity in candidates:
            inliers = 0
            if query_features is not None:
                ref = self._reference_features(card_id)
                if ref is not None:
                    inliers = self._ransac_inliers(*query_features, *ref)
            results.append((card_id, similarity, inliers))

        results.sort(key=lambda r: (-r[2], -r[1]))
        return results[:top_k]

    def compare_images(self, image1, image2):
        """
        Compare two images directly.
        
        Args:
            image1: First cv2 image (numpy array)
            image2: Second cv2 image (numpy array)
            
        Returns:
            Similarity score (0-1, higher is more similar)
        """
        vlad1 = self.encode_vlad(image1).astype(np.float32)
        vlad2 = self.encode_vlad(image2).astype(np.float32)
        similarity = np.dot(vlad1, vlad2)
        return float(similarity)
    
    def batch_search(self, query_images, top_k=5):
        """
        Search for multiple query images at once.
        
        Args:
            query_images: List of cv2 images (numpy arrays)
            top_k: Number of results per query
            
        Returns:
            List of result lists, each being a list of (card_id, similarity) tuples
        """
        results = []
        for query_img in query_images:
            results.append(self.search(query_img, top_k))
        return results


# Example usage
if __name__ == "__main__":
    # Initialize (auto-loads vocab and database)
    searcher = VLADCardSearch(
        vocab_path='vlad_vocab.npz',
        db_path='vectors'
    )
    
    # Search for similar cards
    query_path = 'query_card.jpg'
    query_img = cv2.imread(query_path)
    if query_img is not None:
        results = searcher.search(query_img, top_k=5)
        print(f"\nTop 5 matches for {query_path}:")
        for card_id, similarity in results:
            print(f"  {card_id}: {similarity:.4f}")
    
    # Compare two specific images
    img1 = cv2.imread('card1.jpg')
    img2 = cv2.imread('card2.jpg')
    if img1 is not None and img2 is not None:
        similarity = searcher.compare_images(img1, img2)
        print(f"\nSimilarity between card1 and card2: {similarity:.4f}")
    
    # Batch search multiple queries
    query_paths = ['query1.jpg', 'query2.jpg', 'query3.jpg']
    query_imgs = [cv2.imread(p) for p in query_paths if cv2.imread(p) is not None]
    if query_imgs:
        batch_results = searcher.batch_search(query_imgs, top_k=3)
        print("\nBatch search results:")
        for i, matches in enumerate(batch_results):
            print(f"\nQuery {i+1}:")
            for card_id, sim in matches:
                print(f"  {card_id}: {sim:.4f}")