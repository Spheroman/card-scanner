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

    def __init__(self, vocab_path=None, db_path=None):
        """
        Initialize searcher with paths to vocabulary and database.
        If paths are not provided, it will use the vectors repository.
        """
        self.orb = cv2.ORB_create()
        self.centers = None
        self.k = None
        self.database = {}
        self.update_task = None
        self.repo_path = Path(settings.vectors_repo_path)

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
        data = np.load(self.vocab_path)
        self.centers = data['centers']
        self.k = int(data['k'])
        logger.info(f"Vocabulary loaded: K={self.k}")

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
    
    def encode_vlad(self, image):
        """
        Encode an image to VLAD vector.
        
        Args:
            image: cv2 image (numpy array, BGR or Grayscale)
            
        Returns:
            VLAD vector (normalized)
        """
        if image is None:
            raise ValueError("Invalid image provided")

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            img = image
        
        # Extract ORB descriptors
        _, des = self.orb.detectAndCompute(img, None)
        
        if des is None:
            # Return zero vector if no features
            return np.zeros(self.centers.shape[0] * self.centers.shape[1], dtype=np.float32)
        
        des = des.astype(np.float32)
        K, D = self.centers.shape
        vlad_vector = np.zeros((K, D), dtype=np.float32)
        
        # Assign descriptors to nearest centers and aggregate residuals
        for descriptor in des:
            distances = np.linalg.norm(self.centers - descriptor, axis=1)
            nearest_idx = np.argmin(distances)
            residual = descriptor - self.centers[nearest_idx]
            vlad_vector[nearest_idx] += residual
        
        # Flatten and normalize
        vlad_flat = vlad_vector.flatten()
        vlad_flat = np.sign(vlad_flat) * np.sqrt(np.abs(vlad_flat))  # Power norm
        
        norm = np.linalg.norm(vlad_flat)
        if norm > 0:
            vlad_flat = vlad_flat / norm  # L2 norm
        
        return vlad_flat
    
    def search(self, query_image, top_k=5):
        """
        Search for most similar cards.
        
        Args:
            query_image: cv2 image (numpy array)
            top_k: Number of results to return
            
        Returns:
            List of (card_id, similarity_score) tuples, sorted by similarity
        """
        # Encode query
        query_vlad = self.encode_vlad(query_image)
        
        # Compute similarities
        similarities = {}
        for card_id, db_vlad in self.database.items():
            similarity = np.dot(query_vlad, db_vlad)
            similarities[card_id] = similarity
        
        # Sort and return top-k
        results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return results
    
    def compare_images(self, image1, image2):
        """
        Compare two images directly.
        
        Args:
            image1: First cv2 image (numpy array)
            image2: Second cv2 image (numpy array)
            
        Returns:
            Similarity score (0-1, higher is more similar)
        """
        vlad1 = self.encode_vlad(image1)
        vlad2 = self.encode_vlad(image2)
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
        vocab_path='models/pokemon_vocab.npz',
        db_path='pokemon_database.pkl'
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