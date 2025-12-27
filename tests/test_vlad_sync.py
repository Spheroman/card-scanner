import cv2
import numpy as np
import os
from vlad_matcher import VLADCardSearch
from pathlib import Path

def test_vlad_sync():
    print("Testing VLADCardSearch with repository integration...")
    
    # Initialize without paths to trigger repo sync/defaults
    searcher = VLADCardSearch()
    
    print(f"Sync file exists: {(searcher.REPO_PATH / '.last_sync').exists()}")
    print(f"Vocabulary centers shape: {searcher.centers.shape if searcher.centers is not None else 'None'}")
    print(f"Database size: {len(searcher.database)} cards")
    
    if len(searcher.database) > 0:
        print("Success: Database loaded from split repository files.")
    else:
        print("Error: Database is empty.")
        exit(1)

    if searcher.centers is not None:
        print("Success: Vocabulary loaded from repository.")
    else:
        print("Error: Vocabulary failed to load.")
        exit(1)

    # Test basic identification (on a fake image with zero features)
    fake_img = np.zeros((400, 557, 3), dtype=np.uint8)
    results = searcher.search(fake_img, top_k=3)
    print(f"Search results for dummy image: {results}")

if __name__ == "__main__":
    test_vlad_sync()
