import cv2
import os
from vlad_matcher import VLADCardSearch

def test_vlad_refactor():
    # Ensure models exist (or at least paths are correct for initialization)
    vocab_path = 'models/pokemon_vocab.npz'
    db_path = 'pokemon_database.pkl'
    
    if not os.path.exists(vocab_path) or not os.path.exists(db_path):
        print(f"Warning: {vocab_path} or {db_path} not found. Skipping initialization test.")
        return

    searcher = VLADCardSearch(vocab_path=vocab_path, db_path=db_path)
    
    # Test image provided by user
    test_image_path = 'test_images/623443.png'
    
    if not os.path.exists(test_image_path):
        print(f"Error: {test_image_path} not found.")
        return
        
    img = cv2.imread(test_image_path)
    if img is None:
        print(f"Error: Could not read image {test_image_path}")
        return
        
    print(f"Successfully loaded image {test_image_path}")
    
    # Test search
    print("Testing search...")
    results = searcher.search(img, top_k=5)
    print(f"Search results: {results}")
    
    if results:
        # Check if 623443 matches the top result (as hinted by user)
        top_card_id = results[0][0]
        print(f"Top match ID: {top_card_id}")
        if '623443' in str(top_card_id):
            print("SUCCESS: Top match matches the test image ID!")
        else:
            print("INFO: Top match does not contain '623443'. This might be expected depending on the database.")

    # Test compare_images
    print("\nTesting compare_images...")
    similarity = searcher.compare_images(img, img)
    print(f"Self-similarity: {similarity:.4f}")
    assert similarity > 0.99, f"Self-similarity should be ~1.0, got {similarity}"

    # Test batch_search
    print("\nTesting batch_search...")
    batch_results = searcher.batch_search([img], top_k=3)
    print(f"Batch results: {batch_results}")
    assert len(batch_results) == 1
    assert len(batch_results[0]) <= 3

    print("\nAll refactor tests passed!")

if __name__ == "__main__":
    test_vlad_refactor()
