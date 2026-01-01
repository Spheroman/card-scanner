"""
Handles the scanner for the card scanner.
Uses ultralytics YOLO to segment the card from the image.
Uses OpenCV to crop the image using the segmentation mask.
Checks the card against the VLAD matcher to get the product ID.
Returns the product ID.
"""

from ultralytics import YOLO
import cv2
import numpy as np
import vlad_matcher
import time
from collections import defaultdict


class CardTracker:
    """
    Tracks cards across frames and manages their identification state.
    Only triggers identification when necessary (new track or after cooldown).
    """
    def __init__(self, identification_cooldown=5.0):
        """
        Args:
            identification_cooldown: Minimum seconds between re-identifying the same track
        """
        self.tracks = {}  # track_id -> {'matches': [...], 'last_identified': timestamp, 'box': [...]}
        self.identification_cooldown = identification_cooldown

    def should_identify(self, track_id):
        """
        Determine if we should run identification for this track.
        Returns True if:
        - Track is new (never seen before)
        - Track hasn't been identified yet
        - Enough time has passed since last identification
        """
        if track_id not in self.tracks:
            return True

        track = self.tracks[track_id]
        if 'matches' not in track or not track['matches']:
            return True

        time_since_last = time.time() - track.get('last_identified', 0)
        return time_since_last >= self.identification_cooldown

    def update_track(self, track_id, box, matches=None):
        """
        Update track information.
        Args:
            track_id: The tracking ID from YOLO
            box: Bounding box coordinates
            matches: Optional list of match results from identification
        """
        if track_id not in self.tracks:
            self.tracks[track_id] = {}

        self.tracks[track_id]['box'] = box

        if matches is not None:
            self.tracks[track_id]['matches'] = matches
            self.tracks[track_id]['last_identified'] = time.time()

    def get_track(self, track_id):
        """Get track information by ID."""
        return self.tracks.get(track_id, None)

    def cleanup_old_tracks(self, active_track_ids):
        """Remove tracks that are no longer active."""
        inactive_tracks = set(self.tracks.keys()) - set(active_track_ids)
        for track_id in inactive_tracks:
            del self.tracks[track_id]

    def reset(self):
        """Clear all tracked cards."""
        self.tracks.clear()


class Scanner:
    def __init__(self, model_path='models/best(2).pt', vocab_path=None, db_path=None):
        """
        Initialize the Scanner with YOLO model and VLAD matcher.
        """
        self.model = YOLO(model_path)
        self.matcher = vlad_matcher.VLADCardSearch(vocab_path=vocab_path, db_path=db_path)

    def start_scheduled_updates(self):
        """Start the scheduled update background task for the VLAD matcher."""
        self.matcher.start_scheduled_updates()

    def segment(self, image):
        """
        Use YOLO to segment cards from the image.
        Returns a list of results containing boxes/masks.
        """
        results = self.model(image)
        return results

    def order_points(self, pts):
        """
        Orders points in order: top-left, top-right, bottom-right, bottom-left.
        """
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def crop(self, image, box, mask=None):
        """
        Crop and dewarp the image based on the bounding box and optional mask.
        If mask is provided, performs perspective transformation.
        """
        if mask is not None:
            # Extract polygon from mask
            polygon = mask.xy[0]
            # Approximate the polygon to 4 points
            peri = cv2.arcLength(polygon, True)
            approx = cv2.approxPolyDP(polygon, 0.02 * peri, True)
            
            if len(approx) == 4:
                pts = approx.reshape(4, 2)
                rect = self.order_points(pts)
                (tl, tr, br, bl) = rect

                # Define standard card dimensions (63mm x 88mm ratio)
                # We'll use a fixed width of 400px for consistency
                card_width = 400
                card_height = int(card_width * (88 / 63))
                
                # Check if the detected card is horizontal or vertical
                widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
                widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
                heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
                heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
                
                orig_width = max(int(widthA), int(widthB))
                orig_height = max(int(heightA), int(heightB))
                
                if orig_width > orig_height:
                    # Detected horizontal, rotate mapping to make it vertical
                    # Map: bl->tl, tl->tr, tr->br, br->bl
                    rect = np.array([rect[3], rect[0], rect[1], rect[2]], dtype="float32")

                dst = np.array([
                    [0, 0],
                    [card_width - 1, 0],
                    [card_width - 1, card_height - 1],
                    [0, card_height - 1]], dtype="float32")

                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(image, M, (card_width, card_height))
                return warped

        # Fallback to simple crop
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped = image[y1:y2, x1:x2]
        return cropped

    def match(self, cropped_image, top_k=1):
        """
        Match the cropped card image using VLAD.
        Returns a list of (card_id, similarity) tuples.
        """
        results = self.matcher.search(cropped_image, top_k=top_k)
        return results

    def identify_card(self, image, k=1):
        """
        Identify a single pre-cropped card image.
        Resizes the image to the standard card size used in the pipeline before matching.
        Returns a dictionary with matches and the bounding box (full image).
        """
        # Resize to standard dimensions used in crop()
        card_width = 400
        card_height = int(card_width * (88 / 63))
        
        resized = cv2.resize(image, (card_width, card_height))
        matches = self.match(resized, top_k=k)
        
        return {
            'matches': [
                {'card_id': m[0], 'similarity': float(m[1])}
                for m in matches
            ],
            'box': [0, 0, image.shape[1], image.shape[0]] # Full image box
        }

    def scan(self, image, k=1):
        """
        Full scan pipeline: segment, crop (with dewarp), and match.
        Returns a list of dictionaries, each containing the bounding box and a list of matches.
        """
        results = self.segment(image)
        scanned_cards = []

        for result in results:
            if result.boxes:
                for i in range(len(result.boxes)):
                    box = result.boxes[i]
                    mask = result.masks[i] if result.masks is not None else None
                    cropped = self.crop(image, box, mask)
                    matches = self.match(cropped, top_k=k)
                    
                    if matches:
                        scanned_cards.append({
                            'matches': [
                                {'card_id': m[0], 'similarity': float(m[1])}
                                for m in matches
                            ],
                            'box': box.xyxy[0].tolist()
                        })
        
        return scanned_cards

    def track_and_identify(self, image, tracker, k=1, persist=True):
        """
        Track cards across frames and only identify new or updated tracks.
        Uses YOLO's built-in tracking to avoid re-identifying the same card every frame.

        Args:
            image: Input frame
            tracker: CardTracker instance to manage tracking state
            k: Number of top matches to return for each card
            persist: Whether to persist tracks across calls

        Returns:
            List of dictionaries with track_id, box, and matches (if identified)
        """
        # Run YOLO with tracking enabled
        results = self.model.track(image, persist=persist, verbose=False)

        tracked_cards = []
        active_track_ids = []

        for result in results:
            if result.boxes and result.boxes.id is not None:
                for i in range(len(result.boxes)):
                    box = result.boxes[i]
                    track_id = int(result.boxes.id[i])
                    active_track_ids.append(track_id)

                    # Get mask if available
                    mask = result.masks[i] if result.masks is not None else None

                    # Check if we should identify this track
                    if tracker.should_identify(track_id):
                        # Crop and identify
                        cropped = self.crop(image, box, mask)
                        matches = self.match(cropped, top_k=k)

                        # Convert to dict format
                        matches_list = [
                            {'card_id': m[0], 'similarity': float(m[1])}
                            for m in matches
                        ]

                        # Update tracker with new identification
                        tracker.update_track(track_id, box.xyxy[0].tolist(), matches_list)
                    else:
                        # Just update the box position without re-identifying
                        tracker.update_track(track_id, box.xyxy[0].tolist())

                    # Get current track data
                    track = tracker.get_track(track_id)
                    tracked_cards.append({
                        'track_id': track_id,
                        'box': track['box'],
                        'matches': track.get('matches', [])
                    })

        # Clean up tracks that are no longer detected
        tracker.cleanup_old_tracks(active_track_ids)

        return tracked_cards

if __name__ == "__main__":
    # Test script for scanner
    scanner = Scanner()
    test_images = ['test_images/257279.png', 'test_images/276982.png']
    
    for test_img_path in test_images:
        print(f"\nScanning {test_img_path}...")
        img = cv2.imread(test_img_path)
        if img is not None:
            cards = scanner.scan(img, k=3)
            print(f"Detected {len(cards)} card segments:")
            for i, card in enumerate(cards):
                print(f"  Segment {i+1} matches:")
                for match in card['matches']:
                    print(f"    ID: {match['card_id']}, Similarity: {match['similarity']:.4f}")
        else:
            print(f"Could not load image {test_img_path}")