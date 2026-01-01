# Card Scanner

>[!WARNING]
>This project was created using AI tools. The tools were guided by me, but much of the implementation was left to the tools.

A Pokemon card scanning and identification system using YOLO for segmentation and ORB+VLAD for identification.

## Features

- **High Speed Identification**: Uses VLAD (Vector of Locally Aggregated Descriptors) for fast and accurate card matching.
- **Live Webcam Streaming**: WebSocket API with YOLO object tracking to efficiently track and identify cards in real-time without processing every frame.
- **Automated Vectors Sync**: Automatically pulls pre-generated vectors and vocabulary from a centralized repository.
- **Daily Updates**: Background tasks to keep product prices and match vectors up to date.
- **REST API**: FastAPI-based interface for scanning (`/scan`) and identifying (`/identify`) cards.
- **Production Ready**: Includes Debian installation scripts and systemd service configuration.

## System Architecture

- **Segmentation**: YOLOv11 handles card detection and perspective correction.
- **Identification**: ORB features aggregated into VLAD vectors.
- **Database**: Asynchronous SQLite database stores product metadata and real-time market prices.

## Installation (Debian/Ubuntu)

1. Clone the repository:
   ```bash
   git clone https://github.com/card-sorter/card-scanner.git
   cd card-scanner
   ```

2. Run the installation script:
   ```bash
   bash install.sh
   ```

## API Usage

### Scan an Image
Upload an image containing one or more cards to detect and identify them.

```bash
curl -X POST "http://localhost:8000/scan" -F "image=@your_card_photo.jpg"
```

### Identify a Cropped Card
Identify a pre-cropped card image for maximum accuracy.

```bash
curl -X POST "http://localhost:8000/identify" -F "image=@cropped_card.jpg"
```

### Live Webcam Streaming (WebSocket)
Connect to a WebSocket endpoint for real-time card tracking and identification. Uses YOLO object tracking to efficiently track cards across frames and only runs identification when necessary.

**Endpoint:** `ws://localhost:8000/webcam?top_n=3`

**Protocol:**
- Client sends base64-encoded JPEG frames
- Server responds with JSON containing tracked cards and their identifications

**Smart Identification:**
The tracking system only runs card identification when:
- A new card is detected (new track_id)
- A card hasn't been identified yet
- Sufficient time has passed since last identification (5-second cooldown)

This approach dramatically reduces computational load compared to identifying every frame.

**Example Usage:**
See `webcam_client_example.html` for a complete browser-based implementation.

**Control Commands:**
Send JSON to reset the tracker:
```json
{"command": "reset"}
```

**Response Format:**
```json
{
  "tracks": [
    {
      "track_id": 1,
      "box": [x1, y1, x2, y2],
      "matches": [
        {
          "card_id": 12345,
          "similarity": 0.95,
          "details": {
            "name": "Pikachu",
            "market_price": 12.50,
            ...
          }
        }
      ]
    }
  ]
}
```

## Configuration

- **Sync Interval**: Vectors are synced once every 24 hours.
- **Update Time**: Database updates are scheduled at 3:00 AM, and vector sync is scheduled at 4:00 AM.
- **Environment**: Virtual environment created automatically by `install.sh`.

## License

AGPLv3
