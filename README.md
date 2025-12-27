# Card Scanner

>[!WARNING]
>This project was created using AI tools. The tools were guided by me, but much of the implementation was left to the tools.

A Pokemon card scanning and identification system using YOLO for segmentation and ORB+VLAD for identification.

## Features

- **High Speed Identification**: Uses VLAD (Vector of Locally Aggregated Descriptors) for fast and accurate card matching.
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

## Configuration

- **Sync Interval**: Vectors are synced once every 24 hours.
- **Update Time**: Database updates are scheduled at 3:00 AM, and vector sync is scheduled at 4:00 AM.
- **Environment**: Virtual environment created automatically by `install.sh`.

## License

AGPLv3
