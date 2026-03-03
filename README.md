# Card Scanner

>[!WARNING]
>This project was created using AI tools. The tools were guided by me, but much of the implementation was left to the tools.

A Pokemon card scanning and identification system using YOLO for segmentation and ORB+VLAD for identification.

## Features

- **High Speed Identification**: Uses VLAD (Vector of Locally Aggregated Descriptors) for fast and accurate card matching.
- **Automated Vectors Sync**: Automatically pulls pre-generated vectors and vocabulary from a centralized repository.
- **Daily Updates**: Background tasks to keep product prices and match vectors up to date.
- **Web Frontend**: Built-in webcam scanner UI at `/` — capture cards and view results with pricing.
- **REST API**: FastAPI-based interface for scanning (`/scan`) and identifying (`/identify`) cards.
- **Production Ready**: Includes Debian installation scripts and systemd service configuration.

## System Architecture

- **Segmentation**: YOLOv11 handles card detection and perspective correction.
- **Identification**: ORB features aggregated into VLAD vectors.
- **Database**: Asynchronous SQLite database stores product metadata and real-time market prices.

## Installation

### Docker (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/card-sorter/card-scanner.git
   cd card-scanner
   ```

2. Build and run:
   ```bash
   docker build -t card-scanner .
   docker run -p 8000:8000 card-scanner
   ```

The Docker image uses a multi-stage build with CPU-only PyTorch and a slim Python base image to keep the final image small. The container exposes port 8000.

To persist the database and vectors across restarts, mount volumes:
```bash
docker run -p 8000:8000 \
  -v card-scanner-db:/app/database.db \
  -v card-scanner-vectors:/app/vectors \
  card-scanner
```

To pass configuration via environment variables:
```bash
docker run -p 8000:8000 \
  -e CARD_SCANNER_API_KEYS=your-key-here \
  -e CARD_SCANNER_CORS_ORIGINS=https://yourdomain.com \
  card-scanner
```

### Fly.io

A `fly.toml` is included for deploying to [Fly.io](https://fly.io):
```bash
fly deploy
```

The default configuration uses a shared CPU with 2 GB memory in the `sjc` region.

### Debian/Ubuntu (Native)

1. Clone the repository:
   ```bash
   git clone https://github.com/card-sorter/card-scanner.git
   cd card-scanner
   ```

2. Run the installation script:
   ```bash
   bash install.sh
   ```

## Web Frontend

Navigate to `http://localhost:8000` to use the built-in scanner UI. It provides a live webcam viewfinder — capture a photo to detect and identify cards. Results show the top 3 matches with thumbnails, pricing, and similarity scores. Click any result to see full card details.

## API Usage

### Scanning & Identification

#### Scan an Image
Upload an image containing one or more cards to detect and identify them.

```bash
curl -X POST "http://localhost:8000/scan" \
  -F "image=@your_card_photo.jpg"
```

#### Identify a Cropped Card
Identify a pre-cropped card image for maximum accuracy.

```bash
curl -X POST "http://localhost:8000/identify" \
  -F "image=@cropped_card.jpg"
```

### Pricing & Data

#### Get Card Price
Get market pricing for a specific card.

```bash
curl "http://localhost:8000/price?product_id=123"
```

#### Batch Prices
Get prices for multiple cards in one request.

```bash
curl -X POST "http://localhost:8000/prices" \
  -H "Content-Type: application/json" \
  -d '{"product_ids": [123, 456, 789]}'
```

### Metadata

#### List Categories
Get all supported card categories.

```bash
curl "http://localhost:8000/categories"
```

#### List Groups
Get sets/groups for a category.

```bash
curl "http://localhost:8000/groups?category_id=3"
```

### System Health

- **/health**: Liveness probe (returns 200 OK if service is running).
- **/ready**: Readiness probe (checks database and scanner initialization).

## Configuration

The application is configured via environment variables. Copy `.env.example` to `.env` to customize.

### Key Settings

- **Authentication**: Set `CARD_SCANNER_API_KEYS` to a comma-separated list of keys to enable auth.
- **CORS**: Set `CARD_SCANNER_CORS_ORIGINS` to allow specific domains (default is `*`).
- **Rate Limits**: Adjust `CARD_SCANNER_RATE_LIMIT_*` variables to control request throttling.
- **Schedule**:
    - **Sync Interval**: Vectors are synced once every 24 hours (default 4:00 AM).
    - **Database Update**: Product metadata updates daily (default 3:00 AM).

## License

AGPLv3
