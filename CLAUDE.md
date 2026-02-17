# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pokemon trading card scanner using YOLOv11 for detection/segmentation and ORB+VLAD for identification. FastAPI REST API with async SQLite database for product metadata and pricing from TCGPlayer.

## Commands

```bash
# Install (Debian/Ubuntu)
bash install.sh

# Run server
python main.py
# or: uvicorn api:app --host 0.0.0.0 --port 8000

# Run tests
pytest tests/
```

## Architecture

**Data Flow:**
1. Image upload → YOLO detects card bounding boxes/masks → perspective correction
2. ORB features extracted → VLAD vector encoding → vector database search
3. Database lookup for product details/pricing → return matches

**Key Modules:**
- `config.py` - Centralized settings with environment variable support (pydantic-settings)
- `logging_config.py` - Structured JSON logging with correlation ID support
- `api.py` - FastAPI app with endpoints, middleware, rate limiting, auth
- `scanner.py` - YOLO detection + perspective correction + VLAD matching orchestration
- `vlad_matcher.py` - ORB feature extraction, VLAD encoding, vector search
- `database.py` - Async SQLite operations, CSV sync from tcgcsv.com with retry logic

**Endpoints:**
- `GET /health` - Liveness probe (always returns 200)
- `GET /ready` - Readiness probe (checks DB and scanner)
- `GET /metrics` - Prometheus metrics
- `POST /scan` - Full pipeline: detect and identify cards in image
- `POST /identify` - Fast path: identify pre-cropped card (skips YOLO)
- `GET /price`, `POST /prices` - Pricing lookups
- `POST /update` - Trigger database update (runs in background)

**External Dependencies:**
- Vector database synced from `https://github.com/card-sorter/vectors.git`
- Product data from `tcgcsv.com/tcgplayer/{categoryid}/{groupid}/ProductsAndPrices.csv`

**Scheduled Tasks:**
- Database update: 3:00 AM daily (configurable)
- Vector sync: 4:00 AM daily (configurable)

## Configuration

All settings are configurable via environment variables with `CARD_SCANNER_` prefix.
See `.env.example` for all available options.

**Key Settings:**
- `CARD_SCANNER_API_KEYS` - Comma-separated API keys (empty = auth disabled)
- `CARD_SCANNER_CORS_ORIGINS` - Comma-separated allowed origins
- `CARD_SCANNER_MAX_FILE_SIZE` - Max upload size in bytes (default: 10MB)
- `CARD_SCANNER_LOG_JSON` - Enable JSON logging (default: true)

## Key Files

- `config.py` - All configurable settings with defaults
- `models/best(2).pt` - YOLOv11 trained model (40.8 MB)
- `database.db` - SQLite database with products, groups, categories
- `vectors/` - Auto-synced git repo with VLAD vocabulary and vector databases
- `.env.example` - Example environment configuration

## Production Features

- **Authentication**: Optional API key auth via `X-API-Key` header
- **Rate Limiting**: Configurable per-endpoint limits (slowapi)
- **Metrics**: Prometheus metrics at `/metrics`
- **Health Checks**: `/health` (liveness) and `/ready` (readiness)
- **Correlation IDs**: Request tracing via `X-Correlation-ID` header
- **Structured Logging**: JSON format with correlation ID support
- **Security Headers**: X-Content-Type-Options, X-Frame-Options, HSTS
- **Retry Logic**: Automatic retries for HTTP/git operations (tenacity)
- **Graceful Shutdown**: Proper signal handling for SIGTERM/SIGINT
- **Error Sanitization**: Internal errors logged, sanitized responses to clients

## Notes

- All I/O is async (`aiosqlite`, `httpx`, `asyncio`)
- Default category is 3 (Pokemon)
- Authentication is disabled by default for backwards compatibility
- CORS allows all origins by default (configure `CARD_SCANNER_CORS_ORIGINS` for production)
