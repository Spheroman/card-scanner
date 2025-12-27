"""
Handles the API for the card scanner.
Uses FastAPI to create a REST API.
/scan - scans a card from file upload
/price - gets the price of a card
/columns - gets the columns of the database
/ext-data - gets the extra data for a category
/update - updates the database
"""

import fastapi
from fastapi import UploadFile
import uvicorn
import database
import cv2
import numpy as np
import pickle
import asyncio
import os
from typing import List, Optional
from pydantic import BaseModel
from contextlib import asynccontextmanager
from logging import getLogger
from scanner import Scanner

logger = getLogger(__name__)

@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    # Startup
    scanner = Scanner()
    db = database.Database(categories=[3])
    db.initialize()
    
    try:
        await db.open()
        await db.update()  # Initial update
        db.start_scheduled_updates()
        scanner.start_scheduled_updates()
        app.state.db = db
        app.state.scanner = scanner
        
        logger.info("Application startup complete")
        yield
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    finally:
        # Shutdown - always runs
        logger.info("Shutting down...")
        await db.close()
        logger.info("Shutdown complete")

app = fastapi.FastAPI(lifespan=lifespan)

class ScanResult(BaseModel):
    card_id: int
    similarity: float
    box: List[float]
    details: Optional[dict] = None

@app.post("/scan", response_model=List[ScanResult])
async def scan(image: UploadFile = fastapi.File(...), top_n: int = 3):
    """
    Scan an image from file upload. Returns all data for all cards detected in JSON format.
    Args:
        image: The image file to scan
        top_n: The number of top matches to return per card
    Returns:
        JSON object containing all data for all cards detected
    """
    try:
        # Read the file into memory
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise fastapi.HTTPException(status_code=400, detail="Invalid image file")

        # Use scanner to detect and match cards
        scanner = app.state.scanner
        db = app.state.db
        
        detected_cards = scanner.scan(img, k=top_n)
        results = []
        
        cols = db.return_columns()
        
        for card_segment in detected_cards:
            box = card_segment['box']
            for match in card_segment['matches']:
                product_id = match['card_id']
                similarity = match['similarity']
                
                # Query DB for product details
                product_data = await db.query_by_id(product_id)
                details = None
                if product_data:
                    details = dict(zip(cols, product_data))
                
                results.append(ScanResult(
                    card_id=product_id,
                    similarity=similarity,
                    box=box,
                    details=details
                ))
            
        return results
        
    except Exception as e:
        logger.error(f"Scan failed: {e}")
        raise fastapi.HTTPException(status_code=500, detail=str(e))

@app.post("/identify", response_model=List[ScanResult])
async def identify(image: UploadFile = fastapi.File(...), top_n: int = 3):
    """
    Identify a pre-cropped card image. Skips YOLO detection.
    Args:
        image: The image file of the card
        top_n: The number of top matches to return
    Returns:
        JSON object containing data for the identified card matches
    """
    try:
        # Read the file into memory
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise fastapi.HTTPException(status_code=400, detail="Invalid image file")

        scanner = app.state.scanner
        db = app.state.db
        
        # Use identify_card
        card_result = scanner.identify_card(img, k=top_n)
        results = []
        
        cols = db.return_columns()
        box = card_result['box']
        
        for match in card_result['matches']:
            product_id = match['card_id']
            similarity = match['similarity']
            
            # Query DB for product details
            product_data = await db.query_by_id(product_id)
            details = None
            if product_data:
                details = dict(zip(cols, product_data))
            
            results.append(ScanResult(
                card_id=product_id,
                similarity=similarity,
                box=box,
                details=details
            ))
            
        return results
        
    except Exception as e:
        logger.error(f"Identification failed: {e}")
        raise fastapi.HTTPException(status_code=500, detail=str(e))

@app.get("/price")
async def price(product_id: int):
    """
    Get the price of a card.
    Args:
        product_id: The product ID of the card
    Returns:
        JSON object containing the price of the card
    """
    db = app.state.db
    product_data = await db.query_by_id(product_id)
    if not product_data:
        raise fastapi.HTTPException(status_code=404, detail="Product not found")
    
    cols = db.return_columns()
    product_dict = dict(zip(cols, product_data))
    
    price_cols = ['low_price', 'mid_price', 'high_price', 'market_price', 'direct_low_price']
    return {col: product_dict[col] for col in price_cols}

@app.get("/categories")
async def get_categories():
    """
    Get the categories in the database.
    Returns:
        JSON object containing the categories in the database
    """
    db = app.state.db
    cats = await db.get_categories()
    return {"categories": [{"category_id": c[0], "category_name": c[1]} for c in cats]}

@app.get("/columns")
async def columns():
    """
    Get the columns in the database.
    Returns:
        JSON object containing the columns in the database
    """
    db = app.state.db
    return {"columns": db.return_columns()}

@app.get("/ext-data")
async def ext_data(category_id: int = 3):
    """
    Get the extended data column names for a category
    Args:
        category_id: The category ID to get the extended data column names for
    Returns:
        JSON object containing the extended data column names for the category
    """
    db = app.state.db
    # Get a sample product from this category to see its ext_data keys
    async with db.conn.execute(
        "SELECT ext_data FROM products WHERE category_id = ? LIMIT 1",
        (category_id,)
    ) as cursor:
        row = await cursor.fetchone()
        if row and row[0]:
            ext_dict = pickle.loads(row[0])
            return {"ext_data_columns": list(ext_dict.keys())}
    
    return {"ext_data_columns": []}

@app.post("/update")
async def update():
    """
    Update the database.
    Returns:
        JSON object containing the status of the update
    """
    db = app.state.db
    try:
        # We trigger update in background to not block the request
        asyncio.create_task(db.update())
        return {"status": "Update started"}
    except Exception as e:
        logger.error(f"Update trigger failed: {e}")
        raise fastapi.HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
