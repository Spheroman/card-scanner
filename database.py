"""
Handles the database for the card scanner.
Loads files from tcgcsv.com/tcgplayer/{categoryid}/{groupid}/ProductsAndPrices.csv
Saves the files to categories/{categoryid}/{groupid}/ProductsAndPrices.csv
Loads the data into an asynchronous sqlite database.
Updates the database every 24 hours at a set time.
"""
import asyncio
import aiosqlite
import httpx
import csv
import pickle
import functools
from datetime import datetime, time as dt_time
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE = "database.db"
CSV_PATH = "categories/"
URL = "https://tcgcsv.com/tcgplayer/"
UPDATE_TIME = dt_time(hour=3, minute=0)  # 3 AM daily update
MAX_CONCURRENT_DOWNLOADS = 5

def check_connection(func):
    """Decorator to ensure database connection is open before method execution."""
    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        if not self.is_initialized:
            logger.error(f"Database method {func.__name__} called before initialization")
            raise RuntimeError(f"Database connection not initialized. Call open() first.")
        return await func(self, *args, **kwargs)
    return wrapper

class Database:
    def __init__(self, categories=None):
        self.categories = categories or [3]
        self.conn = None
        self.update_task = None
        self.client = httpx.AsyncClient(timeout=30.0)
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
    
    @property
    def is_initialized(self):
        """Check if the database connection is initialized and open."""
        return self.conn is not None
        
    async def __aenter__(self):
        await self.open()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def open(self):
        """Open database connection and create tables if needed."""
        try:
            self.conn = await aiosqlite.connect(DATABASE)
            await self._create_tables()
            logger.info("Database connection opened")
        except Exception as e:
            logger.error(f"Failed to open database: {e}")
            raise
    
    async def close(self):
        """Close database connection and cancel update task."""
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        
        await self.client.aclose()
        
        if self.conn:
            await self.conn.close()
            logger.info("Database connection closed")
    
    @check_connection
    async def _create_tables(self):
        """Create database tables with actual CSV columns."""
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS products (
                product_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                clean_name TEXT,
                image_url TEXT,
                category_id INTEGER NOT NULL,
                group_id INTEGER NOT NULL,
                url TEXT,
                modified_on TEXT,
                image_count INTEGER,
                low_price REAL,
                mid_price REAL,
                high_price REAL,
                market_price REAL,
                direct_low_price REAL,
                sub_type_name TEXT,
                ext_data BLOB,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(product_id)
            )
        """)
        
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS groups (
                group_id INTEGER PRIMARY KEY,
                category_id INTEGER NOT NULL,
                group_name TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS categories (
                category_id INTEGER PRIMARY KEY,
                category_name TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        await self.conn.commit()
        logger.info("Database tables created/verified")
    
    def initialize(self):
        """Initialize directory structure for CSV storage."""
        Path(CSV_PATH).mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized directory structure at {CSV_PATH}")
    
    async def download_csv(self, category_id, group_id):
        """Download CSV file for a specific category and group."""
        url = f"{URL}{category_id}/{group_id}/ProductsAndPrices.csv"
        save_path = Path(CSV_PATH) / str(category_id) / str(group_id)
        save_path.mkdir(parents=True, exist_ok=True)
        
        file_path = save_path / "ProductsAndPrices.csv"
        
        async with self.semaphore:
            try:
                response = await self.client.get(url)
                response.raise_for_status()
                
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                
                logger.info(f"Downloaded CSV for category {category_id}, group {group_id}")
                return file_path
            except Exception as e:
                logger.error(f"Failed to download CSV for {category_id}/{group_id}: {e}")
                return None
    
    @check_connection
    async def download_groups(self, category_id):
        """Download and store groups for a specific category."""
        url = f"{URL}{category_id}/Groups.csv"
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            
            # Save file locally
            save_path = Path(CSV_PATH) / str(category_id)
            save_path.mkdir(parents=True, exist_ok=True)
            file_path = save_path / "Groups.csv"
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            # Load into database
            content = response.content.decode('utf-8').splitlines()
            reader = csv.DictReader(content)
            
            batch = []
            now = datetime.now()
            for row in reader:
                group_id = int(row.get('groupId', 0))
                group_name = row.get('name', '')
                if group_id:
                    batch.append((group_id, category_id, group_name, now))
            
            if batch:
                await self.conn.executemany("""
                    INSERT OR REPLACE INTO groups (group_id, category_id, group_name, last_updated)
                    VALUES (?, ?, ?, ?)
                """, batch)
                await self.conn.commit()
            
            logger.info(f"Downloaded and loaded groups for category {category_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to download groups for category {category_id}: {e}")
            return False
    
    @check_connection
    async def download_categories(self):
        """Download and store all available categories."""
        url = f"{URL}Categories.csv"
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            
            # Save file locally
            file_path = Path(CSV_PATH) / "Categories.csv"
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            # Load into database
            content = response.content.decode('utf-8').splitlines()
            reader = csv.DictReader(content)
            
            batch = []
            now = datetime.now()
            for row in reader:
                cat_id = int(row.get('categoryId', 0))
                cat_name = row.get('name', '')
                if cat_id:
                    batch.append((cat_id, cat_name, now))
            
            if batch:
                await self.conn.executemany("""
                    INSERT OR REPLACE INTO categories (category_id, category_name, last_updated)
                    VALUES (?, ?, ?)
                """, batch)
                await self.conn.commit()
            
            logger.info("Downloaded and loaded categories")
            return True
        except Exception as e:
            logger.error(f"Failed to download categories: {e}")
            return False
    
    @check_connection
    async def load_csv_to_db(self, file_path, category_id, group_id):
        """Load CSV data into the database."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                batch = []
                now = datetime.now()
                for row in reader:
                    # Dynamically collect all columns that start with 'ext'
                    ext_data = {
                        key: value 
                        for key, value in row.items() 
                        if key.startswith('ext')
                    }
                    
                    # Pickle the extended data
                    pickled_ext_data = pickle.dumps(ext_data)
                    
                    batch.append((
                        int(row.get('productId', 0)),
                        row.get('name', ''),
                        row.get('cleanName', ''),
                        row.get('imageUrl', ''),
                        int(row.get('categoryId', category_id)),
                        int(row.get('groupId', group_id)),
                        row.get('url', ''),
                        row.get('modifiedOn', ''),
                        int(row.get('imageCount', 0) or 0),
                        float(row.get('lowPrice', 0) or 0),
                        float(row.get('midPrice', 0) or 0),
                        float(row.get('highPrice', 0) or 0),
                        float(row.get('marketPrice', 0) or 0),
                        float(row.get('directLowPrice', 0) or 0),
                        row.get('subTypeName', ''),
                        pickled_ext_data,
                        now
                    ))
                
                if batch:
                    await self.conn.executemany("""
                        INSERT OR REPLACE INTO products 
                        (product_id, name, clean_name, image_url, category_id, group_id,
                         url, modified_on, image_count, low_price, mid_price, high_price,
                         market_price, direct_low_price, sub_type_name, ext_data, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, batch)
                    await self.conn.commit()
                
                logger.info(f"Loaded CSV data from {file_path} into database")
        except Exception as e:
            logger.error(f"Failed to load CSV to database: {e}")
            raise
    
    async def process_group(self, category_id, group_id):
        """Helper to download and load a group's CSV."""
        file_path = await self.download_csv(category_id, group_id)
        if file_path:
            await self.load_csv_to_db(file_path, category_id, group_id)

    @check_connection
    async def update(self):
        """Download CSVs and update the database for all categories in parallel."""
        logger.info("Starting database update")
        
        # 1. Update categories
        await self.download_categories()
        
        for category_id in self.categories:
            # 2. Update groups for this category
            await self.download_groups(category_id)
            
            # 3. Get all groups from database to process
            async with self.conn.execute(
                "SELECT group_id FROM groups WHERE category_id = ?", 
                (category_id,)
            ) as cursor:
                groups = await cursor.fetchall()
            
            # Process groups in parallel
            tasks = [self.process_group(category_id, group_id) for (group_id,) in groups]
            await asyncio.gather(*tasks)
        
        logger.info("Database update completed")
    
    async def scheduled_update(self):
        """Run updates at scheduled time every 24 hours."""
        while True:
            now = datetime.now()
            target = datetime.combine(now.date(), UPDATE_TIME)
            
            # If target time has passed today, schedule for tomorrow
            if now.time() > UPDATE_TIME:
                target = datetime.combine(now.date(), UPDATE_TIME)
                target = target.replace(day=target.day + 1)
            
            sleep_seconds = (target - now).total_seconds()
            logger.info(f"Next update scheduled in {sleep_seconds/3600:.2f} hours")
            
            await asyncio.sleep(sleep_seconds)
            await self.update()
    
    def start_scheduled_updates(self):
        """Start the scheduled update background task."""
        if self.update_task is None or self.update_task.done():
            self.update_task = asyncio.create_task(self.scheduled_update())
            logger.info("Scheduled updates started")
    
    @check_connection
    async def query_product(self, product_name):
        """Query a product by name."""
        async with self.conn.execute(
            "SELECT * FROM products WHERE name LIKE ?",
            (f"%{product_name}%",)
        ) as cursor:
            return await cursor.fetchall()
    
    @check_connection
    async def query_by_id(self, product_id):
        """Get a product by ID and unpickle extended data."""
        async with self.conn.execute(
            "SELECT * FROM products WHERE product_id = ?",
            (product_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                # Unpickle the ext_data column (index 15)
                row_list = list(row)
                if row_list[15]:  # ext_data column
                    row_list[15] = pickle.loads(row_list[15])
                return tuple(row_list)
            return None
    
    @check_connection
    async def query_by_category(self, category_id):
        """Get all products for a category."""
        async with self.conn.execute(
            "SELECT * FROM products WHERE category_id = ?",
            (category_id,)
        ) as cursor:
            return await cursor.fetchall()
    
    @check_connection
    async def get_categories(self):
        """Get all categories stored in the database."""
        async with self.conn.execute(
            "SELECT * FROM categories"
        ) as cursor:
            return await cursor.fetchall()

    def return_columns(self):
        """Return the column names for the products table."""
        return [
            'product_id', 'name', 'clean_name', 'image_url', 'category_id',
            'group_id', 'url', 'modified_on', 'image_count', 'low_price',
            'mid_price', 'high_price', 'market_price', 'direct_low_price',
            'sub_type_name', 'ext_data', 'last_updated'
        ]
    
    @check_connection
    async def return_ext_data(self, product_id):
        """Return unpickled extended data for a product."""
        async with self.conn.execute(
            "SELECT ext_data FROM products WHERE product_id = ?",
            (product_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row and row[0]:
                return pickle.loads(row[0])
            return None


# Example usage
async def main():
    db = Database(categories=[3])
    db.initialize()
    
    async with db:
        # Initial update
        await db.update()
        
        # Start scheduled updates
        db.start_scheduled_updates()
        
        # Query example
        results = await db.query_product("example card name")
        print(results)
        
        # Keep running (in production, this would be handled by your main application loop)
        await asyncio.sleep(3600)  # Run for 1 hour as example


if __name__ == "__main__":
    asyncio.run(main())