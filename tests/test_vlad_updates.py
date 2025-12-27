import asyncio
import numpy as np
from vlad_matcher import VLADCardSearch
from scanner import Scanner

async def test_updates():
    print("Testing VLADCardSearch updates...")
    
    # 1. Test VLADCardSearch manual sync and reload
    searcher = VLADCardSearch()
    initial_db_size = len(searcher.database)
    print(f"Initial Database size: {initial_db_size} cards")
    
    # Manually trigger sync and reload
    searcher.sync_and_reload()
    reloaded_db_size = len(searcher.database)
    print(f"Reloaded Database size: {reloaded_db_size} cards")
    
    if reloaded_db_size == initial_db_size:
        print("Success: Database reloaded correctly (size unchanged as expected).")
    else:
        print(f"Warning: Database size changed from {initial_db_size} to {reloaded_db_size}")

    # 2. Test Scanner proxy
    scanner = Scanner()
    print("Starting scheduled updates via Scanner...")
    scanner.start_scheduled_updates()
    
    if scanner.matcher.update_task and not scanner.matcher.update_task.done():
        print("Success: Background update task started.")
        scanner.matcher.update_task.cancel()
    else:
        print("Error: Background update task failed to start.")
        exit(1)

if __name__ == "__main__":
    asyncio.run(test_updates())
