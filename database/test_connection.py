# test_connection.py
# Tests .env file loading and PostgreSQL connection
# IntegriScan — MSIT 5910 Capstone

from dotenv import load_dotenv
import os
from sqlalchemy import create_engine, text

# Load .env file
load_dotenv()

# Step 1: Check .env variables loaded
print("=== Checking .env file ===")
print(f"DB_HOST:     {os.getenv('DB_HOST')}")
print(f"DB_PORT:     {os.getenv('DB_PORT')}")
print(f"DB_NAME:     {os.getenv('DB_NAME')}")
print(f"DB_USER:     {os.getenv('DB_USER')}")
print(f"DB_PASSWORD: {'***' if os.getenv('DB_PASSWORD') else 'NOT FOUND'}")

# Step 2: Test database connection
print("\n=== Testing Database Connection ===")
try:
    from sqlalchemy.engine import URL

    connection_url = URL.create(
        drivername="postgresql+psycopg2",
        username=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT")),
        database=os.getenv("DB_NAME")
    )

    engine = create_engine(connection_url)

    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema = 'public'"
        ))
        count = result.scalar()
        print(f"Connected successfully!")
        print(f"Tables and views in procurement_db: {count}")

        # Also show table names
        tables = conn.execute(text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' ORDER BY table_name"
        ))
        print("\nObjects found:")
        for row in tables:
            print(f"  - {row[0]}")

except Exception as e:
    print(f"Connection FAILED: {e}")

print("\n=== Test Complete ===")