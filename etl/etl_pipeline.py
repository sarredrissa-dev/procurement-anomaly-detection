# =============================================================
# IntegriScan — ETL Pipeline
# Government Procurement Anomaly Detection System
# MSIT 5910 Capstone | University of the People
# Student: Edrissa Sarr
# =============================================================
# Processes: FY2023_097_Contracts_Full_20260706_1.csv to _5.csv
# Source: USASpending.gov — Department of Defense FY2023
# =============================================================

import os
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

# =============================================================
# CONFIGURATION
# =============================================================

# Load environment variables
load_dotenv()

# Data folder path
DATA_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'data')

# CSV files to process (all 5 parts)
CSV_FILES = [
    'FY2023_097_Contracts_Full_20260706_1.csv',
    'FY2023_097_Contracts_Full_20260706_2.csv',
    'FY2023_097_Contracts_Full_20260706_3.csv',
    'FY2023_097_Contracts_Full_20260706_4.csv',
    'FY2023_097_Contracts_Full_20260706_5.csv',
]

# Columns to keep from the 225-column CSV (reduces memory by ~92%)
COLUMNS_TO_KEEP = [
    'contract_award_unique_key',
    'award_id_piid',
    'recipient_name',
    'recipient_name_raw',
    'recipient_state_code',
    'federal_action_obligation',
    'total_dollars_obligated',
    'current_total_value_of_award',
    'potential_total_value_of_award',
    'action_date',
    'action_date_fiscal_year',
    'period_of_performance_start_date',
    'period_of_performance_current_end_date',
    'awarding_agency_name',
    'funding_agency_name',
    'award_type',
    'naics_code',
    'naics_description',
]

# Chunk size — number of rows loaded into memory at once
# 10,000 is safe for 16GB RAM
CHUNK_SIZE = 10_000

# Minimum contract value to include (removes $0 and negative records)
MIN_CONTRACT_VALUE = 1.0

# Null threshold — columns with more than this % nulls are flagged
NULL_THRESHOLD = 0.80

# =============================================================
# LOGGING SETUP
# =============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('etl_pipeline.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# =============================================================
# DATABASE CONNECTION
# =============================================================

def get_engine():
    """Create SQLAlchemy engine using .env credentials."""
    connection_url = URL.create(
        drivername="postgresql+psycopg2",
        username=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", 5432)),
        database=os.getenv("DB_NAME")
    )
    engine = create_engine(connection_url, pool_pre_ping=True)
    log.info("Database engine created successfully")
    return engine

# =============================================================
# STEP 1: EXTRACT
# Load CSV chunk by chunk, keeping only needed columns
# =============================================================

def extract_chunk(filepath, chunk_size=CHUNK_SIZE):
    """
    Generator that yields cleaned chunks from a CSV file.
    Loads only the columns we need to save memory.
    """
    log.info(f"Extracting: {os.path.basename(filepath)}")

    # Read only the columns we need
    reader = pd.read_csv(
        filepath,
        usecols=COLUMNS_TO_KEEP,
        chunksize=chunk_size,
        dtype={
            'naics_code': str,
            'award_id_piid': str,
            'contract_award_unique_key': str,
            'recipient_state_code': str,
            'action_date_fiscal_year': str,
        },
        low_memory=False,
        encoding='utf-8',
        on_bad_lines='skip'  # Skip malformed rows
    )

    for chunk in reader:
        yield chunk

# =============================================================
# STEP 2: TRANSFORM
# Clean, normalize, and prepare data for loading
# =============================================================

def transform_chunk(df):
    """
    Apply all transformations to a DataFrame chunk.
    Returns cleaned DataFrame ready for database loading.
    """

    # --- 2.1 Drop rows missing critical fields ---
    df = df.dropna(subset=['recipient_name', 'total_dollars_obligated'])

    # --- 2.2 Clean numeric fields ---
    df['total_dollars_obligated'] = pd.to_numeric(
        df['total_dollars_obligated'], errors='coerce'
    )
    df['current_total_value_of_award'] = pd.to_numeric(
        df['current_total_value_of_award'], errors='coerce'
    )
    df['potential_total_value_of_award'] = pd.to_numeric(
        df['potential_total_value_of_award'], errors='coerce'
    )

    # --- 2.3 Remove zero/negative value contracts ---
    df = df[df['total_dollars_obligated'] >= MIN_CONTRACT_VALUE]

    # --- 2.4 Normalize recipient name ---
    df['recipient_name'] = (
        df['recipient_name']
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Keep raw name for comparison
    df['recipient_name_raw'] = (
        df['recipient_name_raw']
        .astype(str)
        .str.strip()
    )

    # --- 2.5 Parse and normalize dates ---
    for date_col in [
        'action_date',
        'period_of_performance_start_date',
        'period_of_performance_current_end_date'
    ]:
        df[date_col] = pd.to_datetime(
            df[date_col], errors='coerce'
        )

    # --- 2.6 Extract ML features from dates ---
    df['award_day_of_year'] = df['action_date'].dt.dayofyear
    df['award_month'] = df['action_date'].dt.month
    df['fiscal_year'] = pd.to_numeric(
        df['action_date_fiscal_year'], errors='coerce'
    ).fillna(2023).astype(int)

    # --- 2.7 Clean NAICS code ---
    df['naics_code'] = (
        df['naics_code']
        .astype(str)
        .str.strip()
        .str[:6]  # NAICS codes are 6 digits
    )

    # --- 2.8 Clean agency names ---
    df['awarding_agency_name'] = (
        df['awarding_agency_name']
        .astype(str)
        .str.strip()
    )
    df['funding_agency_name'] = (
        df['funding_agency_name']
        .astype(str)
        .str.strip()
    )

    # --- 2.9 Clean state code ---
    df['recipient_state_code'] = (
        df['recipient_state_code']
        .astype(str)
        .str.strip()
        .str.upper()
        .str[:2]
    )

    # --- 2.10 Remove duplicates within chunk ---
    df = df.drop_duplicates(subset=['contract_award_unique_key'])

    return df

# =============================================================
# STEP 3: LOAD VENDORS
# Insert unique vendors, return vendor_id mapping
# =============================================================

def load_vendors(df, engine, vendor_cache):
    """
    Insert unique vendors into vendors table.
    Returns updated vendor cache {recipient_name: vendor_id}.
    Uses INSERT ... ON CONFLICT to handle duplicates safely.
    """
    # Get unique vendors in this chunk not already in cache
    unique_vendors = df[['recipient_name', 'recipient_name_raw', 'recipient_state_code']].drop_duplicates(
        subset=['recipient_name']
    )
    new_vendors = unique_vendors[
        ~unique_vendors['recipient_name'].isin(vendor_cache.keys())
    ]

    if new_vendors.empty:
        return vendor_cache

    with engine.begin() as conn:
        for _, row in new_vendors.iterrows():
            result = conn.execute(text("""
                INSERT INTO vendors (recipient_name, recipient_name_raw, state)
                VALUES (:name, :name_raw, :state)
                ON CONFLICT DO NOTHING
                RETURNING vendor_id, recipient_name
            """), {
                'name': row['recipient_name'],
                'name_raw': str(row['recipient_name_raw'])[:500],
                'state': str(row['recipient_state_code'])[:2]
                    if row['recipient_state_code'] not in ('nan', '') else None
            })
            row_result = result.fetchone()
            if row_result:
                vendor_cache[row_result[1]] = row_result[0]

    # Fetch any vendors that already existed (ON CONFLICT DO NOTHING)
    missing = [v for v in new_vendors['recipient_name'] if v not in vendor_cache]
    if missing:
        with engine.connect() as conn:
            for name in missing:
                result = conn.execute(text(
                    "SELECT vendor_id FROM vendors WHERE recipient_name = :name"
                ), {'name': name})
                row = result.fetchone()
                if row:
                    vendor_cache[name] = row[0]

    return vendor_cache

# =============================================================
# STEP 4: LOAD CONTRACTS
# Insert contract metadata, return contract_id mapping
# =============================================================

def load_contracts(df, engine, contract_cache):
    """
    Insert unique contracts into contracts table.
    Returns updated contract cache {award_unique_key: contract_id}.
    """
    unique_contracts = df.drop_duplicates(subset=['contract_award_unique_key'])
    new_contracts = unique_contracts[
        ~unique_contracts['contract_award_unique_key'].isin(contract_cache.keys())
    ]

    if new_contracts.empty:
        return contract_cache

    with engine.begin() as conn:
        for _, row in new_contracts.iterrows():
            result = conn.execute(text("""
                INSERT INTO contracts (
                    award_id, piid, awarding_agency, funding_agency,
                    contract_type, naics_code, naics_description,
                    period_start, period_end, award_day_of_year,
                    award_month, fiscal_year
                )
                VALUES (
                    :award_id, :piid, :awarding_agency, :funding_agency,
                    :contract_type, :naics_code, :naics_description,
                    :period_start, :period_end, :award_day_of_year,
                    :award_month, :fiscal_year
                )
                ON CONFLICT (award_id) DO NOTHING
                RETURNING contract_id, award_id
            """), {
                'award_id': str(row['contract_award_unique_key'])[:200],
                'piid': str(row['award_id_piid'])[:200]
                    if pd.notna(row['award_id_piid']) else None,
                'awarding_agency': str(row['awarding_agency_name'])[:300]
                    if pd.notna(row['awarding_agency_name']) else None,
                'funding_agency': str(row['funding_agency_name'])[:300]
                    if pd.notna(row['funding_agency_name']) else None,
                'contract_type': str(row['award_type'])[:200]
                    if pd.notna(row['award_type']) else None,
                'naics_code': str(row['naics_code'])[:20]
                    if pd.notna(row['naics_code']) else None,
                'naics_description': str(row['naics_description'])[:500]
                    if pd.notna(row['naics_description']) else None,
                'period_start': row['period_of_performance_start_date']
                    if pd.notna(row['period_of_performance_start_date']) else None,
                'period_end': row['period_of_performance_current_end_date']
                    if pd.notna(row['period_of_performance_current_end_date']) else None,
                'award_day_of_year': int(row['award_day_of_year'])
                    if pd.notna(row['award_day_of_year']) else None,
                'award_month': int(row['award_month'])
                    if pd.notna(row['award_month']) else None,
                'fiscal_year': int(row['fiscal_year'])
                    if pd.notna(row['fiscal_year']) else 2023,
            })
            row_result = result.fetchone()
            if row_result:
                contract_cache[row_result[1]] = row_result[0]

    # Fetch contracts that already existed
    missing = [k for k in new_contracts['contract_award_unique_key']
               if k not in contract_cache]
    if missing:
        with engine.connect() as conn:
            for award_id in missing:
                result = conn.execute(text(
                    "SELECT contract_id FROM contracts WHERE award_id = :award_id"
                ), {'award_id': str(award_id)[:200]})
                row = result.fetchone()
                if row:
                    contract_cache[award_id] = row[0]

    return contract_cache

# =============================================================
# STEP 5: LOAD PROCUREMENT RECORDS
# Insert the main fact records
# =============================================================

def load_records(df, engine, vendor_cache, contract_cache):
    """
    Insert procurement records into procurement_records table.
    Maps vendor_id and contract_id from caches.
    """
    records = []
    for _, row in df.iterrows():
        vendor_id = vendor_cache.get(row['recipient_name'])
        contract_id = contract_cache.get(row['contract_award_unique_key'])

        if not vendor_id or not contract_id:
            continue  # Skip if lookup failed

        records.append({
            'vendor_id': vendor_id,
            'contract_id': contract_id,
            'total_dollars_obligated': float(row['total_dollars_obligated'])
                if pd.notna(row['total_dollars_obligated']) else None,
            'current_total_value': float(row['current_total_value_of_award'])
                if pd.notna(row['current_total_value_of_award']) else None,
            'potential_total_value': float(row['potential_total_value_of_award'])
                if pd.notna(row['potential_total_value_of_award']) else None,
            'dataset_source': 'USASpending.gov FY2023 DoD',
            'is_synthetic_anomaly': False,
        })

    if not records:
        return 0

    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO procurement_records (
                vendor_id, contract_id,
                total_dollars_obligated, current_total_value,
                potential_total_value, dataset_source,
                is_synthetic_anomaly
            )
            VALUES (
                :vendor_id, :contract_id,
                :total_dollars_obligated, :current_total_value,
                :potential_total_value, :dataset_source,
                :is_synthetic_anomaly
            )
        """), records)

    return len(records)

# =============================================================
# STEP 6: SYNTHETIC ANOMALY INJECTION
# Inject realistic test anomalies after all real data loaded
# Based on Peer 1 & Peer 4 feedback:
# - Log-normal value injection at 99th percentile
# - Character-level vendor name perturbations
# =============================================================

def inject_synthetic_anomalies(engine):
    """
    Inject synthetic anomalies for ML model evaluation.

    Two types:
    1. VALUE ANOMALIES: contracts with amounts drawn from
       log-normal distribution anchored at 99th percentile
       of real category distribution [Peer 4]

    2. VENDOR DUPLICATES: near-duplicate vendor names using
       character-level perturbations [Peer 4]
    """
    log.info("Injecting synthetic anomalies...")

    with engine.begin() as conn:

        # --- Get category statistics for value injection ---
        stats = conn.execute(text("""
            SELECT
                c.naics_code,
                COUNT(pr.record_id) as record_count,
                AVG(pr.total_dollars_obligated) as mean_value,
                STDDEV(pr.total_dollars_obligated) as std_value,
                PERCENTILE_CONT(0.99) WITHIN GROUP
                    (ORDER BY pr.total_dollars_obligated) as p99_value
            FROM procurement_records pr
            JOIN contracts c ON pr.contract_id = c.contract_id
            WHERE pr.is_synthetic_anomaly = FALSE
              AND pr.total_dollars_obligated > 0
            GROUP BY c.naics_code
            HAVING COUNT(pr.record_id) >= 100
            ORDER BY record_count DESC
            LIMIT 10
        """)).fetchall()

        # --- Get top vendors for duplicate injection ---
        top_vendors = conn.execute(text("""
            SELECT vendor_id, recipient_name, state
            FROM vendors
            WHERE is_synthetic = FALSE
            ORDER BY vendor_frequency DESC
            LIMIT 5
        """)).fetchall()

        # --- Get a reference contract for synthetic records ---
        ref_contract = conn.execute(text("""
            SELECT contract_id, naics_code
            FROM contracts
            WHERE naics_code IS NOT NULL
            LIMIT 1
        """)).fetchone()

        if not ref_contract:
            log.warning("No reference contract found for synthetic injection")
            return 0

        synthetic_count = 0

        # --- TYPE 1: Value anomaly injection ---
        # Inject 50 price outlier records per top NAICS category
        for stat_row in stats[:5]:
            naics_code = stat_row[0]
            p99_value = float(stat_row[4]) if stat_row[4] else 1_000_000

            # Draw from log-normal distribution anchored at 99th percentile
            # sigma=0.5 gives realistic spread around the anchor point
            rng = np.random.default_rng(seed=42)
            injected_values = rng.lognormal(
                mean=np.log(p99_value * 3),  # 3x the 99th percentile
                sigma=0.5,
                size=10  # 10 injections per category
            )

            # Get a vendor and contract in this NAICS category
            vendor_contract = conn.execute(text("""
                SELECT pr.vendor_id, pr.contract_id
                FROM procurement_records pr
                JOIN contracts c ON pr.contract_id = c.contract_id
                WHERE c.naics_code = :naics_code
                LIMIT 1
            """), {'naics_code': naics_code}).fetchone()

            if not vendor_contract:
                continue

            for val in injected_values:
                conn.execute(text("""
                    INSERT INTO procurement_records (
                        vendor_id, contract_id,
                        total_dollars_obligated,
                        dataset_source,
                        is_synthetic_anomaly,
                        synthetic_anomaly_type
                    ) VALUES (
                        :vendor_id, :contract_id,
                        :amount,
                        'SYNTHETIC — Value Outlier Injection',
                        TRUE,
                        'value_outlier_lognormal'
                    )
                """), {
                    'vendor_id': vendor_contract[0],
                    'contract_id': vendor_contract[1],
                    'amount': round(float(val), 2)
                })
                synthetic_count += 1

        # --- TYPE 2: Vendor duplicate injection ---
        # Create near-duplicate vendor names using character perturbations
        # Mirrors known shell company patterns [Peer 4]
        perturbations = [
            lambda name: name.replace('LLC', 'L.L.C.'),
            lambda name: name.replace('INC', 'INC.'),
            lambda name: name.replace('CORP', 'CORP.'),
            lambda name: name + ' II',
            lambda name: name.replace(' ', '  '),  # double space
        ]

        for vendor_row in top_vendors:
            original_vendor_id = vendor_row[0]
            original_name = vendor_row[1]
            state = vendor_row[2]

            for i, perturb in enumerate(perturbations[:3]):
                try:
                    perturbed_name = perturb(original_name)
                    if perturbed_name == original_name:
                        continue

                    # Insert synthetic duplicate vendor
                    new_vendor = conn.execute(text("""
                        INSERT INTO vendors (
                            recipient_name, recipient_name_raw,
                            state, is_synthetic
                        ) VALUES (
                            :name, :name_raw, :state, TRUE
                        )
                        ON CONFLICT DO NOTHING
                        RETURNING vendor_id
                    """), {
                        'name': perturbed_name[:500],
                        'name_raw': original_name[:500],
                        'state': state
                    }).fetchone()

                    if not new_vendor:
                        continue

                    # Get a contract for this vendor
                    ref_rec = conn.execute(text("""
                        SELECT contract_id, total_dollars_obligated
                        FROM procurement_records
                        WHERE vendor_id = :vendor_id
                        LIMIT 1
                    """), {'vendor_id': original_vendor_id}).fetchone()

                    if not ref_rec:
                        continue

                    # Insert synthetic procurement record
                    conn.execute(text("""
                        INSERT INTO procurement_records (
                            vendor_id, contract_id,
                            total_dollars_obligated,
                            dataset_source,
                            is_synthetic_anomaly,
                            synthetic_anomaly_type
                        ) VALUES (
                            :vendor_id, :contract_id,
                            :amount,
                            'SYNTHETIC — Vendor Duplicate Injection',
                            TRUE,
                            'vendor_duplicate_perturbation'
                        )
                    """), {
                        'vendor_id': new_vendor[0],
                        'contract_id': ref_rec[0],
                        'amount': float(ref_rec[1])
                    })
                    synthetic_count += 1

                except Exception as e:
                    log.warning(f"Perturbation failed: {e}")
                    continue

    log.info(f"Synthetic anomaly injection complete: {synthetic_count} records injected")
    return synthetic_count

# =============================================================
# STEP 7: UPDATE VENDOR FREQUENCIES
# Count how many contracts each vendor received
# Used as ML feature for detecting high-frequency vendors
# =============================================================

def update_vendor_frequencies(engine):
    """Update vendor_frequency count after all records loaded."""
    log.info("Updating vendor frequencies...")
    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE vendors v
            SET vendor_frequency = sub.freq
            FROM (
                SELECT vendor_id, COUNT(*) as freq
                FROM procurement_records
                WHERE is_synthetic_anomaly = FALSE
                GROUP BY vendor_id
            ) sub
            WHERE v.vendor_id = sub.vendor_id
        """))
    log.info("Vendor frequencies updated")

# =============================================================
# STEP 8: COMPUTE Z-SCORES
# Pre-compute z-score for each record within its NAICS category
# =============================================================

def compute_zscores(engine):
    """
    Pre-compute z-scores for total_dollars_obligated
    within each NAICS category.
    Stored in procurement_records for fast ML processing.
    """
    log.info("Computing z-scores by NAICS category...")
    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE procurement_records pr
            SET
                zscore_within_category = sub.zscore,
                category_mean = sub.cat_mean,
                category_std = sub.cat_std
            FROM (
                SELECT
                    pr2.record_id,
                    AVG(pr2.total_dollars_obligated)
                        OVER (PARTITION BY c.naics_code) as cat_mean,
                    STDDEV(pr2.total_dollars_obligated)
                        OVER (PARTITION BY c.naics_code) as cat_std,
                    CASE
                        WHEN STDDEV(pr2.total_dollars_obligated)
                            OVER (PARTITION BY c.naics_code) > 0
                        THEN (pr2.total_dollars_obligated -
                              AVG(pr2.total_dollars_obligated)
                              OVER (PARTITION BY c.naics_code))
                             / STDDEV(pr2.total_dollars_obligated)
                             OVER (PARTITION BY c.naics_code)
                        ELSE 0
                    END as zscore
                FROM procurement_records pr2
                JOIN contracts c ON pr2.contract_id = c.contract_id
                WHERE pr2.is_synthetic_anomaly = FALSE
                  AND pr2.total_dollars_obligated IS NOT NULL
            ) sub
            WHERE pr.record_id = sub.record_id
        """))
    log.info("Z-scores computed and stored")

# =============================================================
# MAIN ETL RUNNER
# =============================================================

def run_etl():
    """Main ETL orchestration function."""

    start_time = time.time()
    log.info("=" * 60)
    log.info("IntegriScan ETL Pipeline Starting")
    log.info(f"Processing {len(CSV_FILES)} CSV files")
    log.info("=" * 60)

    # Connect to database
    engine = get_engine()

    # Caches to avoid redundant DB lookups
    vendor_cache = {}
    contract_cache = {}

    # Counters
    total_records_loaded = 0
    total_chunks = 0
    total_rows_skipped = 0

    # Process each CSV file
    for file_num, filename in enumerate(CSV_FILES, 1):
        filepath = os.path.join(DATA_FOLDER, filename)

        if not os.path.exists(filepath):
            log.warning(f"File not found, skipping: {filename}")
            continue

        log.info(f"\n{'='*40}")
        log.info(f"Processing file {file_num}/{len(CSV_FILES)}: {filename}")
        log.info(f"{'='*40}")

        file_records = 0
        chunk_num = 0

        try:
            for chunk in extract_chunk(filepath):
                chunk_num += 1
                total_chunks += 1
                rows_before = len(chunk)

                # Transform
                chunk = transform_chunk(chunk)
                rows_after = len(chunk)
                rows_skipped = rows_before - rows_after
                total_rows_skipped += rows_skipped

                if chunk.empty:
                    continue

                # Load vendors
                vendor_cache = load_vendors(chunk, engine, vendor_cache)

                # Load contracts
                contract_cache = load_contracts(chunk, engine, contract_cache)

                # Load procurement records
                records_loaded = load_records(
                    chunk, engine, vendor_cache, contract_cache
                )
                file_records += records_loaded
                total_records_loaded += records_loaded

                # Progress update every 50 chunks
                if chunk_num % 50 == 0:
                    elapsed = time.time() - start_time
                    log.info(
                        f"  Chunk {chunk_num} | "
                        f"File records: {file_records:,} | "
                        f"Total: {total_records_loaded:,} | "
                        f"Elapsed: {elapsed:.0f}s"
                    )

        except Exception as e:
            log.error(f"Error processing {filename}: {e}")
            continue

        log.info(f"File {file_num} complete: {file_records:,} records loaded")

    # Post-load operations
    log.info("\n" + "="*60)
    log.info("All CSV files processed. Running post-load operations...")

    # Update vendor frequencies
    update_vendor_frequencies(engine)

    # Compute z-scores
    compute_zscores(engine)

    # Inject synthetic anomalies
    synthetic_count = inject_synthetic_anomalies(engine)

    # Final statistics
    elapsed_total = time.time() - start_time
    log.info("\n" + "="*60)
    log.info("ETL PIPELINE COMPLETE")
    log.info(f"Total records loaded:       {total_records_loaded:,}")
    log.info(f"Total rows skipped:         {total_rows_skipped:,}")
    log.info(f"Synthetic anomalies:        {synthetic_count:,}")
    log.info(f"Unique vendors cached:      {len(vendor_cache):,}")
    log.info(f"Unique contracts cached:    {len(contract_cache):,}")
    log.info(f"Total time:                 {elapsed_total:.0f}s "
             f"({elapsed_total/60:.1f} minutes)")
    log.info("="*60)

    # Final DB verification
    with engine.connect() as conn:
        counts = conn.execute(text("""
            SELECT
                (SELECT COUNT(*) FROM vendors) as vendors,
                (SELECT COUNT(*) FROM contracts) as contracts,
                (SELECT COUNT(*) FROM procurement_records) as records,
                (SELECT COUNT(*) FROM procurement_records
                 WHERE is_synthetic_anomaly = TRUE) as synthetic
        """)).fetchone()
        log.info(f"\nDatabase Summary:")
        log.info(f"  Vendors:              {counts[0]:,}")
        log.info(f"  Contracts:            {counts[1]:,}")
        log.info(f"  Procurement Records:  {counts[2]:,}")
        log.info(f"  Synthetic Anomalies:  {counts[3]:,}")

    return total_records_loaded

# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":
    run_etl()
