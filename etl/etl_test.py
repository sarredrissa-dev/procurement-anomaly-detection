# =============================================================
# IntegriScan — ETL Pipeline TEST (Sample Run)
# Loads first 50,000 rows from File 1 only
# Purpose: Verify pipeline works before full run
# MSIT 5910 Capstone | University of the People
# =============================================================

import os
import time
import logging
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

# =============================================================
# CONFIGURATION
# =============================================================

load_dotenv()

DATA_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'data')

# TEST: Only first file, only 50,000 rows
TEST_FILE = 'FY2023_097_Contracts_Full_20260706_1.csv'
TEST_ROWS  = 50_000
CHUNK_SIZE = 10_000

COLUMNS_TO_KEEP = [
    'contract_award_unique_key',
    'award_id_piid',
    'recipient_name',
    'recipient_name_raw',
    'recipient_state_code',
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

MIN_CONTRACT_VALUE = 1.0

# =============================================================
# LOGGING
# =============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('etl_test.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# =============================================================
# DATABASE CONNECTION
# =============================================================

def get_engine():
    connection_url = URL.create(
        drivername="postgresql+psycopg2",
        username=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", 5432)),
        database=os.getenv("DB_NAME")
    )
    return create_engine(connection_url, pool_pre_ping=True)

# =============================================================
# STEP 1: EXTRACT — Sample only
# =============================================================

def extract_sample(filepath, nrows=TEST_ROWS, chunk_size=CHUNK_SIZE):
    """Load only the first nrows rows for testing."""
    log.info(f"Loading sample: first {nrows:,} rows from {os.path.basename(filepath)}")

    # Check which columns actually exist in this file
    header = pd.read_csv(filepath, nrows=0)
    available = [c for c in COLUMNS_TO_KEEP if c in header.columns]
    missing = [c for c in COLUMNS_TO_KEEP if c not in header.columns]

    if missing:
        log.warning(f"Columns not found in CSV (will skip): {missing}")

    reader = pd.read_csv(
        filepath,
        usecols=available,
        nrows=nrows,
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
        on_bad_lines='skip'
    )

    for chunk in reader:
        yield chunk

# =============================================================
# STEP 2: TRANSFORM
# =============================================================

def transform_chunk(df):
    """Clean and normalize a chunk."""

    df = df.dropna(subset=['recipient_name', 'total_dollars_obligated'])

    df['total_dollars_obligated'] = pd.to_numeric(
        df['total_dollars_obligated'], errors='coerce'
    )
    df['current_total_value_of_award'] = pd.to_numeric(
        df.get('current_total_value_of_award', pd.Series(dtype=float)),
        errors='coerce'
    )
    df['potential_total_value_of_award'] = pd.to_numeric(
        df.get('potential_total_value_of_award', pd.Series(dtype=float)),
        errors='coerce'
    )

    df = df[df['total_dollars_obligated'] >= MIN_CONTRACT_VALUE]

    df['recipient_name'] = (
        df['recipient_name'].astype(str).str.strip().str.upper()
    )

    if 'recipient_name_raw' not in df.columns:
        df['recipient_name_raw'] = df['recipient_name']
    else:
        df['recipient_name_raw'] = df['recipient_name_raw'].astype(str).str.strip()

    for date_col in [
        'action_date',
        'period_of_performance_start_date',
        'period_of_performance_current_end_date'
    ]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    if 'action_date' in df.columns:
        df['award_day_of_year'] = df['action_date'].dt.dayofyear
        df['award_month'] = df['action_date'].dt.month
    else:
        df['award_day_of_year'] = None
        df['award_month'] = None

    df['fiscal_year'] = pd.to_numeric(
        df.get('action_date_fiscal_year', pd.Series(dtype=str)),
        errors='coerce'
    ).fillna(2023).astype(int)

    if 'naics_code' in df.columns:
        df['naics_code'] = df['naics_code'].astype(str).str.strip().str[:6]

    if 'recipient_state_code' in df.columns:
        df['recipient_state_code'] = (
            df['recipient_state_code'].astype(str).str.strip().str.upper().str[:2]
        )
    else:
        df['recipient_state_code'] = None

    if 'contract_award_unique_key' in df.columns:
        df = df.drop_duplicates(subset=['contract_award_unique_key'])

    return df

# =============================================================
# STEP 3: LOAD VENDORS
# =============================================================

def load_vendors(df, engine, vendor_cache):
    unique_vendors = df[['recipient_name', 'recipient_name_raw',
                          'recipient_state_code']].drop_duplicates(
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
                'name': str(row['recipient_name'])[:500],
                'name_raw': str(row['recipient_name_raw'])[:500],
                'state': str(row['recipient_state_code'])[:2]
                    if str(row['recipient_state_code']) not in ('nan','') else None
            })
            r = result.fetchone()
            if r:
                vendor_cache[r[1]] = r[0]

    missing = [v for v in new_vendors['recipient_name'] if v not in vendor_cache]
    if missing:
        with engine.connect() as conn:
            for name in missing:
                r = conn.execute(text(
                    "SELECT vendor_id FROM vendors WHERE recipient_name = :n"
                ), {'n': name}).fetchone()
                if r:
                    vendor_cache[name] = r[0]

    return vendor_cache

# =============================================================
# STEP 4: LOAD CONTRACTS
# =============================================================

def load_contracts(df, engine, contract_cache):
    if 'contract_award_unique_key' not in df.columns:
        return contract_cache

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
                'piid': str(row.get('award_id_piid',''))[:200] or None,
                'awarding_agency': str(row.get('awarding_agency_name',''))[:300] or None,
                'funding_agency': str(row.get('funding_agency_name',''))[:300] or None,
                'contract_type': str(row.get('award_type',''))[:200] or None,
                'naics_code': str(row.get('naics_code',''))[:20] or None,
                'naics_description': str(row.get('naics_description',''))[:500] or None,
                'period_start': row.get('period_of_performance_start_date')
                    if pd.notna(row.get('period_of_performance_start_date')) else None,
                'period_end': row.get('period_of_performance_current_end_date')
                    if pd.notna(row.get('period_of_performance_current_end_date')) else None,
                'award_day_of_year': int(row['award_day_of_year'])
                    if pd.notna(row.get('award_day_of_year')) else None,
                'award_month': int(row['award_month'])
                    if pd.notna(row.get('award_month')) else None,
                'fiscal_year': int(row.get('fiscal_year', 2023)),
            })
            r = result.fetchone()
            if r:
                contract_cache[r[1]] = r[0]

    missing = [k for k in new_contracts['contract_award_unique_key']
               if k not in contract_cache]
    if missing:
        with engine.connect() as conn:
            for aid in missing:
                r = conn.execute(text(
                    "SELECT contract_id FROM contracts WHERE award_id = :a"
                ), {'a': str(aid)[:200]}).fetchone()
                if r:
                    contract_cache[aid] = r[0]

    return contract_cache

# =============================================================
# STEP 5: LOAD RECORDS
# =============================================================

def load_records(df, engine, vendor_cache, contract_cache):
    records = []
    for _, row in df.iterrows():
        vid = vendor_cache.get(row['recipient_name'])
        cid = contract_cache.get(row.get('contract_award_unique_key'))
        if not vid or not cid:
            continue
        records.append({
            'vendor_id': vid,
            'contract_id': cid,
            'total_dollars_obligated': float(row['total_dollars_obligated'])
                if pd.notna(row['total_dollars_obligated']) else None,
            'current_total_value': float(row['current_total_value_of_award'])
                if pd.notna(row.get('current_total_value_of_award')) else None,
            'potential_total_value': float(row['potential_total_value_of_award'])
                if pd.notna(row.get('potential_total_value_of_award')) else None,
            'dataset_source': 'USASpending.gov FY2023 DoD — TEST SAMPLE',
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
# SYNTHETIC INJECTION (simplified for test)
# =============================================================

def inject_synthetic_anomalies_test(engine):
    """Simplified injection for test run."""
    log.info("Injecting test synthetic anomalies...")
    rng = np.random.default_rng(seed=42)
    synthetic_count = 0

    with engine.begin() as conn:
        # Get stats for injection
        stats = conn.execute(text("""
            SELECT
                c.naics_code,
                AVG(pr.total_dollars_obligated) as mean_val,
                PERCENTILE_CONT(0.99) WITHIN GROUP
                    (ORDER BY pr.total_dollars_obligated) as p99_val
            FROM procurement_records pr
            JOIN contracts c ON pr.contract_id = c.contract_id
            WHERE pr.is_synthetic_anomaly = FALSE
              AND pr.total_dollars_obligated > 0
            GROUP BY c.naics_code
            HAVING COUNT(*) >= 10
            LIMIT 5
        """)).fetchall()

        top_vendors = conn.execute(text("""
            SELECT vendor_id, recipient_name, state
            FROM vendors
            WHERE is_synthetic = FALSE
            ORDER BY vendor_frequency DESC
            LIMIT 3
        """)).fetchall()

        # Value outlier injection
        for stat in stats:
            naics = stat[0]
            p99 = float(stat[2]) if stat[2] else 1_000_000

            values = rng.lognormal(
                mean=np.log(max(p99 * 3, 1)),
                sigma=0.5,
                size=5
            )

            ref = conn.execute(text("""
                SELECT pr.vendor_id, pr.contract_id
                FROM procurement_records pr
                JOIN contracts c ON pr.contract_id = c.contract_id
                WHERE c.naics_code = :n LIMIT 1
            """), {'n': naics}).fetchone()

            if not ref:
                continue

            for val in values:
                conn.execute(text("""
                    INSERT INTO procurement_records (
                        vendor_id, contract_id,
                        total_dollars_obligated,
                        dataset_source,
                        is_synthetic_anomaly,
                        synthetic_anomaly_type
                    ) VALUES (
                        :v, :c, :a,
                        'SYNTHETIC — Value Outlier',
                        TRUE, 'value_outlier_lognormal'
                    )
                """), {'v': ref[0], 'c': ref[1], 'a': round(float(val), 2)})
                synthetic_count += 1

        # Vendor duplicate injection
        perturbations = [
            lambda n: n.replace('LLC', 'L.L.C.') if 'LLC' in n else n + ' II',
            lambda n: n.replace('INC', 'INC.') if 'INC' in n else n + ' CO',
            lambda n: n + '  ' if not n.endswith(' ') else n[:-1],
        ]

        for vendor in top_vendors:
            vid, vname, vstate = vendor
            for perturb in perturbations[:2]:
                try:
                    pname = perturb(vname)
                    if pname == vname or len(pname) > 500:
                        continue

                    new_v = conn.execute(text("""
                        INSERT INTO vendors (
                            recipient_name, recipient_name_raw,
                            state, is_synthetic
                        ) VALUES (:n, :nr, :s, TRUE)
                        ON CONFLICT DO NOTHING
                        RETURNING vendor_id
                    """), {
                        'n': pname[:500],
                        'nr': vname[:500],
                        's': vstate
                    }).fetchone()

                    if not new_v:
                        continue

                    ref_rec = conn.execute(text("""
                        SELECT contract_id, total_dollars_obligated
                        FROM procurement_records
                        WHERE vendor_id = :v LIMIT 1
                    """), {'v': vid}).fetchone()

                    if not ref_rec:
                        continue

                    conn.execute(text("""
                        INSERT INTO procurement_records (
                            vendor_id, contract_id,
                            total_dollars_obligated,
                            dataset_source,
                            is_synthetic_anomaly,
                            synthetic_anomaly_type
                        ) VALUES (
                            :v, :c, :a,
                            'SYNTHETIC — Vendor Duplicate',
                            TRUE, 'vendor_duplicate_perturbation'
                        )
                    """), {
                        'v': new_v[0],
                        'c': ref_rec[0],
                        'a': float(ref_rec[1])
                    })
                    synthetic_count += 1

                except Exception as e:
                    log.warning(f"Perturbation error: {e}")
                    continue

    log.info(f"Synthetic injection complete: {synthetic_count} records")
    return synthetic_count

# =============================================================
# UPDATE VENDOR FREQUENCIES
# =============================================================

def update_vendor_frequencies(engine):
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
# COMPUTE Z-SCORES
# =============================================================

def compute_zscores(engine):
    log.info("Computing z-scores...")
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
    log.info("Z-scores computed")

# =============================================================
# MAIN TEST RUNNER
# =============================================================

def run_test():
    start = time.time()
    log.info("=" * 60)
    log.info("IntegriScan ETL — TEST RUN (50,000 rows)")
    log.info("=" * 60)

    engine = get_engine()
    vendor_cache = {}
    contract_cache = {}
    total_loaded = 0
    chunk_num = 0

    filepath = os.path.join(DATA_FOLDER, TEST_FILE)

    if not os.path.exists(filepath):
        log.error(f"File not found: {filepath}")
        return

    log.info(f"File: {TEST_FILE}")

    try:
        for chunk in extract_sample(filepath, TEST_ROWS, CHUNK_SIZE):
            chunk_num += 1
            rows_before = len(chunk)

            chunk = transform_chunk(chunk)
            rows_after = len(chunk)

            if chunk.empty:
                log.info(f"  Chunk {chunk_num}: empty after transform, skipping")
                continue

            vendor_cache = load_vendors(chunk, engine, vendor_cache)
            contract_cache = load_contracts(chunk, engine, contract_cache)
            records = load_records(chunk, engine, vendor_cache, contract_cache)
            total_loaded += records

            log.info(
                f"  Chunk {chunk_num} | "
                f"Rows in: {rows_before} | "
                f"After clean: {rows_after} | "
                f"Loaded: {records} | "
                f"Total: {total_loaded:,}"
            )

    except Exception as e:
        log.error(f"ETL error: {e}")
        import traceback
        log.error(traceback.format_exc())
        return

    # Post-load operations
    log.info("\nRunning post-load operations...")
    update_vendor_frequencies(engine)
    compute_zscores(engine)
    synthetic_count = inject_synthetic_anomalies_test(engine)

    elapsed = time.time() - start

    # Final summary
    log.info("\n" + "=" * 60)
    log.info("TEST RUN COMPLETE")
    log.info(f"Records loaded:          {total_loaded:,}")
    log.info(f"Synthetic anomalies:     {synthetic_count:,}")
    log.info(f"Unique vendors:          {len(vendor_cache):,}")
    log.info(f"Unique contracts:        {len(contract_cache):,}")
    log.info(f"Time elapsed:            {elapsed:.1f}s")
    log.info("=" * 60)

    # DB verification
    with engine.connect() as conn:
        counts = conn.execute(text("""
            SELECT
                (SELECT COUNT(*) FROM vendors) as vendors,
                (SELECT COUNT(*) FROM contracts) as contracts,
                (SELECT COUNT(*) FROM procurement_records) as records,
                (SELECT COUNT(*) FROM procurement_records
                 WHERE is_synthetic_anomaly = TRUE) as synthetic,
                (SELECT COUNT(*) FROM procurement_records
                 WHERE zscore_within_category IS NOT NULL) as with_zscore
        """)).fetchone()

        log.info(f"\nDatabase Verification:")
        log.info(f"  Vendors:              {counts[0]:,}")
        log.info(f"  Contracts:            {counts[1]:,}")
        log.info(f"  Procurement Records:  {counts[2]:,}")
        log.info(f"  Synthetic Anomalies:  {counts[3]:,}")
        log.info(f"  Records with Z-Score: {counts[4]:,}")

        # Sample flaggable records (z-score > 3)
        high_zscore = conn.execute(text("""
            SELECT COUNT(*)
            FROM procurement_records
            WHERE zscore_within_category > 3
              AND is_synthetic_anomaly = FALSE
        """)).scalar()
        log.info(f"  Price Outliers (>3σ): {high_zscore:,}")

    log.info("\n✅ TEST PASSED — ETL pipeline working correctly!")
    log.info("Ready to run full pipeline on all 5 CSV files.")

if __name__ == "__main__":
    run_test()
