# =============================================================
# IntegriScan — ML Detection Module
# Government Procurement Anomaly Detection System
# MSIT 5910 Capstone | University of the People
# Student: Edrissa Sarr
# =============================================================
# Implements:
#   1. Z-Score baseline anomaly detection
#   2. Isolation Forest anomaly detection
#   3. Synthetic label evaluation (precision/recall)
#   4. Actionability rate metric (target: 85%)
#   5. Explainable risk factor generation [Peer 3]
#   6. Results stored in anomaly_flags table
# =============================================================
# Peer feedback incorporated:
#   [Peer 1] Both IF + z-score stored independently
#   [Peer 1] contamination=0.05 starting point
#   [Peer 2] Actionability rate metric, target 85%
#   [Peer 3] Explainable risk factors per flag
#   [Peer 4] Contamination tuned in 0.01 increments
#   [Peer 4] Log-normal injection already done in ETL
# =============================================================

import os
import time
import logging
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

# =============================================================
# CONFIGURATION
# =============================================================

load_dotenv()

# Isolation Forest parameters [Peer 1 & 4]
CONTAMINATION_START = 0.05   # Starting contamination threshold
CONTAMINATION_STEP  = 0.01   # Tuning increment [Peer 4]
CONTAMINATION_MIN   = 0.01   # Minimum contamination
CONTAMINATION_MAX   = 0.15   # Maximum contamination
RANDOM_STATE        = 42     # Fixed seed for reproducibility

# Z-Score threshold
ZSCORE_THRESHOLD = 3.0       # Flag contracts > 3 std devs from mean

# Actionability rate target [Peer 2]
ACTIONABILITY_TARGET = 0.85  # 85% per Herreros-Martinez et al. (2025)

# Model version
MODEL_VERSION = 'v1.0'

# =============================================================
# LOGGING
# =============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('ml_module.log'),
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
    return create_engine(connection_url, pool_pre_ping=True)

# =============================================================
# STEP 1: LOAD DATA FROM DATABASE
# =============================================================

def load_data(engine):
    """
    Load procurement records from database for ML processing.
    Returns two DataFrames:
      - df_real: real records for training and detection
      - df_all: all records including synthetic for evaluation
    """
    log.info("Loading data from database...")

    query = """
        SELECT
            pr.record_id,
            pr.total_dollars_obligated,
            pr.zscore_within_category,
            pr.category_mean,
            pr.category_std,
            pr.is_synthetic_anomaly,
            pr.synthetic_anomaly_type,
            v.recipient_name,
            v.vendor_frequency,
            v.state,
            c.contract_id,
            c.awarding_agency,
            c.naics_code,
            c.naics_description,
            c.award_day_of_year,
            c.award_month,
            c.fiscal_year,
            c.period_start,
            c.period_end
        FROM procurement_records pr
        JOIN vendors v ON pr.vendor_id = v.vendor_id
        JOIN contracts c ON pr.contract_id = c.contract_id
        WHERE pr.total_dollars_obligated IS NOT NULL
          AND pr.total_dollars_obligated > 0
        ORDER BY pr.record_id
    """

    df_all = pd.read_sql(query, engine)

    # Split real vs synthetic
    df_real = df_all[df_all['is_synthetic_anomaly'] == False].copy()
    df_synthetic = df_all[df_all['is_synthetic_anomaly'] == True].copy()

    log.info(f"Total records loaded:    {len(df_all):,}")
    log.info(f"Real records:            {len(df_real):,}")
    log.info(f"Synthetic anomalies:     {len(df_synthetic):,}")

    return df_real, df_all

# =============================================================
# STEP 2: FEATURE ENGINEERING
# =============================================================

def engineer_features(df):
    """
    Prepare ML features from procurement records.
    Features selected for anomaly detection:
      - total_dollars_obligated: primary amount (log-scaled)
      - zscore_within_category: pre-computed z-score
      - vendor_frequency: number of contracts per vendor
      - award_day_of_year: fiscal year-end detection
      - award_month: seasonal patterns
    """
    log.info("Engineering ML features...")

    df = df.copy()

    # Log-scale the amount to reduce skewness
    df['log_amount'] = np.log1p(df['total_dollars_obligated'])

    # Fill missing values
    df['zscore_within_category'] = df['zscore_within_category'].fillna(0)
    df['vendor_frequency'] = df['vendor_frequency'].fillna(1)
    df['award_day_of_year'] = df['award_day_of_year'].fillna(
        df['award_day_of_year'].median()
    )
    df['award_month'] = df['award_month'].fillna(
        df['award_month'].median()
    )

    # Fiscal year-end flag (last 30 days = days 243-273)
    df['is_yearend'] = df['award_day_of_year'].between(
        243, 273
    ).astype(int)

    # Log vendor frequency
    df['log_vendor_freq'] = np.log1p(df['vendor_frequency'])

    feature_cols = [
        'log_amount',
        'zscore_within_category',
        'log_vendor_freq',
        'award_day_of_year',
        'is_yearend',
    ]

    # Select and return feature matrix
    X = df[feature_cols].copy()

    log.info(f"Features engineered: {feature_cols}")
    log.info(f"Feature matrix shape: {X.shape}")

    return X, feature_cols

# =============================================================
# STEP 3: Z-SCORE BASELINE DETECTION
# =============================================================

def zscore_detection(df_real, df_all):
    """
    Z-Score baseline anomaly detection.
    Flags contracts where zscore_within_category > ZSCORE_THRESHOLD.
    Simple, interpretable baseline for comparison with Isolation Forest.
    """
    log.info(f"Running Z-Score detection (threshold: >{ZSCORE_THRESHOLD} std devs)...")

    # Flag real records
    df_real = df_real.copy()
    df_real['zscore_flagged'] = (
        df_real['zscore_within_category'].abs() > ZSCORE_THRESHOLD
    )

    # Flag all records (including synthetic)
    df_all = df_all.copy()
    df_all['zscore_flagged'] = (
        df_all['zscore_within_category'].abs() > ZSCORE_THRESHOLD
    )

    real_flags = df_real['zscore_flagged'].sum()
    synth_detected = df_all[
        df_all['is_synthetic_anomaly'] == True
    ]['zscore_flagged'].sum()
    total_synthetic = df_all['is_synthetic_anomaly'].sum()

    log.info(f"Z-Score results:")
    log.info(f"  Real price outliers flagged: {real_flags:,} "
             f"({100*real_flags/len(df_real):.1f}%)")
    log.info(f"  Synthetic anomalies detected: {synth_detected}/{total_synthetic}")

    return df_real, df_all

# =============================================================
# STEP 4: ISOLATION FOREST DETECTION
# =============================================================

def isolation_forest_detection(df_real, df_all, contamination=CONTAMINATION_START):
    """
    Isolation Forest anomaly detection.

    Algorithm: Trains on REAL records only (unsupervised).
    Scores ALL records including synthetic.

    Parameters:
      contamination: proportion of expected anomalies [Peer 1 & 4]
      Starting at 0.05, tuned in 0.01 increments [Peer 4]

    Returns:
      df_all with Isolation Forest scores and flags added
    """
    log.info(f"Training Isolation Forest (contamination={contamination})...")

    # Engineer features
    X_real, feature_cols = engineer_features(df_real)
    X_all, _ = engineer_features(df_all)

    # Scale features
    scaler = StandardScaler()
    X_real_scaled = scaler.fit_transform(X_real)
    X_all_scaled = scaler.transform(X_all)

    # Train Isolation Forest on real records only
    iforest = IsolationForest(
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_estimators=100,
        max_samples='auto',
        n_jobs=-1  # Use all CPU cores
    )
    iforest.fit(X_real_scaled)

    # Score all records (real + synthetic)
    # score_samples returns negative scores — more negative = more anomalous
    if_scores = iforest.score_samples(X_all_scaled)
    if_predictions = iforest.predict(X_all_scaled)  # -1 = anomaly, 1 = normal

    # Add scores to DataFrame
    df_all = df_all.copy()
    df_all['if_score'] = if_scores
    df_all['if_flagged'] = (if_predictions == -1)

    real_flags = df_all[
        (df_all['is_synthetic_anomaly'] == False) &
        (df_all['if_flagged'] == True)
    ].shape[0]

    synth_detected = df_all[
        (df_all['is_synthetic_anomaly'] == True) &
        (df_all['if_flagged'] == True)
    ].shape[0]
    total_synthetic = df_all['is_synthetic_anomaly'].sum()

    log.info(f"Isolation Forest results (contamination={contamination}):")
    log.info(f"  Real records flagged:         {real_flags:,} "
             f"({100*real_flags/len(df_all[df_all['is_synthetic_anomaly']==False]):.1f}%)")
    log.info(f"  Synthetic anomalies detected: {synth_detected}/{total_synthetic}")

    return df_all, iforest, scaler, feature_cols

# =============================================================
# STEP 5: EVALUATE WITH SYNTHETIC LABELS
# =============================================================

def evaluate_model(df_all):
    """
    Evaluate model performance using synthetic anomaly labels.

    Since we have no confirmed fraud labels, we use synthetic
    anomalies injected during ETL as ground truth for evaluation.

    Metrics:
      - Precision (IF): of all IF-flagged records, % that are synthetic
      - Recall (IF): of all synthetic records, % detected by IF
      - F1 Score: harmonic mean of precision and recall
      - Z-Score precision and recall
      - Actionability rate (target: 85%) [Peer 2]
    """
    log.info("Evaluating model performance with synthetic labels...")

    # Ground truth: synthetic = True anomaly
    y_true = df_all['is_synthetic_anomaly'].astype(int)

    # Isolation Forest predictions
    y_pred_if = df_all['if_flagged'].astype(int)

    # Z-Score predictions
    y_pred_z = df_all['zscore_flagged'].astype(int)

    # Calculate metrics
    metrics = {}

    # Isolation Forest metrics
    if y_pred_if.sum() > 0:
        metrics['if_precision'] = precision_score(y_true, y_pred_if,
                                                   zero_division=0)
        metrics['if_recall'] = recall_score(y_true, y_pred_if,
                                             zero_division=0)
        metrics['if_f1'] = f1_score(y_true, y_pred_if, zero_division=0)
    else:
        metrics['if_precision'] = 0
        metrics['if_recall'] = 0
        metrics['if_f1'] = 0

    # Z-Score metrics
    if y_pred_z.sum() > 0:
        metrics['z_precision'] = precision_score(y_true, y_pred_z,
                                                  zero_division=0)
        metrics['z_recall'] = recall_score(y_true, y_pred_z,
                                            zero_division=0)
        metrics['z_f1'] = f1_score(y_true, y_pred_z, zero_division=0)
    else:
        metrics['z_precision'] = 0
        metrics['z_recall'] = 0
        metrics['z_f1'] = 0

    # Combined detection (flagged by EITHER method)
    y_pred_combined = ((y_pred_if == 1) | (y_pred_z == 1)).astype(int)
    metrics['combined_precision'] = precision_score(y_true, y_pred_combined,
                                                     zero_division=0)
    metrics['combined_recall'] = recall_score(y_true, y_pred_combined,
                                               zero_division=0)
    metrics['combined_f1'] = f1_score(y_true, y_pred_combined,
                                       zero_division=0)

    # Flag counts
    metrics['total_records'] = len(df_all)
    metrics['total_synthetic'] = int(y_true.sum())
    metrics['if_flags'] = int(y_pred_if.sum())
    metrics['z_flags'] = int(y_pred_z.sum())
    metrics['combined_flags'] = int(y_pred_combined.sum())

    # Actionability rate placeholder [Peer 2]
    # Will be populated as analysts review flags
    # Target: 85% per Herreros-Martinez et al. (2025)
    metrics['actionability_rate'] = None
    metrics['actionability_target'] = ACTIONABILITY_TARGET

    log.info(f"\nModel Evaluation Results:")
    log.info(f"{'='*50}")
    log.info(f"Isolation Forest:")
    log.info(f"  Precision:  {metrics['if_precision']:.3f}")
    log.info(f"  Recall:     {metrics['if_recall']:.3f}")
    log.info(f"  F1 Score:   {metrics['if_f1']:.3f}")
    log.info(f"  Flags:      {metrics['if_flags']:,}")
    log.info(f"Z-Score Baseline:")
    log.info(f"  Precision:  {metrics['z_precision']:.3f}")
    log.info(f"  Recall:     {metrics['z_recall']:.3f}")
    log.info(f"  F1 Score:   {metrics['z_f1']:.3f}")
    log.info(f"  Flags:      {metrics['z_flags']:,}")
    log.info(f"Combined (IF + Z-Score):")
    log.info(f"  Precision:  {metrics['combined_precision']:.3f}")
    log.info(f"  Recall:     {metrics['combined_recall']:.3f}")
    log.info(f"  F1 Score:   {metrics['combined_f1']:.3f}")
    log.info(f"  Flags:      {metrics['combined_flags']:,}")
    log.info(f"Actionability Rate: Pending analyst review "
             f"(target: {ACTIONABILITY_TARGET*100:.0f}%)")
    log.info(f"{'='*50}")

    return metrics

# =============================================================
# STEP 6: GENERATE EXPLAINABLE RISK FACTORS [Peer 3]
# =============================================================

def generate_risk_explanation(row):
    """
    Generate plain-language explanation of why a contract was flagged.
    Implements XAI (Explainable AI) principle from Peer 3 feedback.

    Returns:
      explanation: main explanation string
      factor_1: primary factor
      factor_2: secondary factor (if any)
      factor_3: tertiary factor (if any)
    """
    factors = []

    # Factor 1: Price outlier
    if abs(row.get('zscore_within_category', 0)) > ZSCORE_THRESHOLD:
        z = row['zscore_within_category']
        mean = row.get('category_mean', 0)
        factors.append(
            f"Contract value is {abs(z):.1f} standard deviations "
            f"{'above' if z > 0 else 'below'} the "
            f"{row.get('naics_description', 'category')[:30]} "
            f"category mean "
            f"(mean: ${mean:,.0f})"
        )

    # Factor 2: Fiscal year-end clustering
    doy = row.get('award_day_of_year', 0)
    if doy and 243 <= doy <= 273:
        days_before_ye = 273 - doy
        factors.append(
            f"Contract awarded {days_before_ye} days before "
            f"fiscal year-end (Sep 30) — "
            f"year-end clustering pattern detected"
        )

    # Factor 3: High vendor frequency
    freq = row.get('vendor_frequency', 0)
    if freq and freq > 50:
        factors.append(
            f"Vendor appears in {freq:,} contracts — "
            f"unusually high award frequency"
        )

    # Factor 4: Strong Isolation Forest signal
    if_score = row.get('if_score', 0)
    if if_score and if_score < -0.5:
        factors.append(
            f"Isolation Forest anomaly score: {if_score:.3f} "
            f"(strong anomaly signal)"
        )

    # Factor 5: Synthetic anomaly type
    synth_type = row.get('synthetic_anomaly_type', '')
    if synth_type == 'vendor_duplicate_perturbation':
        factors.append(
            "Vendor name matches another vendor with slight "
            "variations — possible duplicate registration pattern"
        )

    # Build explanation
    if factors:
        explanation = " | ".join(factors[:3])
    else:
        explanation = (
            f"Flagged by Isolation Forest algorithm "
            f"(score: {row.get('if_score', 0):.3f})"
        )

    return (
        explanation[:500],
        factors[0][:500] if len(factors) > 0 else None,
        factors[1][:500] if len(factors) > 1 else None,
        factors[2][:500] if len(factors) > 2 else None,
    )

# =============================================================
# STEP 7: STORE FLAGS IN DATABASE
# =============================================================

def store_flags(df_all, engine, contamination, metrics):
    """
    Store anomaly detection results in anomaly_flags table.

    Stores:
      - Isolation Forest score (independent) [Peer 1]
      - Z-score deviation (independent) [Peer 1]
      - Explainable risk factors [Peer 3]
      - Contamination threshold used [Peer 4]
      - Analyst feedback = 'pending' (default) [Peer 2]
    """
    log.info("Storing anomaly flags in database...")

    # Get records flagged by either method
    flagged = df_all[
        (df_all['if_flagged'] == True) |
        (df_all['zscore_flagged'] == True)
    ].copy()

    log.info(f"Total records to flag: {len(flagged):,}")

    # Clear existing flags for this model version
    with engine.begin() as conn:
        deleted = conn.execute(text(
            "DELETE FROM anomaly_flags WHERE model_version = :v"
        ), {'v': MODEL_VERSION})
        log.info(f"Cleared {deleted.rowcount} existing flags")

    # Insert new flags
    inserted = 0
    with engine.begin() as conn:
        for _, row in flagged.iterrows():
            # Generate explanation [Peer 3]
            explanation, f1, f2, f3 = generate_risk_explanation(row)

            # Determine flag type
            if row.get('synthetic_anomaly_type') == 'vendor_duplicate_perturbation':
                flag_type = 'vendor_duplicate'
            elif abs(row.get('zscore_within_category', 0)) > ZSCORE_THRESHOLD:
                if row.get('award_day_of_year', 0) and \
                   243 <= row['award_day_of_year'] <= 273:
                    flag_type = 'price_outlier_yearend'
                else:
                    flag_type = 'price_outlier'
            elif row.get('award_day_of_year', 0) and \
                 243 <= row['award_day_of_year'] <= 273:
                flag_type = 'yearend_cluster'
            else:
                flag_type = 'isolation_forest'

            conn.execute(text("""
                INSERT INTO anomaly_flags (
                    record_id,
                    isolation_forest_score,
                    isolation_forest_flagged,
                    zscore_deviation,
                    zscore_flagged,
                    flag_type,
                    risk_factor_explanation,
                    risk_factor_1,
                    risk_factor_2,
                    risk_factor_3,
                    contamination_threshold,
                    is_synthetic_anomaly,
                    synthetic_anomaly_type,
                    model_version,
                    analyst_feedback
                ) VALUES (
                    :record_id,
                    :if_score,
                    :if_flagged,
                    :zscore,
                    :zscore_flagged,
                    :flag_type,
                    :explanation,
                    :f1, :f2, :f3,
                    :contamination,
                    :is_synthetic,
                    :synth_type,
                    :model_version,
                    'pending'
                )
            """), {
                'record_id': int(row['record_id']),
                'if_score': float(row['if_score'])
                    if pd.notna(row.get('if_score')) else None,
                'if_flagged': bool(row['if_flagged']),
                'zscore': float(row['zscore_within_category'])
                    if pd.notna(row.get('zscore_within_category')) else None,
                'zscore_flagged': bool(row['zscore_flagged']),
                'flag_type': flag_type,
                'explanation': explanation,
                'f1': f1, 'f2': f2, 'f3': f3,
                'contamination': contamination,
                'is_synthetic': bool(row['is_synthetic_anomaly']),
                'synth_type': row.get('synthetic_anomaly_type'),
                'model_version': MODEL_VERSION,
            })
            inserted += 1

    log.info(f"Flags stored: {inserted:,}")
    return inserted

# =============================================================
# STEP 8: CONTAMINATION TUNING [Peer 4]
# =============================================================

def tune_contamination(df_real, df_all):
    """
    Tune the Isolation Forest contamination parameter iteratively.
    Tests contamination values from CONTAMINATION_MIN to MAX
    in steps of CONTAMINATION_STEP.

    Selects the threshold that maximizes F1 score on synthetic labels
    while keeping flag rate below 10% of total records.

    [Peer 4 feedback: tune in 0.01 increments]
    """
    log.info("Tuning contamination parameter...")

    best_contamination = CONTAMINATION_START
    best_f1 = 0
    results = []

    for c in np.arange(
        CONTAMINATION_MIN,
        CONTAMINATION_MAX + CONTAMINATION_STEP,
        CONTAMINATION_STEP
    ):
        c = round(c, 2)
        df_scored, _, _, _ = isolation_forest_detection(
            df_real, df_all, contamination=c
        )

        # Quick evaluation
        y_true = df_scored['is_synthetic_anomaly'].astype(int)
        y_pred = df_scored['if_flagged'].astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        flags = y_pred.sum()
        flag_rate = flags / len(df_scored)

        results.append({
            'contamination': c,
            'f1': f1,
            'flags': flags,
            'flag_rate': flag_rate
        })

        log.info(f"  c={c:.2f} | F1={f1:.3f} | "
                 f"flags={flags:,} ({flag_rate*100:.1f}%)")

        if f1 > best_f1 and flag_rate < 0.10:
            best_f1 = f1
            best_contamination = c

    log.info(f"Best contamination: {best_contamination} (F1={best_f1:.3f})")
    return best_contamination, results

# =============================================================
# MAIN ML PIPELINE
# =============================================================

def run_ml_pipeline(tune=False):
    """
    Main ML pipeline orchestration.

    Args:
      tune: if True, run contamination tuning before detection
            if False, use CONTAMINATION_START directly (faster)
    """
    start_time = time.time()

    log.info("=" * 60)
    log.info("IntegriScan ML Detection Pipeline Starting")
    log.info(f"Model version: {MODEL_VERSION}")
    log.info("=" * 60)

    # Connect to database
    engine = get_engine()

    # Step 1: Load data
    df_real, df_all = load_data(engine)

    # Step 2: Z-Score detection (baseline)
    df_real, df_all = zscore_detection(df_real, df_all)

    # Step 3: Contamination tuning (optional)
    if tune:
        log.info("\nRunning contamination tuning...")
        best_contamination, tuning_results = tune_contamination(
            df_real, df_all
        )
    else:
        best_contamination = CONTAMINATION_START
        log.info(f"\nUsing starting contamination: {best_contamination}")

    # Step 4: Isolation Forest detection
    df_all, iforest, scaler, feature_cols = isolation_forest_detection(
        df_real, df_all, contamination=best_contamination
    )

    # Merge z-score flags into df_all
    df_all['zscore_flagged'] = df_all['zscore_within_category'].abs() > ZSCORE_THRESHOLD

    # Step 5: Evaluate
    metrics = evaluate_model(df_all)

    # Step 6: Store flags
    flags_stored = store_flags(df_all, engine, best_contamination, metrics)

    # Step 7: Final summary
    elapsed = time.time() - start_time

    log.info("\n" + "=" * 60)
    log.info("ML PIPELINE COMPLETE")
    log.info(f"Contamination used:      {best_contamination}")
    log.info(f"Total flags stored:      {flags_stored:,}")
    log.info(f"IF Precision:            {metrics['if_precision']:.3f}")
    log.info(f"IF Recall:               {metrics['if_recall']:.3f}")
    log.info(f"IF F1 Score:             {metrics['if_f1']:.3f}")
    log.info(f"Z-Score Precision:       {metrics['z_precision']:.3f}")
    log.info(f"Z-Score Recall:          {metrics['z_recall']:.3f}")
    log.info(f"Z-Score F1:              {metrics['z_f1']:.3f}")
    log.info(f"Actionability Target:    {ACTIONABILITY_TARGET*100:.0f}%")
    log.info(f"Time elapsed:            {elapsed:.1f}s")
    log.info("=" * 60)

    # Verify flags in database
    with engine.connect() as conn:
        flag_counts = conn.execute(text("""
            SELECT
                flag_type,
                COUNT(*) as count,
                AVG(isolation_forest_score) as avg_if_score,
                AVG(zscore_deviation) as avg_zscore
            FROM anomaly_flags
            WHERE model_version = :v
            GROUP BY flag_type
            ORDER BY count DESC
        """), {'v': MODEL_VERSION}).fetchall()

        log.info("\nFlags by Type:")
        for row in flag_counts:
            avg_if = round(float(row[2]), 3) if row[2] is not None else 0.0
            avg_z = round(float(row[3]), 2) if row[3] is not None else 0.0
            log.info(
                f"  {row[0]:<30} count={row[1]:,} "
                f"avg_IF={avg_if:.3f} "
                f"avg_Z={avg_z:.2f}"
            )

        total_flags = conn.execute(text(
            "SELECT COUNT(*) FROM anomaly_flags WHERE model_version = :v"
        ), {'v': MODEL_VERSION}).scalar()

        actionability_rate = conn.execute(text("""
            SELECT
                ROUND(100.0 * COUNT(CASE WHEN analyst_feedback = 'actionable' THEN 1 END)
                / NULLIF(COUNT(CASE WHEN analyst_feedback != 'pending' THEN 1 END), 0), 1)
            FROM anomaly_flags
            WHERE model_version = :v
        """), {'v': MODEL_VERSION}).scalar()

        log.info(f"\nDatabase Verification:")
        log.info(f"  Total flags in DB:    {total_flags:,}")
        log.info(f"  Actionability rate:   "
                 f"{actionability_rate if actionability_rate else 'Pending analyst review'}")
        log.info(f"  Target:               {ACTIONABILITY_TARGET*100:.0f}%")

    log.info("\nML Pipeline complete. Ready for dashboard integration (Unit 6).")

    return metrics, flags_stored

# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":
    # Run without tuning for faster execution
    # Set tune=True to run full contamination tuning
    metrics, flags = run_ml_pipeline(tune=False)
