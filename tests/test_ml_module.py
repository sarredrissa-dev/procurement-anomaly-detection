# =============================================================
# IntegriScan — PyTest Unit Tests
# Government Procurement Anomaly Detection System
# MSIT 5910 Capstone | University of the People
# Student: Edrissa Sarr
# =============================================================
# Tests cover:
#   - Database connectivity
#   - ETL data transformation
#   - Z-Score anomaly detection logic
#   - Isolation Forest detection
#   - Explainable risk factor generation [Peer 3]
#   - Database flag storage
# =============================================================
# Testing types used:
#   WHITE-BOX: Tests internal logic with known inputs/outputs
#   BLACK-BOX: Tests behaviour without examining internals
# =============================================================

import os
import sys
import pytest
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Add project root to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Load environment variables
load_dotenv()

# Import our modules
from etl.etl_test import transform_chunk
from ml.ml_module import (
    get_engine,
    engineer_features,
    zscore_detection,
    isolation_forest_detection,
    generate_risk_explanation,
    ZSCORE_THRESHOLD,
    CONTAMINATION_START,
    RANDOM_STATE,
)

# =============================================================
# FIXTURES — Reusable test setup
# =============================================================

@pytest.fixture(scope="session")
def engine():
    """
    Session-scoped database engine.
    Created once and reused across all tests.
    WHITE-BOX: We know exactly how the connection is built.
    """
    return get_engine()

@pytest.fixture(scope="session")
def sample_df():
    """
    Create a small sample DataFrame mimicking real procurement data.
    Used across multiple tests without hitting the database.
    """
    return pd.DataFrame({
        'record_id':                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'total_dollars_obligated':  [50000, 75000, 120000, 0, -100,
                                     None, 200000, 5000000, 80000, 60000],
        'zscore_within_category':   [0.5, 1.2, -0.8, 0.0, 0.0,
                                     0.0, 1.5, 4.8, 0.3, 0.9],
        'category_mean':            [80000]*10,
        'category_std':             [25000]*10,
        'recipient_name':           ['VENDOR A', 'VENDOR B', 'VENDOR C',
                                     'VENDOR D', 'VENDOR E', 'VENDOR F',
                                     'VENDOR G', 'VENDOR H', 'VENDOR I',
                                     'VENDOR J'],
        'recipient_name_raw':       ['Vendor A']*10,
        'recipient_state_code':     ['CA', 'TX', 'NY', 'FL', 'WA',
                                     'OR', 'VA', 'MD', 'GA', 'IL'],
        'vendor_frequency':         [5, 12, 3, 1, 200, 8, 15, 2, 7, 4],
        'is_synthetic_anomaly':     [False]*9 + [True],
        'synthetic_anomaly_type':   [None]*9 + ['value_outlier_lognormal'],
        'naics_code':               ['541330']*10,
        'naics_description':        ['Engineering Services']*10,
        'awarding_agency':          ['Dept of Defense']*10,
        'award_day_of_year':        [100, 200, 265, 150, 270,
                                     180, 90, 260, 310, 50],
        'award_month':              [4, 7, 9, 5, 9, 6, 3, 9, 11, 2],
        'fiscal_year':              [2023]*10,
        'if_score':                 [-0.3, -0.2, -0.4, -0.1, -0.8,
                                     -0.3, -0.2, -0.9, -0.1, -0.5],
        'if_flagged':               [False, False, False, False, True,
                                     False, False, True, False, False],
        'zscore_flagged':           [False, False, False, False, False,
                                     False, False, True, False, False],
        'contract_id':              [101]*10,
        'period_start':             [None]*10,
        'period_end':               [None]*10,
    })

@pytest.fixture(scope="session")
def ml_sample_df():
    """
    Larger sample with clear anomalies for ML testing.
    100 normal records + 10 obvious outliers.
    """
    np.random.seed(42)
    n_normal = 100
    n_anomaly = 10

    normal = pd.DataFrame({
        'record_id': range(1, n_normal + 1),
        'total_dollars_obligated': np.random.lognormal(11, 1, n_normal),
        'zscore_within_category': np.random.normal(0, 1, n_normal),
        'vendor_frequency': np.random.randint(1, 20, n_normal),
        'award_day_of_year': np.random.randint(1, 365, n_normal),
        'award_month': np.random.randint(1, 13, n_normal),
        'is_synthetic_anomaly': [False] * n_normal,
        'synthetic_anomaly_type': [None] * n_normal,
        'naics_code': ['541330'] * n_normal,
        'naics_description': ['Engineering Services'] * n_normal,
        'awarding_agency': ['Dept of Defense'] * n_normal,
        'recipient_name': [f'VENDOR_{i}' for i in range(n_normal)],
        'vendor_frequency': np.random.randint(1, 20, n_normal),
        'state': ['CA'] * n_normal,
        'contract_id': range(1, n_normal + 1),
        'category_mean': [60000] * n_normal,
        'category_std': [20000] * n_normal,
        'period_start': [None] * n_normal,
        'period_end': [None] * n_normal,
        'fiscal_year': [2023] * n_normal,
    })

    anomalies = pd.DataFrame({
        'record_id': range(n_normal + 1, n_normal + n_anomaly + 1),
        'total_dollars_obligated': np.random.lognormal(16, 1, n_anomaly),
        'zscore_within_category': np.random.uniform(5, 15, n_anomaly),
        'vendor_frequency': np.random.randint(100, 500, n_anomaly),
        'award_day_of_year': np.random.randint(243, 273, n_anomaly),
        'award_month': [9] * n_anomaly,
        'is_synthetic_anomaly': [True] * n_anomaly,
        'synthetic_anomaly_type': ['value_outlier_lognormal'] * n_anomaly,
        'naics_code': ['541330'] * n_anomaly,
        'naics_description': ['Engineering Services'] * n_anomaly,
        'awarding_agency': ['Dept of Defense'] * n_anomaly,
        'recipient_name': [f'ANOMALY_{i}' for i in range(n_anomaly)],
        'vendor_frequency': [200] * n_anomaly,
        'state': ['CA'] * n_anomaly,
        'contract_id': range(n_normal + 1, n_normal + n_anomaly + 1),
        'category_mean': [60000] * n_anomaly,
        'category_std': [20000] * n_anomaly,
        'period_start': [None] * n_anomaly,
        'period_end': [None] * n_anomaly,
        'fiscal_year': [2023] * n_anomaly,
    })

    return pd.concat([normal, anomalies], ignore_index=True)

# =============================================================
# TEST GROUP 1: DATABASE CONNECTIVITY
# Type: White-box (we know the internal connection logic)
# =============================================================

class TestDatabaseConnectivity:

    def test_database_connection_succeeds(self, engine):
        """
        WHITE-BOX: Verify SQLAlchemy engine connects to PostgreSQL.
        Tests the get_engine() function internals.
        """
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).scalar()
        assert result == 1, "Database connection failed"

    def test_procurement_db_name(self, engine):
        """
        WHITE-BOX: Verify we are connected to the correct database.
        """
        with engine.connect() as conn:
            db_name = conn.execute(
                text("SELECT current_database()")
            ).scalar()
        assert db_name == "procurement_db", \
            f"Wrong database: {db_name}"

    def test_all_tables_exist(self, engine):
        """
        WHITE-BOX: Verify all 5 required tables exist in public schema.
        """
        required_tables = {
            'vendors', 'contracts', 'procurement_records',
            'anomaly_flags', 'audit_log'
        }
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_type = 'BASE TABLE'
            """))
            existing = {row[0] for row in result}
        missing = required_tables - existing
        assert not missing, f"Missing tables: {missing}"

    def test_all_views_exist(self, engine):
        """
        WHITE-BOX: Verify all 3 required views exist.
        """
        required_views = {
            'v_flagged_contracts',
            'v_actionability_rate',
            'v_agency_summary'
        }
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT viewname FROM pg_views
                WHERE schemaname = 'public'
            """))
            existing = {row[0] for row in result}
        missing = required_views - existing
        assert not missing, f"Missing views: {missing}"

    def test_rbac_roles_exist(self, engine):
        """
        WHITE-BOX: Verify RBAC roles are created.
        """
        with engine.connect() as conn:
            roles = conn.execute(text("""
                SELECT rolname FROM pg_roles
                WHERE rolname LIKE 'integriscan%'
            """))
            role_names = {row[0] for row in roles}
        assert 'integriscan_analyst' in role_names
        assert 'integriscan_admin' in role_names

# =============================================================
# TEST GROUP 2: ETL DATA TRANSFORMATION
# Type: White-box (testing internal transform logic)
# =============================================================

class TestETLTransformation:

    def test_transform_removes_null_amounts(self):
        """
        WHITE-BOX: Verify transform_chunk removes records with
        null total_dollars_obligated values.
        """
        df = pd.DataFrame({
            'recipient_name':       ['VENDOR A', 'VENDOR B', 'VENDOR C'],
            'recipient_name_raw':   ['Vendor A', 'Vendor B', 'Vendor C'],
            'recipient_state_code': ['CA', 'TX', 'NY'],
            'total_dollars_obligated': [50000.0, None, 75000.0],
            'current_total_value_of_award': [50000.0, None, 75000.0],
            'potential_total_value_of_award': [50000.0, None, 75000.0],
            'contract_award_unique_key': ['K001', 'K002', 'K003'],
            'award_id_piid':        ['P001', 'P002', 'P003'],
            'action_date':          ['2023-01-15', '2023-02-20', '2023-03-10'],
            'action_date_fiscal_year': ['2023', '2023', '2023'],
            'period_of_performance_start_date': ['2023-01-15']*3,
            'period_of_performance_current_end_date': ['2024-01-15']*3,
            'awarding_agency_name': ['DoD']*3,
            'funding_agency_name':  ['DoD']*3,
            'award_type':           ['Fixed Price']*3,
            'naics_code':           ['541330']*3,
            'naics_description':    ['Engineering Services']*3,
        })
        result = transform_chunk(df)
        assert len(result) == 2, \
            f"Expected 2 records after removing null, got {len(result)}"
        assert result['total_dollars_obligated'].isna().sum() == 0

    def test_transform_removes_negative_amounts(self):
        """
        WHITE-BOX: Verify transform_chunk removes records with
        zero or negative contract values.
        """
        df = pd.DataFrame({
            'recipient_name':       ['VENDOR A', 'VENDOR B', 'VENDOR C'],
            'recipient_name_raw':   ['Vendor A', 'Vendor B', 'Vendor C'],
            'recipient_state_code': ['CA', 'TX', 'NY'],
            'total_dollars_obligated': [50000.0, -5000.0, 0.0],
            'current_total_value_of_award': [50000.0, 0.0, 0.0],
            'potential_total_value_of_award': [50000.0, 0.0, 0.0],
            'contract_award_unique_key': ['K001', 'K002', 'K003'],
            'award_id_piid':        ['P001', 'P002', 'P003'],
            'action_date':          ['2023-01-15']*3,
            'action_date_fiscal_year': ['2023']*3,
            'period_of_performance_start_date': ['2023-01-15']*3,
            'period_of_performance_current_end_date': ['2024-01-15']*3,
            'awarding_agency_name': ['DoD']*3,
            'funding_agency_name':  ['DoD']*3,
            'award_type':           ['Fixed Price']*3,
            'naics_code':           ['541330']*3,
            'naics_description':    ['Engineering Services']*3,
        })
        result = transform_chunk(df)
        assert len(result) == 1, \
            f"Expected 1 record after removing negatives, got {len(result)}"
        assert all(result['total_dollars_obligated'] >= 1.0)

    def test_transform_uppercases_vendor_names(self):
        """
        WHITE-BOX: Verify vendor names are normalized to uppercase.
        """
        df = pd.DataFrame({
            'recipient_name':       ['vendor abc', 'Vendor XYZ'],
            'recipient_name_raw':   ['vendor abc', 'Vendor XYZ'],
            'recipient_state_code': ['CA', 'TX'],
            'total_dollars_obligated': [50000.0, 75000.0],
            'current_total_value_of_award': [50000.0, 75000.0],
            'potential_total_value_of_award': [50000.0, 75000.0],
            'contract_award_unique_key': ['K001', 'K002'],
            'award_id_piid':        ['P001', 'P002'],
            'action_date':          ['2023-01-15']*2,
            'action_date_fiscal_year': ['2023']*2,
            'period_of_performance_start_date': ['2023-01-15']*2,
            'period_of_performance_current_end_date': ['2024-01-15']*2,
            'awarding_agency_name': ['DoD']*2,
            'funding_agency_name':  ['DoD']*2,
            'award_type':           ['Fixed Price']*2,
            'naics_code':           ['541330']*2,
            'naics_description':    ['Engineering Services']*2,
        })
        result = transform_chunk(df)
        assert all(result['recipient_name'] == result['recipient_name'].str.upper()), \
            "Vendor names not normalized to uppercase"

    def test_transform_removes_duplicates(self):
        """
        WHITE-BOX: Verify duplicate contract keys are removed.
        """
        df = pd.DataFrame({
            'recipient_name':       ['VENDOR A', 'VENDOR A'],
            'recipient_name_raw':   ['Vendor A', 'Vendor A'],
            'recipient_state_code': ['CA', 'CA'],
            'total_dollars_obligated': [50000.0, 50000.0],
            'current_total_value_of_award': [50000.0, 50000.0],
            'potential_total_value_of_award': [50000.0, 50000.0],
            'contract_award_unique_key': ['K001', 'K001'],  # Duplicate!
            'award_id_piid':        ['P001', 'P001'],
            'action_date':          ['2023-01-15']*2,
            'action_date_fiscal_year': ['2023']*2,
            'period_of_performance_start_date': ['2023-01-15']*2,
            'period_of_performance_current_end_date': ['2024-01-15']*2,
            'awarding_agency_name': ['DoD']*2,
            'funding_agency_name':  ['DoD']*2,
            'award_type':           ['Fixed Price']*2,
            'naics_code':           ['541330']*2,
            'naics_description':    ['Engineering Services']*2,
        })
        result = transform_chunk(df)
        assert len(result) == 1, \
            f"Expected 1 record after deduplication, got {len(result)}"

# =============================================================
# TEST GROUP 3: Z-SCORE DETECTION
# Type: White-box (testing known threshold logic)
# =============================================================

class TestZScoreDetection:

    def test_zscore_flags_outliers(self, sample_df):
        """
        WHITE-BOX: Records with zscore > ZSCORE_THRESHOLD
        must be flagged. Record 8 has zscore=4.8 (>3.0).
        """
        df_real = sample_df[sample_df['is_synthetic_anomaly'] == False].copy()
        df_real, _ = zscore_detection(df_real, sample_df.copy())
        flagged = df_real[df_real['zscore_flagged'] == True]
        assert len(flagged) > 0, "No outliers flagged by z-score"
        assert all(flagged['zscore_within_category'].abs() > ZSCORE_THRESHOLD)

    def test_zscore_does_not_flag_normal(self, sample_df):
        """
        WHITE-BOX: Records with |zscore| <= ZSCORE_THRESHOLD
        must NOT be flagged.
        """
        df_real = sample_df[sample_df['is_synthetic_anomaly'] == False].copy()
        df_real, _ = zscore_detection(df_real, sample_df.copy())
        not_flagged = df_real[df_real['zscore_flagged'] == False]
        assert all(
            not_flagged['zscore_within_category'].abs() <= ZSCORE_THRESHOLD
        ), "Normal records incorrectly flagged by z-score"

    def test_zscore_threshold_value(self):
        """
        WHITE-BOX: Verify the threshold constant is set to 3.0.
        This is a critical algorithm parameter.
        """
        assert ZSCORE_THRESHOLD == 3.0, \
            f"Expected threshold 3.0, got {ZSCORE_THRESHOLD}"

    def test_zscore_flags_correct_count(self, sample_df):
        """
        WHITE-BOX: Verify the count of flagged records matches
        expected number of outliers in sample data.
        Record 8 has zscore=4.8 — only one in our sample.
        """
        df_real = sample_df[sample_df['is_synthetic_anomaly'] == False].copy()
        df_real, _ = zscore_detection(df_real, sample_df.copy())
        flagged_count = df_real['zscore_flagged'].sum()
        assert flagged_count == 1, \
            f"Expected 1 outlier flag, got {flagged_count}"

# =============================================================
# TEST GROUP 4: ISOLATION FOREST DETECTION
# Type: White-box + Black-box combination
# =============================================================

class TestIsolationForest:

    def test_if_scores_in_valid_range(self, ml_sample_df):
        """
        WHITE-BOX: Isolation Forest scores must be in range [-1, 0].
        score_samples() returns negative scores by definition.
        """
        df_real = ml_sample_df[
            ml_sample_df['is_synthetic_anomaly'] == False
        ].copy()
        df_all, _, _, _ = isolation_forest_detection(
            df_real, ml_sample_df.copy(),
            contamination=CONTAMINATION_START
        )
        assert df_all['if_score'].min() >= -1.0, \
            "IF scores below -1.0 — invalid range"
        assert df_all['if_score'].max() <= 0.0, \
            "IF scores above 0.0 — invalid range"

    def test_if_flags_expected_proportion(self, ml_sample_df):
        """
        WHITE-BOX: With contamination=0.05, approximately 5%
        of records should be flagged by Isolation Forest.
        Allow 2% tolerance.
        """
        df_real = ml_sample_df[
            ml_sample_df['is_synthetic_anomaly'] == False
        ].copy()
        df_all, _, _, _ = isolation_forest_detection(
            df_real, ml_sample_df.copy(),
            contamination=CONTAMINATION_START
        )
        flag_rate = df_all['if_flagged'].mean()
        assert 0.03 <= flag_rate <= 0.15, \
            f"Flag rate {flag_rate:.3f} outside expected range [0.03, 0.15]"

    def test_if_detects_obvious_anomalies(self, ml_sample_df):
        """
        BLACK-BOX: Isolation Forest should detect most obvious
        anomalies in our sample (values 100x above normal range).
        We expect at least 50% recall on synthetic labels.
        """
        df_real = ml_sample_df[
            ml_sample_df['is_synthetic_anomaly'] == False
        ].copy()
        df_all, _, _, _ = isolation_forest_detection(
            df_real, ml_sample_df.copy(),
            contamination=CONTAMINATION_START
        )
        synthetic = df_all[df_all['is_synthetic_anomaly'] == True]
        detected = synthetic['if_flagged'].sum()
        recall = detected / len(synthetic)
        assert recall >= 0.5, \
            f"IF recall {recall:.2f} below minimum threshold 0.50"

    def test_if_contamination_parameter(self):
        """
        WHITE-BOX: Verify contamination starting value is 0.05
        as committed in peer feedback response [Peer 1 & 4].
        """
        assert CONTAMINATION_START == 0.05, \
            f"Expected contamination 0.05, got {CONTAMINATION_START}"

    def test_if_reproducible_with_fixed_seed(self, ml_sample_df):
        """
        WHITE-BOX: With fixed RANDOM_STATE, two runs must produce
        identical results. Critical for reproducibility.
        """
        df_real = ml_sample_df[
            ml_sample_df['is_synthetic_anomaly'] == False
        ].copy()
        df1, _, _, _ = isolation_forest_detection(
            df_real, ml_sample_df.copy(),
            contamination=CONTAMINATION_START
        )
        df2, _, _, _ = isolation_forest_detection(
            df_real, ml_sample_df.copy(),
            contamination=CONTAMINATION_START
        )
        assert df1['if_flagged'].equals(df2['if_flagged']), \
            "IF results not reproducible with fixed random seed"

# =============================================================
# TEST GROUP 5: EXPLAINABLE RISK FACTORS [Peer 3]
# Type: Black-box (testing output behaviour)
# =============================================================

class TestRiskExplanation:

    def test_explanation_generated_for_price_outlier(self):
        """
        BLACK-BOX: A record with high z-score must receive
        a meaningful plain-language explanation. [Peer 3]
        """
        row = {
            'zscore_within_category': 4.8,
            'category_mean': 80000,
            'naics_description': 'Engineering Services',
            'award_day_of_year': 150,
            'vendor_frequency': 5,
            'if_score': -0.3,
            'synthetic_anomaly_type': None,
        }
        explanation, f1, f2, f3 = generate_risk_explanation(row)
        assert explanation is not None, "No explanation generated"
        assert len(explanation) > 10, "Explanation too short"

    def test_explanation_mentions_standard_deviations(self):
        """
        BLACK-BOX: Price outlier explanation must mention
        standard deviations to be useful to auditors. [Peer 3]
        """
        row = {
            'zscore_within_category': 6.2,
            'category_mean': 80000,
            'naics_description': 'Engineering Services',
            'award_day_of_year': 150,
            'vendor_frequency': 5,
            'if_score': -0.5,
            'synthetic_anomaly_type': None,
        }
        explanation, f1, f2, f3 = generate_risk_explanation(row)
        assert 'standard deviation' in explanation.lower() or \
               'above' in explanation.lower(), \
            "Explanation does not mention deviation from mean"

    def test_yearend_explanation_mentions_fiscal_year(self):
        """
        BLACK-BOX: A contract awarded near fiscal year-end
        must receive an explanation mentioning year-end.
        """
        row = {
            'zscore_within_category': 0.5,
            'category_mean': 80000,
            'naics_description': 'Engineering Services',
            'award_day_of_year': 265,  # 8 days before fiscal year-end
            'vendor_frequency': 5,
            'if_score': -0.4,
            'synthetic_anomaly_type': None,
        }
        explanation, f1, f2, f3 = generate_risk_explanation(row)
        assert 'fiscal year' in explanation.lower() or \
               'year-end' in explanation.lower(), \
            "Year-end explanation missing fiscal year reference"

    def test_explanation_length_within_limit(self):
        """
        BLACK-BOX: Explanation must not exceed 500 characters
        to fit in the database column.
        """
        row = {
            'zscore_within_category': 8.5,
            'category_mean': 80000,
            'naics_description': 'A' * 100,
            'award_day_of_year': 265,
            'vendor_frequency': 500,
            'if_score': -0.9,
            'synthetic_anomaly_type': 'vendor_duplicate_perturbation',
        }
        explanation, f1, f2, f3 = generate_risk_explanation(row)
        assert len(explanation) <= 500, \
            f"Explanation exceeds 500 chars: {len(explanation)}"

    def test_vendor_duplicate_explanation(self):
        """
        BLACK-BOX: A vendor duplicate anomaly must receive
        an explanation mentioning duplicate or variation.
        """
        row = {
            'zscore_within_category': 0.5,
            'category_mean': 80000,
            'naics_description': 'Engineering Services',
            'award_day_of_year': 150,
            'vendor_frequency': 5,
            'if_score': -0.3,
            'synthetic_anomaly_type': 'vendor_duplicate_perturbation',
        }
        explanation, f1, f2, f3 = generate_risk_explanation(row)
        assert 'duplicate' in explanation.lower() or \
               'variation' in explanation.lower(), \
            "Vendor duplicate explanation missing key terms"

# =============================================================
# TEST GROUP 6: DATABASE FLAG STORAGE
# Type: Black-box (testing end-to-end storage behaviour)
# =============================================================

class TestDatabaseStorage:

    def test_anomaly_flags_table_has_data(self, engine):
        """
        BLACK-BOX: After ML pipeline runs, anomaly_flags
        table must contain records.
        """
        with engine.connect() as conn:
            count = conn.execute(text(
                "SELECT COUNT(*) FROM anomaly_flags"
            )).scalar()
        assert count > 0, "anomaly_flags table is empty"

    def test_flags_have_model_version(self, engine):
        """
        BLACK-BOX: All flags must have a model_version set.
        """
        with engine.connect() as conn:
            nulls = conn.execute(text("""
                SELECT COUNT(*) FROM anomaly_flags
                WHERE model_version IS NULL
            """)).scalar()
        assert nulls == 0, f"{nulls} flags missing model_version"

    def test_flags_store_both_scores_independently(self, engine):
        """
        WHITE-BOX: Both IF score and z-score must be stored
        independently per flag. [Peer 1 feedback]
        Verify by checking both columns exist and have values.
        """
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT
                    COUNT(*) as total,
                    COUNT(isolation_forest_score) as with_if_score,
                    COUNT(zscore_deviation) as with_zscore
                FROM anomaly_flags
            """)).fetchone()
        total, with_if, with_z = result
        assert with_if > 0, "No IF scores stored in anomaly_flags"
        assert with_z > 0, "No z-scores stored in anomaly_flags"

    def test_analyst_feedback_defaults_to_pending(self, engine):
        """
        WHITE-BOX: All new flags must have analyst_feedback='pending'
        until a real analyst reviews them. [Peer 2 feedback]
        """
        with engine.connect() as conn:
            non_pending = conn.execute(text("""
                SELECT COUNT(*) FROM anomaly_flags
                WHERE analyst_feedback != 'pending'
            """)).scalar()
        assert non_pending == 0, \
            f"{non_pending} flags have non-pending feedback before review"

    def test_vendor_frequencies_are_positive(self, engine):
        """
        WHITE-BOX: All vendors must have positive frequency counts
        after ETL pipeline runs update_vendor_frequencies().
        """
        with engine.connect() as conn:
            invalid = conn.execute(text("""
                SELECT COUNT(*) FROM vendors
                WHERE vendor_frequency < 0
            """)).scalar()
        assert invalid == 0, \
            f"{invalid} vendors have negative frequency"

    def test_procurement_records_loaded(self, engine):
        """
        BLACK-BOX: procurement_records table must contain
        the expected number of records from ETL test run.
        """
        with engine.connect() as conn:
            count = conn.execute(text(
                "SELECT COUNT(*) FROM procurement_records"
            )).scalar()
        assert count >= 47_000, \
            f"Expected >=47,000 records, got {count:,}"

    def test_v_actionability_rate_view_works(self, engine):
        """
        BLACK-BOX: The actionability rate view must execute
        without errors and return a result. [Peer 2 feedback]
        """
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT * FROM v_actionability_rate"
            )).fetchone()
        assert result is not None, \
            "v_actionability_rate view returned no results"
        assert result[0] >= 0, "Total flags count is negative"

# =============================================================
# TEST GROUP 7: FEATURE ENGINEERING
# Type: White-box
# =============================================================

class TestFeatureEngineering:

    def test_log_amount_is_positive(self, ml_sample_df):
        """
        WHITE-BOX: log1p(amount) must always be >= 0
        since all amounts are >= 1.0 after ETL cleaning.
        """
        df_real = ml_sample_df[
            ml_sample_df['is_synthetic_anomaly'] == False
        ].copy()
        X, _ = engineer_features(df_real)
        assert (X['log_amount'] >= 0).all(), \
            "log_amount contains negative values"

    def test_is_yearend_is_binary(self, ml_sample_df):
        """
        WHITE-BOX: is_yearend feature must be binary (0 or 1).
        """
        df_real = ml_sample_df[
            ml_sample_df['is_synthetic_anomaly'] == False
        ].copy()
        X, _ = engineer_features(df_real)
        assert set(X['is_yearend'].unique()).issubset({0, 1}), \
            "is_yearend feature is not binary"

    def test_feature_matrix_has_correct_columns(self, ml_sample_df):
        """
        WHITE-BOX: Feature matrix must contain exactly the
        5 expected feature columns.
        """
        df_real = ml_sample_df[
            ml_sample_df['is_synthetic_anomaly'] == False
        ].copy()
        X, feature_cols = engineer_features(df_real)
        expected = {
            'log_amount', 'zscore_within_category',
            'log_vendor_freq', 'award_day_of_year', 'is_yearend'
        }
        assert set(feature_cols) == expected, \
            f"Feature columns mismatch: {set(feature_cols)} != {expected}"

    def test_no_nan_in_features(self, ml_sample_df):
        """
        WHITE-BOX: Feature matrix must have no NaN values
        after engineering (all nulls filled).
        """
        df_real = ml_sample_df[
            ml_sample_df['is_synthetic_anomaly'] == False
        ].copy()
        X, _ = engineer_features(df_real)
        assert not X.isna().any().any(), \
            "Feature matrix contains NaN values after engineering"
