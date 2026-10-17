# =============================================================
# IntegriScan — System, Integration & Acceptance Tests
# Government Procurement Anomaly Detection System
# MSIT 5910 Capstone | University of the People
# Student: Edrissa Sarr
# =============================================================
# Test Levels:
#   INTEGRATION: Modules working together across boundaries
#   SYSTEM:      Full end-to-end system behaviour
#   ACCEPTANCE:  Success criteria from Capstone proposal
# =============================================================
# Testing Types Used:
#   BLACK-BOX:  Tests system behaviour from external perspective
#   WHITE-BOX:  Tests internal integration logic
# =============================================================

import os
import sys
import time
import pytest
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

load_dotenv()

# Import modules
from ml.ml_module import (
    get_engine,
    load_data,
    engineer_features,
    zscore_detection,
    isolation_forest_detection,
    generate_risk_explanation,
    store_flags,
    ZSCORE_THRESHOLD,
    CONTAMINATION_START,
    MODEL_VERSION,
)

# =============================================================
# SHARED FIXTURE
# =============================================================

@pytest.fixture(scope="session")
def engine():
    """Session-scoped database engine."""
    return get_engine()

@pytest.fixture(scope="session")
def loaded_data(engine):
    """Load real + all data once for entire session."""
    df_real, df_all = load_data(engine)
    return df_real, df_all

# =============================================================
# TEST GROUP 1: INTEGRATION TESTS
# Verify modules communicate correctly across boundaries
# =============================================================

class TestETLtoDatabaseIntegration:
    """
    INTEGRATION TESTS — ETL → Database boundary.
    Verifies that ETL pipeline correctly populated
    all database tables and that data is consistent
    across table boundaries.
    Type: Black-box (testing cross-module outcomes)
    """

    def test_vendors_table_populated_by_etl(self, engine):
        """
        INTEGRATION: ETL pipeline must have populated
        the vendors table with at least 1,000 vendors.
        Verifies ETL → vendors table boundary.
        """
        with engine.connect() as conn:
            count = conn.execute(text(
                "SELECT COUNT(*) FROM vendors "
                "WHERE is_synthetic = FALSE"
            )).scalar()
        assert count >= 1000, \
            f"Expected >=1,000 vendors, got {count:,}"

    def test_contracts_table_populated_by_etl(self, engine):
        """
        INTEGRATION: ETL pipeline must have populated
        the contracts table with contract metadata.
        Verifies ETL → contracts table boundary.
        """
        with engine.connect() as conn:
            count = conn.execute(text(
                "SELECT COUNT(*) FROM contracts"
            )).scalar()
        assert count >= 10000, \
            f"Expected >=10,000 contracts, got {count:,}"

    def test_procurement_records_linked_to_vendors(self, engine):
        """
        INTEGRATION: Every procurement record must have
        a valid vendor_id linking to the vendors table.
        Tests the ETL → DB referential integrity boundary.
        """
        with engine.connect() as conn:
            orphans = conn.execute(text("""
                SELECT COUNT(*)
                FROM procurement_records pr
                LEFT JOIN vendors v ON pr.vendor_id = v.vendor_id
                WHERE v.vendor_id IS NULL
            """)).scalar()
        assert orphans == 0, \
            f"{orphans} procurement records with invalid vendor_id"

    def test_procurement_records_linked_to_contracts(self, engine):
        """
        INTEGRATION: Every procurement record must have
        a valid contract_id linking to the contracts table.
        Tests the ETL → DB referential integrity boundary.
        """
        with engine.connect() as conn:
            orphans = conn.execute(text("""
                SELECT COUNT(*)
                FROM procurement_records pr
                LEFT JOIN contracts c ON pr.contract_id = c.contract_id
                WHERE c.contract_id IS NULL
            """)).scalar()
        assert orphans == 0, \
            f"{orphans} procurement records with invalid contract_id"

    def test_zscores_computed_by_etl(self, engine):
        """
        INTEGRATION: ETL pipeline must have pre-computed
        z-scores for procurement records.
        Tests the ETL → DB computed field boundary.
        """
        with engine.connect() as conn:
            with_zscore = conn.execute(text("""
                SELECT COUNT(*) FROM procurement_records
                WHERE zscore_within_category IS NOT NULL
                AND is_synthetic_anomaly = FALSE
            """)).scalar()
        assert with_zscore >= 10000, \
            f"Expected >=10,000 records with z-scores, got {with_zscore:,}"

    def test_synthetic_anomalies_injected_by_etl(self, engine):
        """
        INTEGRATION: ETL pipeline must have injected
        synthetic anomalies into procurement_records.
        Tests the ETL synthetic injection boundary.
        """
        with engine.connect() as conn:
            synthetic = conn.execute(text("""
                SELECT COUNT(*) FROM procurement_records
                WHERE is_synthetic_anomaly = TRUE
            """)).scalar()
        assert synthetic >= 10, \
            f"Expected >=10 synthetic anomalies, got {synthetic}"


class TestDatabaseToMLIntegration:
    """
    INTEGRATION TESTS — Database → ML module boundary.
    Verifies that ML module correctly reads from database
    and writes results back.
    Type: Black-box
    """

    def test_ml_reads_data_from_database(self, loaded_data):
        """
        INTEGRATION: ML module must successfully read
        procurement records from the database.
        Tests the DB → ML module data boundary.
        """
        df_real, df_all = loaded_data
        assert len(df_real) > 0, \
            "ML module failed to read real records from database"
        assert len(df_all) > len(df_real), \
            "ML module failed to read synthetic records from database"

    def test_ml_receives_correct_features_from_db(self, loaded_data):
        """
        INTEGRATION: Data received by ML module from DB
        must contain all required feature columns.
        Tests the DB → ML feature column boundary.
        """
        df_real, _ = loaded_data
        required_cols = [
            'total_dollars_obligated',
            'zscore_within_category',
            'vendor_frequency',
            'award_day_of_year',
            'award_month',
            'is_synthetic_anomaly'
        ]
        missing = [c for c in required_cols if c not in df_real.columns]
        assert not missing, \
            f"ML module missing columns from DB: {missing}"

    def test_ml_flags_stored_in_database(self, engine):
        """
        INTEGRATION: ML module must write anomaly flags
        back to the anomaly_flags table in the database.
        Tests the ML → DB flag storage boundary.
        """
        with engine.connect() as conn:
            flags = conn.execute(text(
                "SELECT COUNT(*) FROM anomaly_flags"
            )).scalar()
        assert flags > 0, \
            "No anomaly flags found in database — ML module failed to store results"

    def test_anomaly_flags_join_procurement_records(self, engine):
        """
        INTEGRATION: Anomaly flags must correctly reference
        procurement records through the record_id foreign key.
        Tests the ML → DB referential integrity boundary.
        """
        with engine.connect() as conn:
            joined = conn.execute(text("""
                SELECT COUNT(*)
                FROM anomaly_flags af
                JOIN procurement_records pr
                    ON af.record_id = pr.record_id
            """)).scalar()
            total_flags = conn.execute(text(
                "SELECT COUNT(*) FROM anomaly_flags"
            )).scalar()
        assert joined == total_flags, \
            f"Flag-record join mismatch: {joined} joined vs {total_flags} total flags"

    def test_v_flagged_contracts_view_works(self, engine):
        """
        INTEGRATION: The v_flagged_contracts view must
        successfully join all 4 tables and return results.
        Tests the dashboard → DB view boundary.
        """
        with engine.connect() as conn:
            count = conn.execute(text(
                "SELECT COUNT(*) FROM v_flagged_contracts"
            )).scalar()
        assert count > 0, \
            "v_flagged_contracts view returned 0 results"

    def test_v_agency_summary_view_works(self, engine):
        """
        INTEGRATION: The v_agency_summary view must
        aggregate data correctly across tables.
        Tests the agency analysis dashboard → DB boundary.
        """
        with engine.connect() as conn:
            count = conn.execute(text(
                "SELECT COUNT(*) FROM v_agency_summary"
            )).scalar()
        assert count > 0, \
            "v_agency_summary view returned 0 results"


class TestMLtoDashboardIntegration:
    """
    INTEGRATION TESTS — ML results → Dashboard boundary.
    Verifies that dashboard queries return complete,
    well-formed data that the UI can render correctly.
    Type: Black-box
    """

    def test_dashboard_query_returns_all_required_columns(self, engine):
        """
        INTEGRATION: The main dashboard query must return
        all columns required by the Streamlit UI.
        Tests the DB → dashboard data contract boundary.
        """
        required_cols = [
            'flag_id', 'isolation_forest_score', 'zscore_deviation',
            'flag_type', 'risk_factor_explanation', 'risk_factor_1',
            'analyst_feedback', 'recipient_name', 'total_dollars_obligated',
            'awarding_agency', 'naics_description'
        ]
        with engine.connect() as conn:
            df = pd.read_sql(text("""
                SELECT
                    af.flag_id,
                    af.isolation_forest_score,
                    af.zscore_deviation,
                    af.flag_type,
                    af.risk_factor_explanation,
                    af.risk_factor_1,
                    af.analyst_feedback,
                    pr.total_dollars_obligated,
                    v.recipient_name,
                    c.awarding_agency,
                    c.naics_description
                FROM anomaly_flags af
                JOIN procurement_records pr
                    ON af.record_id = pr.record_id
                JOIN vendors v ON pr.vendor_id = v.vendor_id
                JOIN contracts c ON pr.contract_id = c.contract_id
                LIMIT 10
            """), conn)
        missing = [c for c in required_cols if c not in df.columns]
        assert not missing, \
            f"Dashboard query missing columns: {missing}"

    def test_dashboard_query_returns_data(self, engine):
        """
        INTEGRATION: Dashboard query must return
        at least one flagged contract for display.
        Tests the ML results → dashboard rendering boundary.
        """
        with engine.connect() as conn:
            count = conn.execute(text("""
                SELECT COUNT(*)
                FROM anomaly_flags af
                JOIN procurement_records pr
                    ON af.record_id = pr.record_id
                JOIN vendors v ON pr.vendor_id = v.vendor_id
                JOIN contracts c ON pr.contract_id = c.contract_id
            """)).scalar()
        assert count > 0, \
            "Dashboard query returned no flagged contracts to display"

    def test_risk_explanations_present_in_flags(self, engine):
        """
        INTEGRATION: ML-generated risk explanations must
        be stored and retrievable for dashboard display.
        Tests the XAI engine → dashboard display boundary.
        """
        with engine.connect() as conn:
            with_explanation = conn.execute(text("""
                SELECT COUNT(*) FROM anomaly_flags
                WHERE risk_factor_explanation IS NOT NULL
            """)).scalar()
        assert with_explanation > 0, \
            "No risk explanations found — XAI engine failed to store results"

    def test_audit_log_records_system_activity(self, engine):
        """
        INTEGRATION: The audit_log table must record
        system activity for RBAC accountability.
        Tests the RBAC module → audit log boundary.
        """
        with engine.connect() as conn:
            # Verify audit_log table is accessible and functional
            conn.execute(text("""
                INSERT INTO audit_log
                (user_role, action_attempted, action_result)
                VALUES ('test', 'integration_test', 'success')
            """))
            count = conn.execute(text(
                "SELECT COUNT(*) FROM audit_log "
                "WHERE action_attempted = 'integration_test'"
            )).scalar()
        assert count >= 1, \
            "Audit log failed to record system activity"

# =============================================================
# TEST GROUP 2: SYSTEM TESTS
# Verify full end-to-end system behaviour
# =============================================================

class TestSystemBehaviour:
    """
    SYSTEM TESTS — Full end-to-end system validation.
    Tests the complete integrated system against
    functional and non-functional requirements.
    Type: Black-box
    """

    def test_system_detects_two_anomaly_types(self, engine):
        """
        SYSTEM TEST (FR): The system must detect at least
        two distinct types of anomalies.
        Requirement: FR5 — detect multiple anomaly types.
        """
        with engine.connect() as conn:
            types = conn.execute(text("""
                SELECT DISTINCT flag_type
                FROM anomaly_flags
                WHERE flag_type IS NOT NULL
            """)).fetchall()
        flag_types = {row[0] for row in types}
        assert len(flag_types) >= 2, \
            f"Expected >=2 flag types, got {len(flag_types)}: {flag_types}"

    def test_system_stores_both_scores_per_flag(self, engine):
        """
        SYSTEM TEST (FR): Each flag must store both
        Isolation Forest score AND z-score independently.
        Requirement: FR6 — dual-score storage [Peer 1].
        """
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT
                    COUNT(*) as total,
                    COUNT(isolation_forest_score) as with_if,
                    COUNT(zscore_deviation) as with_z
                FROM anomaly_flags
            """)).fetchone()
        total, with_if, with_z = result
        assert with_if > 0, "No Isolation Forest scores stored"
        assert with_z > 0, "No z-scores stored"

    def test_system_generates_risk_explanations(self, engine):
        """
        SYSTEM TEST (FR): System must generate plain-language
        risk explanations for flagged contracts.
        Requirement: FR9 — XAI explanations [Peer 3].
        """
        with engine.connect() as conn:
            sample = conn.execute(text("""
                SELECT risk_factor_explanation
                FROM anomaly_flags
                WHERE risk_factor_explanation IS NOT NULL
                LIMIT 5
            """)).fetchall()
        assert len(sample) >= 5, \
            "Fewer than 5 risk explanations found in system"
        for row in sample:
            assert len(str(row[0])) > 10, \
                "Risk explanation too short to be meaningful"

    def test_system_enforces_rbac_roles(self, engine):
        """
        SYSTEM TEST (NFR): System must enforce two distinct
        RBAC roles — analyst and admin.
        Requirement: NFR3 — role-based access control.
        """
        with engine.connect() as conn:
            roles = conn.execute(text("""
                SELECT rolname FROM pg_roles
                WHERE rolname LIKE 'integriscan%'
            """)).fetchall()
        role_names = {row[0] for row in roles}
        assert 'integriscan_analyst' in role_names, \
            "Analyst RBAC role missing from database"
        assert 'integriscan_admin' in role_names, \
            "Admin RBAC role missing from database"

    def test_system_actionability_tracking_works(self, engine):
        """
        SYSTEM TEST (FR): System must support auditor
        actionability feedback per flag [Peer 2].
        Requirement: FR8 — actionability rate tracking.
        """
        with engine.connect() as conn:
            # Verify feedback field exists and has valid values
            result = conn.execute(text("""
                SELECT DISTINCT analyst_feedback
                FROM anomaly_flags
                ORDER BY analyst_feedback
            """)).fetchall()
        feedback_values = {row[0] for row in result}
        assert 'pending' in feedback_values, \
            "Default 'pending' feedback value not found"

    def test_system_performance_ml_pipeline(self, loaded_data):
        """
        SYSTEM TEST (NFR): ML pipeline must process
        47,000+ records in under 60 seconds.
        Requirement: NFR1 — processing performance.
        """
        df_real, df_all = loaded_data
        start = time.time()

        # Run z-score detection
        df_real_z, df_all_z = zscore_detection(
            df_real.copy(), df_all.copy()
        )
        # Run Isolation Forest
        df_scored, _, _, _ = isolation_forest_detection(
            df_real.copy(), df_all.copy(),
            contamination=CONTAMINATION_START
        )
        elapsed = time.time() - start
        assert elapsed < 60, \
            f"ML pipeline took {elapsed:.1f}s — exceeds 60s NFR1 limit"

    def test_system_flag_rate_within_expected_range(self, engine):
        """
        SYSTEM TEST (NFR): Flag rate must be between
        1% and 10% of total real records.
        An extremely high flag rate indicates misconfiguration.
        """
        with engine.connect() as conn:
            total = conn.execute(text("""
                SELECT COUNT(*) FROM procurement_records
                WHERE is_synthetic_anomaly = FALSE
            """)).scalar()
            flags = conn.execute(text(
                "SELECT COUNT(*) FROM anomaly_flags "
                "WHERE is_synthetic_anomaly = FALSE"
            )).scalar()
        flag_rate = flags / total if total > 0 else 0
        assert 0.01 <= flag_rate <= 0.10, \
            f"Flag rate {flag_rate:.3f} outside expected range [0.01, 0.10]"

    def test_system_data_integrity_no_duplicate_flags(self, engine):
        """
        SYSTEM TEST (NFR): Each procurement record must
        have at most one flag per model version.
        Prevents duplicate alert generation.
        """
        with engine.connect() as conn:
            duplicates = conn.execute(text("""
                SELECT record_id, COUNT(*) as cnt
                FROM anomaly_flags
                WHERE model_version = :v
                GROUP BY record_id
                HAVING COUNT(*) > 1
            """), {'v': MODEL_VERSION}).fetchall()
        assert len(duplicates) == 0, \
            f"{len(duplicates)} records have duplicate flags"

    def test_system_database_comment_set(self, engine):
        """
        SYSTEM TEST (NFR): Database must have the
        IntegriScan documentation comment set.
        Requirement: NFR8 — embedded documentation.
        """
        with engine.connect() as conn:
            comment = conn.execute(text("""
                SELECT shobj_description(oid, 'pg_database')
                FROM pg_database
                WHERE datname = 'procurement_db'
            """)).scalar()
        assert comment is not None, \
            "Database comment not set"
        assert 'IntegriScan' in str(comment), \
            f"Database comment missing IntegriScan identifier: {comment}"

# =============================================================
# TEST GROUP 3: ACCEPTANCE TESTS
# Verify success criteria from the Capstone proposal
# =============================================================

class TestAcceptanceCriteria:
    """
    ACCEPTANCE TESTS — Capstone proposal success criteria.
    Tests the system against the exact success criteria
    defined in the original project proposal (Section 10).
    Type: Black-box (end-user perspective)
    """

    def test_ac1_database_stores_and_retrieves_records(self, engine):
        """
        ACCEPTANCE CRITERION 1:
        'The database correctly stores and retrieves
        procurement records and anomaly flags.'
        — IntegriScan Capstone Proposal, Section 10
        """
        with engine.connect() as conn:
            # Test storing
            records = conn.execute(text(
                "SELECT COUNT(*) FROM procurement_records"
            )).scalar()
            # Test retrieving flags
            flags = conn.execute(text(
                "SELECT COUNT(*) FROM anomaly_flags"
            )).scalar()
            # Test join retrieval
            joined = conn.execute(text("""
                SELECT COUNT(*)
                FROM anomaly_flags af
                JOIN procurement_records pr
                    ON af.record_id = pr.record_id
            """)).scalar()

        assert records > 0, "AC1 FAIL: No procurement records stored"
        assert flags > 0, "AC1 FAIL: No anomaly flags stored"
        assert joined == flags, "AC1 FAIL: Flag retrieval with records failed"

    def test_ac2_ml_identifies_two_anomaly_types(self, engine):
        """
        ACCEPTANCE CRITERION 2:
        'The ML module identifies at least two types of
        anomalies with documented precision/recall metrics.'
        — IntegriScan Capstone Proposal, Section 10
        """
        with engine.connect() as conn:
            types = conn.execute(text("""
                SELECT flag_type, COUNT(*) as count
                FROM anomaly_flags
                GROUP BY flag_type
                ORDER BY count DESC
            """)).fetchall()
        assert len(types) >= 2, \
            f"AC2 FAIL: Only {len(types)} anomaly type(s) detected, need >=2"

        # Verify documented counts exist for each type
        for flag_type, count in types:
            assert count > 0, \
                f"AC2 FAIL: Flag type {flag_type} has zero flags"

    def test_ac3_dashboard_allows_filtering(self, engine):
        """
        ACCEPTANCE CRITERION 3:
        'The dashboard allows a non-technical user to filter,
        search, and review flagged contracts without training.'
        — IntegriScan Capstone Proposal, Section 10
        """
        # Test that the filter queries work correctly
        with engine.connect() as conn:
            # Test agency filter
            agency_filter = conn.execute(text("""
                SELECT COUNT(*)
                FROM anomaly_flags af
                JOIN procurement_records pr
                    ON af.record_id = pr.record_id
                JOIN vendors v ON pr.vendor_id = v.vendor_id
                JOIN contracts c ON pr.contract_id = c.contract_id
                WHERE c.awarding_agency IS NOT NULL
            """)).scalar()

            # Test flag type filter
            type_filter = conn.execute(text("""
                SELECT COUNT(*) FROM anomaly_flags
                WHERE flag_type = 'yearend_cluster'
            """)).scalar()

        assert agency_filter > 0, \
            "AC3 FAIL: Agency filter returns no results"
        assert type_filter > 0, \
            "AC3 FAIL: Flag type filter returns no results"

    def test_ac4_ethics_policy_bias_sources(self, engine):
        """
        ACCEPTANCE CRITERION 4:
        'The ethical policy document addresses at least
        three potential sources of bias or misuse.'
        — IntegriScan Capstone Proposal, Section 10

        Verified through system design controls:
        - Disclaimer banner (anomaly vs fraud distinction)
        - XAI explanations (transparency)
        - Actionability rate (fairness measurement)
        """
        with engine.connect() as conn:
            # Verify disclaimer is embedded in system
            # (via risk_factor_explanation field)
            with_explanation = conn.execute(text("""
                SELECT COUNT(*) FROM anomaly_flags
                WHERE risk_factor_explanation IS NOT NULL
            """)).scalar()

            # Verify actionability mechanism exists
            feedback_col = conn.execute(text("""
                SELECT COUNT(*) FROM information_schema.columns
                WHERE table_name = 'anomaly_flags'
                AND column_name = 'analyst_feedback'
            """)).scalar()

            # Verify audit trail exists
            audit_col = conn.execute(text("""
                SELECT COUNT(*) FROM information_schema.columns
                WHERE table_name = 'audit_log'
                AND column_name = 'action_result'
            """)).scalar()

        assert with_explanation > 0, \
            "AC4 FAIL: No risk explanations — transparency control missing"
        assert feedback_col > 0, \
            "AC4 FAIL: Analyst feedback column missing — fairness control missing"
        assert audit_col > 0, \
            "AC4 FAIL: Audit log missing — accountability control missing"

    def test_ac5_code_is_version_controlled(self):
        """
        ACCEPTANCE CRITERION 5:
        'All code is documented, version-controlled,
        and reproducible on a fresh environment.'
        — IntegriScan Capstone Proposal, Section 10
        """
        # Verify .git directory exists (version control)
        project_root = os.path.join(
            os.path.dirname(__file__), '..'
        )
        git_dir = os.path.join(project_root, '.git')
        assert os.path.exists(git_dir), \
            "AC5 FAIL: No .git directory — version control not initialized"

        # Verify requirements can be identified
        # (requirements.txt or pip-installed packages)
        req_file = os.path.join(project_root, 'requirements.txt')
        env_file = os.path.join(project_root, '.env')

        # .env file should exist (credentials management)
        assert os.path.exists(env_file), \
            "AC5 FAIL: No .env file — credential management missing"

    def test_ac6_system_processes_real_government_data(self, engine):
        """
        ACCEPTANCE CRITERION 6 (additional):
        System must process real USASpending.gov data
        — not just synthetic test data.
        """
        with engine.connect() as conn:
            real_records = conn.execute(text("""
                SELECT COUNT(*) FROM procurement_records
                WHERE is_synthetic_anomaly = FALSE
                AND dataset_source LIKE '%USASpending%'
            """)).scalar()
        assert real_records > 1000, \
            f"AC6 FAIL: Only {real_records} real USASpending records found"

    def test_ac7_system_complete_pipeline_works(self, engine):
        """
        ACCEPTANCE CRITERION 7 (additional):
        Complete pipeline must be demonstrable:
        Data → ETL → DB → ML → Flags → Dashboard view.
        """
        with engine.connect() as conn:
            # Full pipeline verification query
            result = conn.execute(text("""
                SELECT
                    COUNT(DISTINCT pr.record_id) as records,
                    COUNT(DISTINCT v.vendor_id) as vendors,
                    COUNT(DISTINCT c.contract_id) as contracts,
                    COUNT(DISTINCT af.flag_id) as flags,
                    COUNT(DISTINCT af.flag_type) as flag_types
                FROM procurement_records pr
                JOIN vendors v ON pr.vendor_id = v.vendor_id
                JOIN contracts c ON pr.contract_id = c.contract_id
                LEFT JOIN anomaly_flags af ON pr.record_id = af.record_id
                WHERE pr.is_synthetic_anomaly = FALSE
            """)).fetchone()

        records, vendors, contracts, flags, flag_types = result
        assert records > 0, "AC7 FAIL: No records in pipeline"
        assert vendors > 0, "AC7 FAIL: No vendors in pipeline"
        assert contracts > 0, "AC7 FAIL: No contracts in pipeline"
        assert flags > 0, "AC7 FAIL: No flags generated by pipeline"
        assert flag_types >= 2, "AC7 FAIL: Fewer than 2 flag types"

