-- ============================================================
-- Government Procurement Anomaly Detection System
-- IntegriScan — Database DDL Scripts
-- MSIT 5910 Capstone | University of the People
-- Student: Edrissa Sarr
-- Database: PostgreSQL 18
-- Created: October 2026
-- ============================================================

-- ============================================================
-- STEP 1: Create the database
-- Run this first in pgAdmin or psql as postgres superuser
-- ============================================================

-- DROP DATABASE IF EXISTS procurement_db;
CREATE DATABASE procurement_db
    WITH
    OWNER = postgres
    ENCODING = 'UTF8'
    LC_COLLATE = 'en_US.UTF-8'
    LC_CTYPE = 'en_US.UTF-8'
    TEMPLATE = template0;

COMMENT ON DATABASE procurement_db IS
    'IntegriScan — Government Procurement Anomaly Detection System';

-- ============================================================
-- Connect to procurement_db before running the rest
-- \c procurement_db  (in psql)
-- or switch to procurement_db in pgAdmin
-- ============================================================

-- ============================================================
-- TABLE 1: vendors
-- Stores vendor/recipient information
-- ============================================================

CREATE TABLE IF NOT EXISTS vendors (
    vendor_id           SERIAL PRIMARY KEY,
    recipient_name      VARCHAR(500)    NOT NULL,
    recipient_name_raw  VARCHAR(500),           -- original unmodified name
    vendor_category     VARCHAR(100),           -- small business, large business etc
    state               VARCHAR(100),
    country             VARCHAR(100)    DEFAULT 'United States',
    vendor_frequency    INTEGER         DEFAULT 0,  -- number of contracts awarded
    is_synthetic        BOOLEAN         DEFAULT FALSE,  -- TRUE for injected test vendors
    created_at          TIMESTAMP       DEFAULT CURRENT_TIMESTAMP
);

-- Index for fast vendor name lookup (anomaly detection uses this heavily)
CREATE INDEX IF NOT EXISTS idx_vendors_name
    ON vendors (recipient_name);

CREATE INDEX IF NOT EXISTS idx_vendors_frequency
    ON vendors (vendor_frequency DESC);

COMMENT ON TABLE vendors IS
    'Stores unique vendor/recipient information extracted from USASpending.gov dataset';
COMMENT ON COLUMN vendors.is_synthetic IS
    'TRUE for vendors injected as part of synthetic anomaly testing';
COMMENT ON COLUMN vendors.vendor_frequency IS
    'Count of contracts awarded to this vendor — used as ML feature';

-- ============================================================
-- TABLE 2: contracts
-- Stores contract metadata
-- ============================================================

CREATE TABLE IF NOT EXISTS contracts (
    contract_id         SERIAL PRIMARY KEY,
    award_id            VARCHAR(200)    UNIQUE,     -- USASpending unique award ID
    piid                VARCHAR(200),               -- Procurement Instrument Identifier
    awarding_agency     VARCHAR(300),
    funding_agency      VARCHAR(300),
    contract_type       VARCHAR(200),               -- Fixed Price, Cost Plus etc
    naics_code          VARCHAR(20),                -- Industry classification
    naics_description   VARCHAR(500),
    period_start        DATE,
    period_end          DATE,
    award_day_of_year   INTEGER,                    -- Day of year (1-365) — ML feature
    award_month         INTEGER,                    -- Month (1-12) — ML feature
    fiscal_year         INTEGER,
    is_synthetic        BOOLEAN         DEFAULT FALSE,
    created_at          TIMESTAMP       DEFAULT CURRENT_TIMESTAMP
);

-- Index for agency filtering (dashboard uses this)
CREATE INDEX IF NOT EXISTS idx_contracts_agency
    ON contracts (awarding_agency);

CREATE INDEX IF NOT EXISTS idx_contracts_naics
    ON contracts (naics_code);

CREATE INDEX IF NOT EXISTS idx_contracts_fiscal_year
    ON contracts (fiscal_year);

CREATE INDEX IF NOT EXISTS idx_contracts_award_doy
    ON contracts (award_day_of_year);

COMMENT ON TABLE contracts IS
    'Stores contract metadata from USASpending.gov FY2023 DoD dataset';
COMMENT ON COLUMN contracts.award_day_of_year IS
    'Day of year contract was awarded (1-365) — used to detect fiscal year-end clustering anomalies';

-- ============================================================
-- TABLE 3: procurement_records
-- Stores individual contract line items with financial data
-- This is the main fact table
-- ============================================================

CREATE TABLE IF NOT EXISTS procurement_records (
    record_id                   SERIAL PRIMARY KEY,
    vendor_id                   INTEGER     NOT NULL
                                REFERENCES vendors(vendor_id)
                                ON DELETE RESTRICT,
    contract_id                 INTEGER     NOT NULL
                                REFERENCES contracts(contract_id)
                                ON DELETE RESTRICT,
    total_dollars_obligated     DECIMAL(18,2),      -- Primary amount field
    current_total_value         DECIMAL(18,2),      -- Current contract ceiling
    potential_total_value       DECIMAL(18,2),      -- Maximum possible value
    zscore_within_category      DECIMAL(10,4),      -- Pre-computed z-score vs NAICS mean
    category_mean               DECIMAL(18,2),      -- Mean for this NAICS category
    category_std                DECIMAL(18,2),      -- Std dev for this NAICS category
    dataset_source              VARCHAR(100)    DEFAULT 'USASpending.gov FY2023 DoD',
    is_synthetic_anomaly        BOOLEAN         DEFAULT FALSE,  -- TRUE = injected test record
    synthetic_anomaly_type      VARCHAR(100),   -- value_outlier, vendor_duplicate etc
    loaded_at                   TIMESTAMP       DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for ML module queries
CREATE INDEX IF NOT EXISTS idx_records_vendor
    ON procurement_records (vendor_id);

CREATE INDEX IF NOT EXISTS idx_records_contract
    ON procurement_records (contract_id);

CREATE INDEX IF NOT EXISTS idx_records_amount
    ON procurement_records (total_dollars_obligated DESC);

CREATE INDEX IF NOT EXISTS idx_records_zscore
    ON procurement_records (zscore_within_category DESC);

CREATE INDEX IF NOT EXISTS idx_records_synthetic
    ON procurement_records (is_synthetic_anomaly);

COMMENT ON TABLE procurement_records IS
    'Main fact table storing individual contract records with financial data and pre-computed ML features';
COMMENT ON COLUMN procurement_records.zscore_within_category IS
    'Z-score of total_dollars_obligated relative to NAICS category mean — computed during ETL';
COMMENT ON COLUMN procurement_records.is_synthetic_anomaly IS
    'TRUE for records injected as synthetic anomalies for ML model evaluation [Peer 1 & 4 feedback]';

-- ============================================================
-- TABLE 4: anomaly_flags
-- Stores ML detection results
-- Both IF score and z-score stored independently [Peer 1]
-- Auditor actionability feedback stored [Peer 2]
-- Explainable risk factors stored [Peer 3]
-- ============================================================

CREATE TABLE IF NOT EXISTS anomaly_flags (
    flag_id                     SERIAL PRIMARY KEY,
    record_id                   INTEGER     NOT NULL
                                REFERENCES procurement_records(record_id)
                                ON DELETE CASCADE,

    -- ML Detection Results [Peer 1: both scores stored independently]
    isolation_forest_score      DECIMAL(10,6),      -- Range: -1 (anomaly) to 1 (normal)
    isolation_forest_flagged    BOOLEAN     DEFAULT FALSE,
    zscore_deviation            DECIMAL(10,4),      -- Standard deviations from category mean
    zscore_flagged              BOOLEAN     DEFAULT FALSE,
    flag_type                   VARCHAR(100),       -- price_outlier, vendor_duplicate, yend_cluster

    -- Explainable Risk Factors [Peer 3: plain language explanation]
    risk_factor_explanation     TEXT,               -- e.g. "Contract value 4.2σ above category mean"
    risk_factor_1               VARCHAR(500),       -- Primary factor
    risk_factor_2               VARCHAR(500),       -- Secondary factor (if any)
    risk_factor_3               VARCHAR(500),       -- Tertiary factor (if any)

    -- Contamination parameter used [Peer 4]
    contamination_threshold     DECIMAL(5,3)    DEFAULT 0.050,

    -- Synthetic anomaly tracking
    is_synthetic_anomaly        BOOLEAN         DEFAULT FALSE,
    synthetic_anomaly_type      VARCHAR(100),

    -- Audit trail
    flagged_at                  TIMESTAMP       DEFAULT CURRENT_TIMESTAMP,
    model_version               VARCHAR(50)     DEFAULT 'v1.0',

    -- Auditor Actionability Feedback [Peer 2: feedback mechanism]
    analyst_feedback            VARCHAR(20)     DEFAULT 'pending'
                                CHECK (analyst_feedback IN
                                    ('actionable', 'not_actionable', 'pending')),
    feedback_notes              TEXT,
    feedback_timestamp          TIMESTAMP,
    reviewed_by_role            VARCHAR(50)     DEFAULT 'analyst'
);

-- Indexes for dashboard queries
CREATE INDEX IF NOT EXISTS idx_flags_record
    ON anomaly_flags (record_id);

CREATE INDEX IF NOT EXISTS idx_flags_if_score
    ON anomaly_flags (isolation_forest_score ASC);  -- Most negative = most anomalous

CREATE INDEX IF NOT EXISTS idx_flags_zscore
    ON anomaly_flags (zscore_deviation DESC);

CREATE INDEX IF NOT EXISTS idx_flags_feedback
    ON anomaly_flags (analyst_feedback);

CREATE INDEX IF NOT EXISTS idx_flags_type
    ON anomaly_flags (flag_type);

CREATE INDEX IF NOT EXISTS idx_flags_flagged_at
    ON anomaly_flags (flagged_at DESC);

COMMENT ON TABLE anomaly_flags IS
    'Stores ML anomaly detection results. Both IF score and z-score stored independently [Peer 1]. Auditor feedback enables actionability rate calculation [Peer 2]. Risk factor explanation supports XAI [Peer 3].';
COMMENT ON COLUMN anomaly_flags.isolation_forest_score IS
    'Isolation Forest anomaly score: more negative = stronger anomaly. Range -1 to 1.';
COMMENT ON COLUMN anomaly_flags.risk_factor_explanation IS
    'Plain language explanation of why this contract was flagged [Peer 3 — XAI principle]';
COMMENT ON COLUMN anomaly_flags.analyst_feedback IS
    'Auditor actionability feedback — enables actionability rate calculation targeting 85% [Peer 2]';

-- ============================================================
-- TABLE 5: audit_log
-- Stores access control events for RBAC [OWASP A01:2025]
-- ============================================================

CREATE TABLE IF NOT EXISTS audit_log (
    log_id              SERIAL PRIMARY KEY,
    user_role           VARCHAR(50),            -- analyst or admin
    action_attempted    VARCHAR(300),           -- what the user tried to do
    action_result       VARCHAR(50)
                        CHECK (action_result IN
                            ('success', 'denied', 'error')),
    resource_accessed   VARCHAR(200),           -- which table/endpoint was accessed
    record_id_accessed  INTEGER,                -- if a specific record was accessed
    ip_address          VARCHAR(50),
    session_token_hash  VARCHAR(100),           -- hashed session token (never store raw)
    logged_at           TIMESTAMP       DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_audit_role
    ON audit_log (user_role);

CREATE INDEX IF NOT EXISTS idx_audit_result
    ON audit_log (action_result);

CREATE INDEX IF NOT EXISTS idx_audit_logged_at
    ON audit_log (logged_at DESC);

COMMENT ON TABLE audit_log IS
    'Stores all access control events for RBAC audit trail. Aligned to OWASP A01:2025 Broken Access Control mitigation.';

-- ============================================================
-- VIEWS: Useful pre-built queries for the dashboard
-- ============================================================

-- View: Flagged contracts with full details for dashboard
CREATE OR REPLACE VIEW v_flagged_contracts AS
SELECT
    af.flag_id,
    af.isolation_forest_score,
    af.zscore_deviation,
    af.isolation_forest_flagged,
    af.zscore_flagged,
    af.flag_type,
    af.risk_factor_explanation,
    af.risk_factor_1,
    af.risk_factor_2,
    af.risk_factor_3,
    af.analyst_feedback,
    af.flagged_at,
    af.is_synthetic_anomaly,
    pr.record_id,
    pr.total_dollars_obligated,
    pr.current_total_value,
    pr.zscore_within_category,
    v.recipient_name,
    v.vendor_frequency,
    v.state,
    c.award_id,
    c.awarding_agency,
    c.contract_type,
    c.naics_code,
    c.naics_description,
    c.period_start,
    c.period_end,
    c.award_day_of_year,
    c.fiscal_year
FROM anomaly_flags af
JOIN procurement_records pr ON af.record_id = pr.record_id
JOIN vendors v ON pr.vendor_id = v.vendor_id
JOIN contracts c ON pr.contract_id = c.contract_id
ORDER BY af.isolation_forest_score ASC;

COMMENT ON VIEW v_flagged_contracts IS
    'Main dashboard view joining anomaly flags with contract and vendor details';

-- View: Actionability rate summary [Peer 2]
CREATE OR REPLACE VIEW v_actionability_rate AS
SELECT
    COUNT(*) AS total_flags,
    COUNT(CASE WHEN analyst_feedback = 'actionable' THEN 1 END) AS actionable_count,
    COUNT(CASE WHEN analyst_feedback = 'not_actionable' THEN 1 END) AS not_actionable_count,
    COUNT(CASE WHEN analyst_feedback = 'pending' THEN 1 END) AS pending_count,
    ROUND(
        100.0 * COUNT(CASE WHEN analyst_feedback = 'actionable' THEN 1 END)
        / NULLIF(COUNT(CASE WHEN analyst_feedback != 'pending' THEN 1 END), 0),
        1
    ) AS actionability_rate_pct
FROM anomaly_flags
WHERE is_synthetic_anomaly = FALSE;

COMMENT ON VIEW v_actionability_rate IS
    'Live actionability rate calculation — target 85% per Herreros-Martinez et al. (2025) [Peer 2]';

-- View: Agency summary statistics
CREATE OR REPLACE VIEW v_agency_summary AS
SELECT
    c.awarding_agency,
    COUNT(pr.record_id) AS total_contracts,
    COUNT(af.flag_id) AS total_flags,
    ROUND(100.0 * COUNT(af.flag_id) / NULLIF(COUNT(pr.record_id), 0), 2) AS flag_rate_pct,
    ROUND(AVG(pr.total_dollars_obligated), 2) AS avg_contract_value,
    SUM(pr.total_dollars_obligated) AS total_obligated
FROM procurement_records pr
JOIN contracts c ON pr.contract_id = c.contract_id
LEFT JOIN anomaly_flags af ON pr.record_id = af.record_id
GROUP BY c.awarding_agency
ORDER BY total_flags DESC;

COMMENT ON VIEW v_agency_summary IS
    'Agency-level summary statistics for dashboard summary panel';

-- ============================================================
-- ROLES: RBAC implementation [OWASP A01:2025]
-- ============================================================

-- Create analyst role (read-only)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'integriscan_analyst') THEN
        CREATE ROLE integriscan_analyst LOGIN PASSWORD 'analyst_password_change_me';
    END IF;
END $$;

-- Create admin role (read + write)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'integriscan_admin') THEN
        CREATE ROLE integriscan_admin LOGIN PASSWORD 'admin_password_change_me';
    END IF;
END $$;

-- Grant permissions to analyst role (READ ONLY)
GRANT CONNECT ON DATABASE procurement_db TO integriscan_analyst;
GRANT USAGE ON SCHEMA public TO integriscan_analyst;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO integriscan_analyst;
GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO integriscan_analyst;

-- Grant permissions to admin role (READ + WRITE)
GRANT CONNECT ON DATABASE procurement_db TO integriscan_admin;
GRANT USAGE ON SCHEMA public TO integriscan_admin;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO integriscan_admin;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO integriscan_admin;

-- Prevent analyst from modifying anomaly_flags contamination threshold
REVOKE UPDATE (contamination_threshold) ON anomaly_flags FROM integriscan_analyst;

-- ============================================================
-- VERIFICATION QUERIES
-- Run these after setup to confirm everything is working
-- ============================================================

-- Check all tables were created
SELECT table_name, table_type
FROM information_schema.tables
WHERE table_schema = 'public'
ORDER BY table_name;

-- Check all indexes
SELECT indexname, tablename
FROM pg_indexes
WHERE schemaname = 'public'
ORDER BY tablename, indexname;

-- Check all views
SELECT viewname
FROM pg_views
WHERE schemaname = 'public'
ORDER BY viewname;

-- Check RBAC roles
SELECT rolname, rolcanlogin
FROM pg_roles
WHERE rolname LIKE 'integriscan%';

-- ============================================================
-- END OF DDL SCRIPTS
-- ============================================================
