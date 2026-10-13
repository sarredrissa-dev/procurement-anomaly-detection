# IntegriScan — Security Controls Documentation

**Government Procurement Anomaly Detection System**  
**Version:** 1.0  
**Last Updated:** October 2026  
**Author:** Edrissa Sarr — MSIT 5910 Capstone, University of the People  

---

## Overview

IntegriScan implements a defense-in-depth security architecture aligned with the **OWASP Top 10:2025** web application security framework. Security controls are distributed across all five system layers — data source, ETL pipeline, database, analytics, and presentation — rather than concentrated in a single security module. This document provides a consolidated reference of all security controls, their implementation location, and their OWASP alignment.

> **No credentials, API keys, or sensitive data are stored in this repository.**  
> All credentials are managed via `.env` file excluded from version control by `.gitignore`.

---

## OWASP Top 10:2025 Alignment

| OWASP Category | Risk | IntegriScan Control | Implemented In |
|----------------|------|--------------------|--------------:|
| A01:2025 — Broken Access Control | Unauthorized access to admin functions | Dual-layer RBAC at PostgreSQL engine + Streamlit app | `integriscan_ddl_scripts.sql` + `dashboard_app.py` |
| A02:2025 — Cryptographic Failures | Exposed credentials | `.env` file excluded from GitHub via `.gitignore` | `.gitignore` + `.env` |
| A05:2025 — Security Misconfiguration | SQL injection via unsanitized inputs | SQLAlchemy `text()` wrapper with named parameters | `ml_module.py` + `dashboard_app.py` |
| A09:2025 — Security Logging & Monitoring | Undetected unauthorized access | `audit_log` table records all access events | `integriscan_ddl_scripts.sql` + `dashboard_app.py` |

---

## Role-Based Access Control (RBAC)

### Database Level — PostgreSQL Roles

Two PostgreSQL roles enforce access control at the database engine level, independent of the application layer. Even a direct database connection bypassing the dashboard cannot exceed these permissions.

```sql
-- Analyst role — read only
CREATE ROLE integriscan_analyst WITH LOGIN;
GRANT CONNECT ON DATABASE procurement_db TO integriscan_analyst;
GRANT USAGE ON SCHEMA public TO integriscan_analyst;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO integriscan_analyst;

-- Admin role — read + write
CREATE ROLE integriscan_admin WITH LOGIN;
GRANT CONNECT ON DATABASE procurement_db TO integriscan_admin;
GRANT USAGE ON SCHEMA public TO integriscan_admin;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO integriscan_admin;
```

**File:** `database/integriscan_ddl_scripts.sql`

### Application Level — Streamlit Session State

Role-based page access is enforced in the Streamlit dashboard through session state checks on every page load. The Admin Settings page is restricted to administrator-role users only.

```python
# Role check on Admin Settings page
if st.session_state.get("user_role") != "admin":
    st.error("Access denied — administrator role required.")
    st.stop()
```

**File:** `dashboard/dashboard_app.py`

---

## Credential Management

### .env File

All database credentials are stored in a `.env` file that is **never committed to the GitHub repository**.

```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=procurement_db
DB_USER=integriscan_admin
DB_PASSWORD=<your_password_here>
```

### .gitignore Exclusion

The `.gitignore` file explicitly excludes the `.env` file:

```
# Environment variables — never commit credentials
.env
.env.*
```

### SQLAlchemy Secure Connection

Database connections use `URL.create()` to safely handle special characters in passwords without exposing credentials in connection strings:

```python
from sqlalchemy.engine import URL
db_url = URL.create(
    drivername="postgresql+psycopg2",
    username=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=int(os.getenv("DB_PORT", 5432)),
    database=os.getenv("DB_NAME")
)
```

**File:** `etl/etl_pipeline.py` + `ml/ml_module.py` + `dashboard/dashboard_app.py`

---

## SQL Injection Prevention

All database queries use SQLAlchemy's `text()` wrapper with named parameters, preventing SQL injection by separating query structure from user-supplied values.

```python
from sqlalchemy import text

# Safe parameterized query — user input never interpolated directly
query = text("""
    SELECT * FROM v_flagged_contracts
    WHERE flag_type = :flag_type
    AND isolation_forest_score <= :min_score
    ORDER BY isolation_forest_score ASC
    LIMIT 100
""")
df = pd.read_sql(query, engine, params={
    "flag_type": flag_type,
    "min_score": min_score
})
```

**File:** `dashboard/dashboard_app.py` + `ml/ml_module.py`

---

## Audit Logging

All significant system events are recorded in the `audit_log` table with timestamp, user role, action attempted, result, and resource accessed. This provides a complete, tamper-evident audit trail for accountability and security monitoring.

### Events Logged

| Event | Logged As |
|-------|-----------|
| Login attempt (success or failure) | `login_attempt` — success / denied |
| Admin page access | `admin_page_access` — success / denied |
| Analyst feedback submission | `submitted_feedback_actionable` / `submitted_feedback_not_actionable` |
| Contamination threshold change | `changed_contamination_threshold` |
| Unauthorized page access attempt | `admin_page_access` — denied |

### Audit Log Schema

```sql
CREATE TABLE audit_log (
    log_id          SERIAL PRIMARY KEY,
    user_role       VARCHAR(50),
    action_attempted VARCHAR(300),
    action_result   VARCHAR(50) CHECK (action_result IN ('success', 'denied', 'error')),
    resource_accessed VARCHAR(200),
    record_id_accessed INTEGER,
    logged_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**File:** `database/integriscan_ddl_scripts.sql` + `dashboard/dashboard_app.py`

---

## Known Limitations (Post-Capstone Roadmap)

The following security limitations are acknowledged and documented as Phase 2 post-Capstone enhancements:

| Limitation | Risk | Phase 2 Mitigation |
|------------|------|-------------------|
| Hardcoded credentials in dashboard login | Compromised admin account cannot be detected until incident | Multi-factor authentication (TOTP/SSO) |
| No server-side session management | Active sessions cannot be remotely revoked | Redis-backed session store with admin revocation |
| Credentials in local .env file | Local machine compromise exposes DB credentials | AWS Secrets Manager / Azure Key Vault |
| Single-person threshold changes | Malicious admin could suppress flags for specific vendors | Second-person approval workflow for config changes |

These limitations are fully documented in:
- `docs/IntegriScan_Ethical_Use_Data_Governance_Policy_v1.docx` — Section 5.3
- `docs/IntegriScan_Future_Work_Product_Roadmap_v3.docx` — Phase 2

---

## Security Testing

Security controls are verified through the automated test suite:

| Test | File | What It Verifies |
|------|------|-----------------|
| `test_rbac_roles_exist` | `tests/test_ml_module.py` | Both PostgreSQL RBAC roles exist |
| `test_system_enforces_rbac_roles` | `tests/test_system.py` | Analyst role cannot perform admin actions |
| `test_audit_log_records_system_activity` | `tests/test_system.py` | Audit log records access events correctly |

Run security tests:
```
pytest tests/ -v -k "rbac or audit"
```

---

## Reporting Security Issues

This is an academic Capstone project. If you identify a security issue in the codebase, please open a GitHub Issue with the label `security` at:

**github.com/sarredrissa-dev/procurement-anomaly-detection/issues**

---

*IntegriScan — Follow the money. Flag the risk.*
