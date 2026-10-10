# =============================================================
# IntegriScan — Streamlit Dashboard
# Government Procurement Anomaly Detection System
# MSIT 5910 Capstone | University of the People
# Student: Edrissa Sarr
# =============================================================
# Features:
#   - Flagged contracts with IF + z-score side by side [Peer 1]
#   - Anomaly vs fraud disclaimer on every flag [Peers 1,2,3]
#   - Explainable risk factors per flag [Peer 3]
#   - Auditor actionability feedback [Peer 2]
#   - Live actionability rate (target 85%) [Peer 2]
#   - Agency and summary statistics
#   - RBAC — analyst vs admin roles [Security]
# =============================================================

import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

# =============================================================
# PAGE CONFIGURATION — Must be first Streamlit command
# =============================================================

st.set_page_config(
    page_title="IntegriScan — Procurement Anomaly Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================
# LOAD ENVIRONMENT AND DATABASE
# =============================================================

load_dotenv()

@st.cache_resource
def get_engine():
    """Cached database engine — created once per session."""
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
# CUSTOM CSS — IntegriScan Brand Styling
# =============================================================

st.markdown("""
<style>
    /* Main background */
    .main { background-color: #F8F9FA; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1F4E79;
        color: white;
    }
    [data-testid="stSidebar"] .stMarkdown { color: white; }
    [data-testid="stSidebar"] label { color: white !important; }
    [data-testid="stSidebar"] .stSelectbox label { color: white !important; }

    /* Disclaimer banner */
    .disclaimer-banner {
        background-color: #FFF3CD;
        border: 2px solid #FFC107;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0 16px 0;
        font-weight: 600;
        color: #856404;
    }

    /* Flag card */
    .flag-card {
        background-color: white;
        border: 1px solid #DEE2E6;
        border-left: 4px solid #C00000;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }

    /* Score badge */
    .score-if {
        background-color: #C00000;
        color: white;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 14px;
    }
    .score-z {
        background-color: #1F4E79;
        color: white;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 14px;
    }

    /* Metric card */
    .metric-card {
        background-color: white;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Header */
    .integriscan-header {
        background: linear-gradient(135deg, #1F4E79, #2E75B6);
        color: white;
        padding: 20px 24px;
        border-radius: 8px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================
# RBAC — ROLE-BASED ACCESS CONTROL [OWASP A01:2025]
# =============================================================

def check_authentication():
    """
    Simple RBAC authentication.
    In production: replace with proper authentication system.
    Analyst: read-only access
    Admin: read + write access to configuration
    """
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.user_role = None

def login_page():
    """Render login page."""
    st.markdown("""
    <div class="integriscan-header">
        <h1 style="margin:0; font-size:32px;">🔍 IntegriScan</h1>
        <p style="margin:4px 0 0 0; font-style:italic; color:#BDD7EE;">
            Follow the money. Flag the risk.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Please log in to access the dashboard")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input(
            "Password", type="password", placeholder="Enter password"
        )

        if st.button("Login", use_container_width=True, type="primary"):
            # Simplified credentials for Capstone demo
            # Production: use proper authentication with hashed passwords
            if username == "analyst" and password == "analyst123":
                st.session_state.authenticated = True
                st.session_state.user_role = "analyst"
                st.session_state.username = "analyst"
                st.rerun()
            elif username == "admin" and password == "admin123":
                st.session_state.authenticated = True
                st.session_state.user_role = "admin"
                st.session_state.username = "admin"
                st.rerun()
            else:
                st.error("Invalid credentials. Please try again.")
                # Log failed attempt
                try:
                    engine = get_engine()
                    with engine.begin() as conn:
                        conn.execute(text("""
                            INSERT INTO audit_log
                            (user_role, action_attempted, action_result)
                            VALUES ('unknown', 'login_attempt', 'denied')
                        """))
                except Exception:
                    pass

        st.markdown("---")
        st.markdown("""
        **Demo credentials:**
        - Analyst: `analyst` / `analyst123` (read-only)
        - Admin: `admin` / `admin123` (read + write)
        """)

# =============================================================
# DATA LOADING FUNCTIONS
# =============================================================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_summary_stats(_engine):
    """Load dashboard summary statistics."""
    with _engine.connect() as conn:
        stats = conn.execute(text("""
            SELECT
                (SELECT COUNT(*) FROM procurement_records
                 WHERE is_synthetic_anomaly = FALSE) as total_contracts,
                (SELECT COUNT(*) FROM anomaly_flags) as total_flags,
                (SELECT COUNT(*) FROM vendors
                 WHERE is_synthetic = FALSE) as total_vendors,
                (SELECT COUNT(DISTINCT awarding_agency)
                 FROM contracts) as total_agencies
        """)).fetchone()

        actionability = conn.execute(text("""
            SELECT * FROM v_actionability_rate
        """)).fetchone()

    return stats, actionability

@st.cache_data(ttl=60)
def load_flagged_contracts(_engine, agency_filter=None,
                            flag_type_filter=None,
                            min_if_score=None):
    """Load flagged contracts with filters applied."""
    try:
        # Build filter conditions
        conditions = ["1=1"]
        params = {}

        if agency_filter and agency_filter != "All Agencies":
            conditions.append("c.awarding_agency = :agency")
            params["agency"] = agency_filter

        if flag_type_filter and flag_type_filter != "All Types":
            conditions.append("af.flag_type = :flag_type")
            params["flag_type"] = flag_type_filter

        where_clause = " AND ".join(conditions)

        query = text(f"""
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
                af.analyst_feedback,
                af.flagged_at,
                af.is_synthetic_anomaly,
                pr.record_id,
                pr.total_dollars_obligated,
                pr.zscore_within_category,
                v.recipient_name,
                v.vendor_frequency,
                c.award_id,
                c.awarding_agency,
                c.naics_description,
                c.award_day_of_year,
                c.fiscal_year
            FROM anomaly_flags af
            JOIN procurement_records pr
                ON af.record_id = pr.record_id
            JOIN vendors v
                ON pr.vendor_id = v.vendor_id
            JOIN contracts c
                ON pr.contract_id = c.contract_id
            WHERE {where_clause}
            ORDER BY af.flag_id DESC
            LIMIT 100
        """)

        with _engine.connect() as conn:
            df = pd.read_sql(query, conn, params=params)
        return df

    except Exception as e:
        st.error(f"Query error: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_agency_summary(_engine):
    """Load agency summary statistics."""
    with _engine.connect() as conn:
        df = pd.read_sql(text("SELECT * FROM v_agency_summary"), conn)
    return df

@st.cache_data(ttl=300)
def load_agencies(_engine):
    """Load list of agencies for filter dropdown."""
    with _engine.connect() as conn:
        result = conn.execute(text("""
            SELECT DISTINCT awarding_agency
            FROM contracts
            WHERE awarding_agency IS NOT NULL
            ORDER BY awarding_agency
        """))
        return ["All Agencies"] + [row[0] for row in result]

# =============================================================
# FEEDBACK FUNCTION [Peer 2]
# =============================================================

def submit_feedback(engine, flag_id, feedback, username):
    """
    Submit auditor actionability feedback for a flag.
    Updates analyst_feedback field in anomaly_flags table.
    Logs action to audit_log for accountability.
    [Peer 2 feedback — actionability mechanism]
    """
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                UPDATE anomaly_flags
                SET analyst_feedback = :feedback,
                    feedback_timestamp = CURRENT_TIMESTAMP,
                    reviewed_by_role = :role
                WHERE flag_id = :flag_id
            """), {
                'feedback': feedback,
                'flag_id': flag_id,
                'role': username
            })
            # Log to audit trail
            conn.execute(text("""
                INSERT INTO audit_log
                (user_role, action_attempted, action_result, record_id_accessed)
                VALUES (:role, :action, 'success', :record_id)
            """), {
                'role': username,
                'action': f'submitted_feedback_{feedback}',
                'record_id': flag_id
            })
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"Error submitting feedback: {e}")
        return False

# =============================================================
# FORMAT HELPERS
# =============================================================

def format_currency(value):
    """Format number as USD currency."""
    if value is None:
        return "N/A"
    if value >= 1_000_000:
        return f"${value/1_000_000:.2f}M"
    elif value >= 1_000:
        return f"${value/1_000:.1f}K"
    return f"${value:,.2f}"

def format_if_score(score):
    """Format IF score with color indicator."""
    if score is None:
        return "N/A"
    if score < -0.6:
        return f"🔴 {score:.3f} (Strong)"
    elif score < -0.4:
        return f"🟠 {score:.3f} (Moderate)"
    else:
        return f"🟡 {score:.3f} (Mild)"

def format_flag_type(flag_type):
    """Format flag type as readable label."""
    labels = {
        'price_outlier': '💰 Price Outlier',
        'price_outlier_yearend': '💰📅 Price Outlier + Year-End',
        'yearend_cluster': '📅 Year-End Cluster',
        'vendor_duplicate': '👥 Vendor Duplicate',
        'isolation_forest': '🤖 IF Algorithm',
    }
    return labels.get(flag_type, flag_type)

# =============================================================
# MAIN DASHBOARD
# =============================================================

def main_dashboard():
    """Render the main dashboard interface."""
    engine = get_engine()
    username = st.session_state.get('username', 'analyst')
    user_role = st.session_state.get('user_role', 'analyst')

    # ── SIDEBAR ──
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 10px 0;'>
            <h2 style='color:white; margin:0;'>🔍 IntegriScan</h2>
            <p style='color:#BDD7EE; font-style:italic; font-size:12px;
                      margin:4px 0;'>
                Follow the money. Flag the risk.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Navigation
        st.markdown(
            "<p style='color:#BDD7EE; font-size:12px; "
            "font-weight:bold;'>NAVIGATION</p>",
            unsafe_allow_html=True
        )
        page = st.radio(
            "Select Page",
            ["📊 Dashboard Overview",
             "🚩 Flagged Contracts",
             "📈 Agency Analysis",
             "⚙️ Admin Settings"
             if user_role == "admin" else "🔒 Admin (Restricted)"],
            label_visibility="collapsed"
        )

        st.markdown("---")

        # Role badge
        role_color = "#375623" if user_role == "admin" else "#2E75B6"
        st.markdown(f"""
        <div style='background:{role_color}; padding:8px 12px;
                    border-radius:6px; margin-top:8px;'>
            <p style='color:white; margin:0; font-size:12px;'>
                👤 Logged in as: <b>{username}</b><br/>
                Role: <b>{user_role.upper()}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Filters (show on Flagged Contracts page)
        if "Flagged" in page:
            st.markdown(
                "<p style='color:#BDD7EE; font-size:12px; "
                "font-weight:bold;'>FILTERS</p>",
                unsafe_allow_html=True
            )
            agencies = load_agencies(engine)
            agency_filter = st.selectbox(
                "Agency", agencies, key="agency_filter"
            )
            flag_type_filter = st.selectbox(
                "Flag Type",
                ["All Types", "price_outlier", "price_outlier_yearend",
                 "yearend_cluster", "vendor_duplicate"],
                key="flag_type_filter"
            )
            min_if_score = st.slider(
                "Min IF Anomaly Score",
                min_value=-1.0, max_value=0.0,
                value=-1.0, step=0.05,
                key="if_score_filter",
                help="More negative = stronger anomaly"
            )
        else:
            agency_filter = "All Agencies"
            flag_type_filter = "All Types"
            min_if_score = -1.0

        # Logout
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user_role = None
            st.rerun()

    # ── MAIN CONTENT ──

    # Header
    st.markdown("""
    <div class="integriscan-header">
        <h1 style="margin:0; font-size:28px;">
            🔍 IntegriScan — Government Procurement Anomaly Detection
        </h1>
        <p style="margin:4px 0 0 0; color:#BDD7EE; font-size:14px;">
            FY2023 Department of Defense Contracts | USASpending.gov
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── PAGE: DASHBOARD OVERVIEW ──
    if "Overview" in page:
        render_overview(engine)

    # ── PAGE: FLAGGED CONTRACTS ──
    elif "Flagged" in page:
        render_flagged_contracts(
            engine, agency_filter,
            flag_type_filter, min_if_score,
            username, user_role
        )

    # ── PAGE: AGENCY ANALYSIS ──
    elif "Agency" in page:
        render_agency_analysis(engine)

    # ── PAGE: ADMIN SETTINGS ──
    elif "Admin Settings" in page and user_role == "admin":
        render_admin_settings(engine, username)

    elif "Restricted" in page:
        st.error(
            "🔒 Access Denied: Admin Settings require administrator role. "
            "Contact your system administrator to request elevated access."
        )
        # Log access attempt
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO audit_log
                (user_role, action_attempted, action_result)
                VALUES (:role, 'access_admin_settings', 'denied')
            """), {'role': user_role})

# =============================================================
# PAGE: OVERVIEW
# =============================================================

def render_overview(engine):
    """Render the dashboard overview page."""

    stats, actionability = load_summary_stats(engine)

    # Summary metrics row
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Total Contracts",
            f"{stats[0]:,}",
            help="Total real procurement records analyzed"
        )
    with col2:
        st.metric(
            "Total Flags",
            f"{stats[1]:,}",
            help="Total anomaly flags generated by ML module"
        )
    with col3:
        st.metric(
            "Unique Vendors",
            f"{stats[2]:,}",
            help="Unique vendor/recipient entities in dataset"
        )
    with col4:
        st.metric(
            "Agencies",
            f"{stats[3]:,}",
            help="Number of distinct awarding agencies"
        )
    with col5:
        # Actionability rate [Peer 2]
        total_reviewed = (
            (actionability[1] or 0) + (actionability[2] or 0)
        )
        if total_reviewed > 0:
            rate = f"{actionability[4]}%"
            delta = f"Target: 85%"
        else:
            rate = "Pending"
            delta = "Awaiting analyst review"
        st.metric(
            "Actionability Rate",
            rate,
            delta=delta,
            help="% of reviewed flags confirmed actionable by auditors"
        )

    st.markdown("---")

    # Flag breakdown chart
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Flags by Type")
        try:
            with engine.connect() as conn:
                flag_data = pd.read_sql(text("""
                    SELECT
                        flag_type,
                        COUNT(*) as count,
                        ROUND(AVG(isolation_forest_score)::numeric, 3)
                            as avg_if_score,
                        ROUND(AVG(zscore_deviation)::numeric, 2)
                            as avg_zscore
                    FROM anomaly_flags
                    GROUP BY flag_type
                    ORDER BY count DESC
                """), conn)

            if not flag_data.empty:
                fig, ax = plt.subplots(figsize=(8, 4))
                colors = ['#C00000', '#1F4E79', '#2E75B6',
                          '#7030A0', '#375623']
                bars = ax.barh(
                    flag_data['flag_type'],
                    flag_data['count'],
                    color=colors[:len(flag_data)],
                    edgecolor='white'
                )
                ax.set_xlabel("Number of Flags", fontsize=11)
                ax.set_title(
                    "Anomaly Flags by Detection Type",
                    fontsize=13, fontweight='bold', pad=12
                )
                ax.xaxis.set_major_formatter(
                    mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
                )
                for bar, val in zip(bars, flag_data['count']):
                    ax.text(
                        bar.get_width() + 10,
                        bar.get_y() + bar.get_height()/2,
                        f"{val:,}", va='center', fontsize=10
                    )
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        except Exception as e:
            st.error(f"Chart error: {e}")

    with col_right:
        st.subheader("Detection Summary")
        if not flag_data.empty:
            st.dataframe(
                flag_data.rename(columns={
                    'flag_type': 'Flag Type',
                    'count': 'Count',
                    'avg_if_score': 'Avg IF Score',
                    'avg_zscore': 'Avg Z-Score'
                }),
                use_container_width=True,
                hide_index=True
            )

    st.markdown("---")

    # Actionability tracker [Peer 2]
    st.subheader("📊 Actionability Rate Tracker")
    if actionability:
        total_flags = actionability[0] or 0
        actionable = actionability[1] or 0
        not_actionable = actionability[2] or 0
        pending = actionability[3] or 0
        rate = actionability[4]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Flags", f"{total_flags:,}")
        with col2:
            st.metric("✅ Actionable", f"{actionable:,}",
                      delta=None)
        with col3:
            st.metric("❌ Not Actionable", f"{not_actionable:,}")
        with col4:
            st.metric("⏳ Pending Review", f"{pending:,}")

        if total_flags > 0:
            progress_pct = (total_flags - pending) / total_flags
            st.progress(progress_pct,
                        text=f"Review progress: "
                             f"{total_flags - pending:,} of "
                             f"{total_flags:,} flags reviewed "
                             f"({100*progress_pct:.1f}%)")

        target_note = (
            "✅ Target achieved!" if rate and float(rate) >= 85
            else f"🎯 Target: 85% (Herreros-Martínez et al., 2025)"
        )
        st.info(f"**Actionability Rate:** "
                f"{rate if rate else 'Pending'} | {target_note}")

# =============================================================
# PAGE: FLAGGED CONTRACTS
# =============================================================

def render_flagged_contracts(engine, agency_filter,
                              flag_type_filter, min_if_score,
                              username, user_role):
    """Render the flagged contracts review page."""

    # ── DISCLAIMER BANNER [Peers 1, 2, 3] ──
    st.markdown("""
    <div class="disclaimer-banner">
        ⚠️ <strong>IMPORTANT NOTICE:</strong> Flags shown below indicate
        statistically unusual patterns that require human review.
        They do <strong>NOT</strong> constitute evidence of fraud or
        misconduct. All flagged contracts must be reviewed by a qualified
        auditor before any investigative action is taken.
    </div>
    """, unsafe_allow_html=True)

    # Load data
    df = load_flagged_contracts(
        engine, agency_filter, flag_type_filter, min_if_score
    )

    if df.empty:
        st.warning(
            "No flagged contracts match the current filters. "
            "Try adjusting your filter criteria."
        )
        return

    st.markdown(f"**Showing {len(df):,} flagged contracts** "
                f"(filtered from 2,397 total flags)")

    # Column headers
    header_cols = st.columns([2, 2, 1, 1, 1, 1, 1])
    headers = ["Vendor", "Agency", "Amount",
               "IF Score", "Z-Score", "Flag Type", "Action"]
    for col, header in zip(header_cols, headers):
        col.markdown(f"**{header}**")
    st.markdown("---")

    # Render each flagged contract
    for _, row in df.head(20).iterrows():
        with st.container():
            # Flag header
            flag_col1, flag_col2 = st.columns([6, 1])

            with flag_col1:
                vendor_name = str(row['recipient_name'])[:50]
                agency = str(row['awarding_agency'])[:40]
                amount = format_currency(row['total_dollars_obligated'])
                flag_type = format_flag_type(row['flag_type'])

                # Scores side by side [Peer 1]
                if_score = row['isolation_forest_score']
                z_score = row['zscore_deviation']

                st.markdown(f"""
                **{vendor_name}**
                | Agency: {agency}
                | Amount: **{amount}**
                | {flag_type}
                """)

                score_col1, score_col2 = st.columns(2)
                with score_col1:
                    if if_score is not None:
                        st.markdown(
                            f"🤖 **Isolation Forest:** "
                            f"`{if_score:.4f}` "
                            f"{format_if_score(if_score).split()[0]}"
                        )
                with score_col2:
                    if z_score is not None:
                        st.markdown(
                            f"📊 **Z-Score:** "
                            f"`{z_score:.2f}σ` "
                            f"{'🔴' if abs(z_score) > 3 else '🟡'}"
                        )

                # Explainable risk factors [Peer 3]
                if row['risk_factor_1']:
                    with st.expander("🔍 View Explainable Risk Factors"):
                        if row['risk_factor_1']:
                            st.markdown(f"① {row['risk_factor_1']}")
                        if row['risk_factor_2']:
                            st.markdown(f"② {row['risk_factor_2']}")
                        if pd.notna(row.get('risk_factor_explanation')):
                            st.caption(
                                f"Summary: {row['risk_factor_explanation'][:200]}"
                            )

            with flag_col2:
                # Auditor feedback mechanism [Peer 2]
                current = row['analyst_feedback']
                flag_id = int(row['flag_id'])

                if current == 'pending':
                    if st.button("👍", key=f"yes_{flag_id}",
                                 help="Mark as Actionable"):
                        if submit_feedback(
                            engine, flag_id, 'actionable', username
                        ):
                            st.success("Marked actionable!")
                            st.rerun()
                    if st.button("👎", key=f"no_{flag_id}",
                                 help="Mark as Not Actionable"):
                        if submit_feedback(
                            engine, flag_id, 'not_actionable', username
                        ):
                            st.info("Marked not actionable")
                            st.rerun()
                elif current == 'actionable':
                    st.markdown("✅ **Actionable**")
                elif current == 'not_actionable':
                    st.markdown("❌ **Not Actionable**")

            st.markdown("---")

# =============================================================
# PAGE: AGENCY ANALYSIS
# =============================================================

def render_agency_analysis(engine):
    """Render the agency analysis page."""
    st.subheader("Agency Spending Analysis")

    try:
        agency_df = load_agency_summary(engine)

        if agency_df.empty:
            st.warning("No agency data available.")
            return

        # Top agencies by flag count
        top_agencies = agency_df.head(10)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Top 10 Agencies by Contract Count**")
            fig, ax = plt.subplots(figsize=(8, 5))
            names = [
                str(n)[:35] + "..." if len(str(n)) > 35 else str(n)
                for n in top_agencies['awarding_agency']
            ]
            ax.barh(
                names,
                top_agencies['total_contracts'],
                color='#1F4E79', edgecolor='white'
            )
            ax.invert_yaxis()
            ax.set_xlabel("Number of Contracts")
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
            )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("**Top 10 Agencies by Total Spend**")
            fig, ax = plt.subplots(figsize=(8, 5))
            spend_billions = top_agencies['total_obligated'] / 1e9
            ax.barh(
                names,
                spend_billions,
                color='#2E75B6', edgecolor='white'
            )
            ax.invert_yaxis()
            ax.set_xlabel("Total Spend (USD Billions)")
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"${x:.1f}B")
            )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("---")
        st.markdown("**Agency Summary Table**")
        display_df = agency_df.copy()
        display_df['total_obligated'] = display_df[
            'total_obligated'
        ].apply(lambda x: format_currency(x) if x else "N/A")
        display_df['avg_contract_value'] = display_df[
            'avg_contract_value'
        ].apply(lambda x: format_currency(x) if x else "N/A")
        display_df['flag_rate_pct'] = display_df[
            'flag_rate_pct'
        ].apply(lambda x: f"{x}%" if x else "0%")

        st.dataframe(
            display_df.rename(columns={
                'awarding_agency': 'Agency',
                'total_contracts': 'Contracts',
                'total_flags': 'Flags',
                'flag_rate_pct': 'Flag Rate',
                'avg_contract_value': 'Avg Value',
                'total_obligated': 'Total Spend'
            }).head(15),
            use_container_width=True,
            hide_index=True
        )

    except Exception as e:
        st.error(f"Error loading agency data: {e}")

# =============================================================
# PAGE: ADMIN SETTINGS [Admin only — RBAC]
# =============================================================

def render_admin_settings(engine, username):
    """
    Admin-only settings page.
    Only accessible to admin role users.
    Restricted from analysts per RBAC design.
    [OWASP A01:2025 — Broken Access Control mitigation]
    """
    st.subheader("⚙️ Admin Settings")
    st.warning(
        "**Admin Zone:** Changes made here affect the anomaly "
        "detection configuration. All changes are logged."
    )

    # Contamination threshold management
    st.markdown("#### Isolation Forest Configuration")
    current_contamination = 0.05

    new_contamination = st.slider(
        "Contamination Threshold",
        min_value=0.01, max_value=0.15,
        value=current_contamination, step=0.01,
        help="Expected proportion of anomalies in dataset. "
             "Start: 0.05, Tune in 0.01 increments."
    )

    reason = st.text_area(
        "Reason for change (required)",
        placeholder="Document the reason for changing this parameter...",
        help="All configuration changes require documented justification."
    )

    if st.button("Apply Configuration Change", type="primary"):
        if not reason.strip():
            st.error("Please provide a reason for the change.")
        else:
            # Log configuration change to audit trail
            with engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO audit_log (
                        user_role,
                        action_attempted,
                        action_result,
                        resource_accessed
                    ) VALUES (
                        :role,
                        :action,
                        'success',
                        'contamination_threshold'
                    )
                """), {
                    'role': username,
                    'action': (
                        f'changed_contamination_from_'
                        f'{current_contamination}_to_{new_contamination}'
                        f'_reason:{reason[:100]}'
                    )
                })
            st.success(
                f"Configuration change logged. "
                f"New contamination: {new_contamination}. "
                f"Re-run ML pipeline to apply."
            )

    st.markdown("---")
    st.markdown("#### Audit Log — Recent Activity")
    with engine.connect() as conn:
        audit_df = pd.read_sql(text("""
            SELECT
                logged_at,
                user_role,
                action_attempted,
                action_result
            FROM audit_log
            ORDER BY logged_at DESC
            LIMIT 20
        """), conn)

    if not audit_df.empty:
        st.dataframe(audit_df, use_container_width=True, hide_index=True)
    else:
        st.info("No audit log entries yet.")

# =============================================================
# ENTRY POINT
# =============================================================

def main():
    """Main application entry point."""
    check_authentication()

    if not st.session_state.authenticated:
        login_page()
    else:
        main_dashboard()

if __name__ == "__main__":
    main()
