import os
from databricks import sql
from databricks.sdk.core import Config
import streamlit as st
import pandas as pd
from datetime import datetime, date
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

# Ensure environment variable is set correctly
assert os.getenv('DATABRICKS_WAREHOUSE_ID'), "DATABRICKS_WAREHOUSE_ID must be set in app.yaml."

# Databricks config
cfg = Config()

# Schema definition for the client data table
SCHEMA = {
    'koo_clientid': 'string',
    'koo_contactid': 'string',
    'client_name': 'string',
    'nurse_name': 'string',
    'createdon': 'date',
    'response_house': 'string',
    'response_impa': 'string',
    'response_mmh': 'string',
    'housing_summary': 'string',
    'impairments_summary': 'string',
    'mmh_summary': 'string',
    'topic_tags_house': 'string',
    'topic_tags_impairment': 'string',
    'topic_tags_mmh': 'string',
    'housing_risk_flag': 'string',
    'impairment_risk_flag': 'string',
    'mmh_risk_flag': 'string'
}

# Table configuration
TABLE_NAME = "dev_structured.analytics.all_measures_with_ai"


def sql_query_with_service_principal(query: str) -> pd.DataFrame:
    """Execute a SQL query and return the result as a pandas DataFrame."""
    with sql.connect(
        server_hostname=cfg.host,
        http_path=f"/sql/1.0/warehouses/{cfg.warehouse_id}",
        credentials_provider=lambda: cfg.authenticate
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall_arrow().to_pandas()


def sql_query_with_user_token(query: str, user_token: str) -> pd.DataFrame:
    """Execute a SQL query and return the result as a pandas DataFrame."""
    with sql.connect(
        server_hostname=cfg.host,
        http_path=f"/sql/1.0/warehouses/{cfg.warehouse_id}",
        access_token=user_token
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall_arrow().to_pandas()


def search_client_data(client_name: str = None, nurse_name: str = None, assessment_date: date = None, user_token: str = None) -> pd.DataFrame:
    """
    Search client data from Databricks table with partial matching.
    At least one search parameter must be provided.
    Returns: pandas DataFrame with search results or empty DataFrame if not found.
    """
    conditions = []
    
    if client_name and client_name.strip():
        escaped_name = client_name.strip().replace("'", "''")
        conditions.append(f"LOWER(client_name) LIKE LOWER('%{escaped_name}%')")
    
    if nurse_name and nurse_name.strip():
        escaped_name = nurse_name.strip().replace("'", "''")
        conditions.append(f"LOWER(nurse_name) LIKE LOWER('%{escaped_name}%')")
    
    if assessment_date:
        date_str = assessment_date.strftime('%Y-%m-%d')
        conditions.append(f"DATE(createdon) = '{date_str}'")
    
    if not conditions:
        return pd.DataFrame()
    
    where_clause = " AND ".join(conditions)
    
    query = f"""
    SELECT 
        koo_clientid,
        koo_contactid,
        client_name,
        nurse_name,
        createdon,
        response_house,
        response_impa,
        response_mmh,
        housing_summary,
        impairments_summary,
        mmh_summary,
        topic_tags_house,
        topic_tags_impairment,
        topic_tags_mmh,
        housing_risk_flag,
        impairment_risk_flag,
        mmh_risk_flag
    FROM {TABLE_NAME}
    WHERE {where_clause}
    ORDER BY createdon DESC
    """
    
    try:
        if user_token:
            df = sql_query_with_user_token(query, user_token)
        else:
            df = sql_query_with_service_principal(query)
        return df
    except Exception as e:
        st.error(f"Error searching data: {str(e)}")
        return pd.DataFrame()


def load_client_data(client_id: str, contact_id: str, assessment_date: date, user_token: str = None) -> pd.DataFrame:
    """
    Load and filter client data from Databricks table.
    Returns: pandas DataFrame with filtered results or empty DataFrame if not found.
    """
    date_str = assessment_date.strftime('%Y-%m-%d')
    
    query = f"""
    SELECT 
        koo_clientid,
        koo_contactid,
        client_name,
        nurse_name,
        createdon,
        response_house,
        response_impa,
        response_mmh,
        housing_summary,
        impairments_summary,
        mmh_summary,
        topic_tags_house,
        topic_tags_impairment,
        topic_tags_mmh,
        housing_risk_flag,
        impairment_risk_flag,
        mmh_risk_flag
    FROM {TABLE_NAME}
    WHERE koo_clientid = '{client_id}'
      AND koo_contactid = '{contact_id}'
      AND DATE(createdon) = '{date_str}'
    """
    
    try:
        if user_token:
            df = sql_query_with_user_token(query, user_token)
        else:
            df = sql_query_with_service_principal(query)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()


def normalize_flag(flag) -> bool:
    """
    Normalize a risk flag value to a boolean.
    Handles bool, None/NaN, int, float, and string values.
    """
    if isinstance(flag, bool):
        return flag
    
    if flag is None or (isinstance(flag, float) and pd.isna(flag)):
        return False
    
    if isinstance(flag, (int, float)):
        return bool(flag)
    
    if isinstance(flag, str):
        v = flag.strip().lower()
        if v in {"true", "t", "1", "yes", "y"}:
            return True
        if v in {"false", "f", "0", "no", "n", ""}:
            return False
    
    return False


def get_risk_level(housing_risk, impairment_risk, mmh_risk) -> str:
    """Determine overall risk level based on individual risk flags."""
    flags = [
        normalize_flag(housing_risk),
        normalize_flag(impairment_risk),
        normalize_flag(mmh_risk),
    ]
    risk_count = sum(flags)
    
    if risk_count >= 2:
        return "HIGH RISK"
    elif risk_count == 1:
        return "MODERATE RISK"
    else:
        return "LOW RISK"


def get_risk_color(risk_level: str) -> colors.Color:
    """Get color based on risk level."""
    if risk_level == "HIGH RISK":
        return colors.HexColor("#C41E3A")
    elif risk_level == "MODERATE RISK":
        return colors.HexColor("#FF8C00")
    else:
        return colors.HexColor("#228B22")


def safe_str(value) -> str:
    """Safely convert value to string, handling None/NaN values."""
    if pd.isna(value) or value is None:
        return "Not available"
    return str(value)


def format_risk_flag(flag) -> str:
    """Format risk flag for display."""
    if flag is None or (isinstance(flag, float) and pd.isna(flag)):
        return "Not assessed"
    return "HIGH RISK" if normalize_flag(flag) else "LOW RISK"


def generate_pdf(data_row: pd.Series) -> BytesIO:
    """
    Generate formatted PDF report from data row.
    Returns: PDF file buffer.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1*cm,
        leftMargin=1*cm,
        topMargin=1*cm,
        bottomMargin=1*cm
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor("#C41E3A"),
        spaceAfter=6,
        alignment=TA_LEFT
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor("#666666"),
        spaceAfter=12
    )
    
    section_header_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor("#333333"),
        spaceBefore=16,
        spaceAfter=8,
        borderPadding=4
    )
    
    subsection_style = ParagraphStyle(
        'SubSection',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=colors.HexColor("#444444"),
        spaceBefore=12,
        spaceAfter=6
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor("#333333"),
        spaceAfter=8,
        leading=14
    )
    
    risk_high_style = ParagraphStyle(
        'RiskHigh',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor("#C41E3A"),
        spaceBefore=4,
        spaceAfter=4,
        backColor=colors.HexColor("#FFF0F0"),
        borderPadding=8
    )
    
    risk_low_style = ParagraphStyle(
        'RiskLow',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor("#228B22"),
        spaceBefore=4,
        spaceAfter=4,
        backColor=colors.HexColor("#F0FFF0"),
        borderPadding=8
    )
    
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor("#999999"),
        alignment=TA_RIGHT,
        spaceBefore=20
    )
    
    housing_risk = data_row.get('housing_risk_flag', False)
    impairment_risk = data_row.get('impairment_risk_flag', False)
    mmh_risk = data_row.get('mmh_risk_flag', False)
    
    story = []
    
    story.append(Paragraph("CLIENT BACKGROUND REPORT", title_style))
    story.append(Paragraph("Based on Plunket AI Model Analysis", subtitle_style))
    
    story.append(HRFlowable(
        width="100%",
        thickness=1,
        color=colors.HexColor("#DDDDDD"),
        spaceBefore=4,
        spaceAfter=12
    ))
    
    client_name = safe_str(data_row.get('client_name', ''))
    nurse_name = safe_str(data_row.get('nurse_name', ''))
    created_on = data_row.get('createdon', '')
    if pd.notna(created_on):
        if hasattr(created_on, 'strftime'):
            date_str = created_on.strftime('%Y-%m-%d')
        else:
            date_str = str(created_on)[:10]
    else:
        date_str = "Not available"
    
    case_data = [
        ['Client:', client_name, 'Nurse:', nurse_name],
        ['Date:', date_str, 'Generated:', datetime.now().strftime('%Y-%m-%d %H:%M')]
    ]
    
    case_table = Table(case_data, colWidths=[2.5*cm, 5*cm, 2.5*cm, 5*cm])
    case_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor("#666666")),
        ('TEXTCOLOR', (2, 0), (2, -1), colors.HexColor("#666666")),
        ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor("#333333")),
        ('TEXTCOLOR', (3, 0), (3, -1), colors.HexColor("#333333")),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica-Bold'),
        ('FONTNAME', (3, 0), (3, -1), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(case_table)
    story.append(Spacer(1, 12))
    
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#DDDDDD"), spaceAfter=8))
    
    story.append(Paragraph("DISCUSSION TOPICS", section_header_style))
    
    housing_topics = safe_str(data_row.get('topic_tags_house', ''))
    impairment_topics = safe_str(data_row.get('topic_tags_impairment', ''))
    mmh_topics = safe_str(data_row.get('topic_tags_mmh', ''))
    
    story.append(Paragraph(f"&bull; <b>Housing:</b> {housing_topics}", body_style))
    story.append(Paragraph(f"&bull; <b>Impairment:</b> {impairment_topics}", body_style))
    story.append(Paragraph(f"&bull; <b>Mental/Maternal Health:</b> {mmh_topics}", body_style))
    
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#DDDDDD"), spaceAfter=8))
    
    story.append(Paragraph("SUMMARIES", section_header_style))
    
    story.append(Paragraph("Housing Situation:", subsection_style))
    housing_summary = safe_str(data_row.get('housing_summary', ''))
    story.append(Paragraph(housing_summary, body_style))
    
    story.append(Paragraph("Impairment Status:", subsection_style))
    impairment_summary = safe_str(data_row.get('impairments_summary', ''))
    story.append(Paragraph(impairment_summary, body_style))
    
    story.append(Paragraph("Mental/Maternal Health:", subsection_style))
    mmh_summary = safe_str(data_row.get('mmh_summary', ''))
    story.append(Paragraph(mmh_summary, body_style))
    
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#DDDDDD"), spaceAfter=8))
    
    story.append(Paragraph("RISK FLAGS", section_header_style))
    
    # Use raw risk flag values from database columns
    housing_risk_raw = safe_str(data_row.get('housing_risk_flag', ''))
    impairment_risk_raw = safe_str(data_row.get('impairment_risk_flag', ''))
    mmh_risk_raw = safe_str(data_row.get('mmh_risk_flag', ''))
    
    story.append(Paragraph(f"&bull; <b>Housing:</b> {housing_risk_raw}", body_style))
    story.append(Paragraph(f"&bull; <b>Impairment:</b> {impairment_risk_raw}", body_style))
    story.append(Paragraph(f"&bull; <b>MMH:</b> {mmh_risk_raw}", body_style))
    
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#DDDDDD"), spaceAfter=8))
    
    story.append(Paragraph("Generated by Plunket AI Model.", footer_style))
    
    doc.build(story)
    buffer.seek(0)
    return buffer


def render_report_preview(data_row: pd.Series):
    """Render a preview of the report in Streamlit using native components."""
    client_name = safe_str(data_row.get('client_name', ''))
    nurse_name = safe_str(data_row.get('nurse_name', ''))
    created_on = data_row.get('createdon', '')
    if pd.notna(created_on):
        if hasattr(created_on, 'strftime'):
            date_str = created_on.strftime('%Y-%m-%d')
        else:
            date_str = str(created_on)[:10]
    else:
        date_str = "Not available"
    
    housing_topics = safe_str(data_row.get('topic_tags_house', ''))
    impairment_topics = safe_str(data_row.get('topic_tags_impairment', ''))
    mmh_topics = safe_str(data_row.get('topic_tags_mmh', ''))
    
    housing_summary = safe_str(data_row.get('housing_summary', ''))
    impairment_summary = safe_str(data_row.get('impairments_summary', ''))
    mmh_summary = safe_str(data_row.get('mmh_summary', ''))
    
    # Get raw risk flag values from database columns
    housing_risk_raw = safe_str(data_row.get('housing_risk_flag', ''))
    impairment_risk_raw = safe_str(data_row.get('impairment_risk_flag', ''))
    mmh_risk_raw = safe_str(data_row.get('mmh_risk_flag', ''))
    
    # Use native Streamlit components for reliable rendering
    with st.container():
        st.subheader("Client Background Report")
        st.caption("Based on Plunket AI Model Analysis")
        
        st.markdown(f"**Client:** {client_name} | **Nurse:** {nurse_name} | **Date:** {date_str}")
        
        st.divider()
        
        st.markdown("### DISCUSSION TOPICS")
        st.markdown(f"- **Housing:** {housing_topics}")
        st.markdown(f"- **Impairment:** {impairment_topics}")
        st.markdown(f"- **Mental/Maternal Health:** {mmh_topics}")
        
        st.divider()
        
        st.markdown("### SUMMARIES")
        
        st.markdown(f"**Housing Situation:**")
        st.markdown(housing_summary)
        
        st.markdown(f"**Impairment Status:**")
        st.markdown(impairment_summary)
        
        st.markdown(f"**Mental/Maternal Health:**")
        st.markdown(mmh_summary)
        
        st.divider()
        
        st.markdown("### RISK FLAGS")
        st.markdown(f"- **Housing:** {housing_risk_raw}")
        st.markdown(f"- **Impairment:** {impairment_risk_raw}")
        st.markdown(f"- **MMH:** {mmh_risk_raw}")


st.set_page_config(
    page_title="Healthcare Dashboard",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>PDF</text></svg>",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        background-color: #0D1117;
    }
    .main-header {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }
    .pdf-icon {
        width: 40px;
        height: 40px;
        background-color: #E91E63;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 4px;
        margin-right: 15px;
        font-weight: bold;
        font-size: 12px;
    }
    .section-header {
        display: flex;
        align-items: center;
        margin-bottom: 16px;
        color: #E6EDF3;
    }
    .section-header::before {
        content: '';
        width: 4px;
        height: 24px;
        background-color: #E91E63;
        margin-right: 12px;
        border-radius: 2px;
    }
    .stButton > button {
        background-color: #E91E63 !important;
        color: white !important;
        border: none;
        padding: 10px 20px;
        border-radius: 6px;
        font-weight: 500;
    }
    .stButton > button:hover {
        background-color: #C2185B !important;
    }
    .stButton > button:disabled {
        background-color: #30363D !important;
        color: #8B949E !important;
    }
    .card {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .stTextInput > div > div > input {
        background-color: #0D1117;
        border: 1px solid #30363D;
        color: #E6EDF3;
    }
    .stDateInput > div > div > input {
        background-color: #0D1117;
        border: 1px solid #30363D;
        color: #E6EDF3;
    }
    h1, h2, h3, p, label {
        color: #E6EDF3 !important;
    }
    .stDataFrame {
        background-color: #161B22;
    }
    .empty-state {
        text-align: center;
        padding: 40px;
        color: #8B949E;
    }
    /* Hide 'Press Enter to apply' hint */
    .stTextInput > div > div > div[data-testid="InputInstructions"],
    .stDateInput > div > div > div[data-testid="InputInstructions"] {
        display: none !important;
    }
    /* Preview button - blue */
    .preview-btn button {
        background-color: #2563EB !important;
        color: white !important;
        max-width: 150px !important;
    }
    .preview-btn button:hover {
        background-color: #1D4ED8 !important;
    }
    /* Download button - light blue */
    .download-btn button {
        background-color: #60A5FA !important;
        color: white !important;
        max-width: 150px !important;
    }
    .download-btn button:hover {
        background-color: #3B82F6 !important;
    }
    /* Radio button alignment with table rows */
    .results-radio-column [role="radiogroup"] {
        display: flex;
        flex-direction: column;
        row-gap: 14px;
        align-items: flex-end;
        padding-right: 0px;
    }
    .results-radio-column [role="radiogroup"] label {
        margin-bottom: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <div class="pdf-icon">PDF</div>
    <div>
        <h1 style="margin: 0; font-size: 28px; color: #E6EDF3;">Healthcare Dashboard</h1>
        <p style="color: #8B949E; margin: 0;">Search for client records and generate reports.</p>
    </div>
</div>
""", unsafe_allow_html=True)

user_token = st.context.headers.get('X-Forwarded-Access-Token')

if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'selected_row_index' not in st.session_state:
    st.session_state.selected_row_index = None
if 'report_data' not in st.session_state:
    st.session_state.report_data = None

st.markdown('<div class="section-header">Search Criteria</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

with col1:
    client_name_input = st.text_input("Client Name", placeholder="Enter client name", key="client_name_input")

with col2:
    nurse_name_input = st.text_input("Nurse Name", placeholder="Enter nurse name", key="nurse_name_input")

with col3:
    report_date_input = st.date_input("Report Date", value=None, key="report_date_input")

with col4:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    search_clicked = st.button("Search", type="primary", use_container_width=True, key="search_btn")

if search_clicked:
    if not client_name_input and not nurse_name_input and not report_date_input:
        st.error("Please enter at least one search criteria (Client Name, Nurse Name, or Report Date).")
    else:
        with st.spinner("Searching..."):
            df = search_client_data(
                client_name=client_name_input if client_name_input else None,
                nurse_name=nurse_name_input if nurse_name_input else None,
                assessment_date=report_date_input if report_date_input else None,
                user_token=user_token
            )
            st.session_state.search_results = df
            st.session_state.selected_row_index = None
            st.session_state.report_data = None

st.markdown("---")

st.markdown('<div class="section-header">Search Results</div>', unsafe_allow_html=True)

if st.session_state.search_results is not None:
    df = st.session_state.search_results
    
    if df.empty:
        st.markdown("""
        <div class="empty-state">
            <p>No records found. Try adjusting your search criteria.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        display_df = pd.DataFrame({
            'Client Name': df['client_name'].apply(safe_str),
            'Nurse Name': df['nurse_name'].apply(safe_str),
            'Date': df['createdon'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) and hasattr(x, 'strftime') else str(x)[:10] if pd.notna(x) else 'N/A'),
            'Housing Risk': df['housing_risk_flag'].apply(safe_str),
            'Impairment Risk': df['impairment_risk_flag'].apply(safe_str),
            'MMH Risk': df['mmh_risk_flag'].apply(safe_str)
        })
        
        options = list(range(len(display_df)))
        default_index = st.session_state.selected_row_index if st.session_state.selected_row_index is not None and st.session_state.selected_row_index < len(options) else 0
        
        radio_col, table_col = st.columns([0.3, 9.7])
        
        with radio_col:
            st.markdown('<div class="results-radio-column">', unsafe_allow_html=True)
            st.markdown('<div style="height: 36px;"></div>', unsafe_allow_html=True)
            selected = st.radio(
                "Select",
                options=options,
                format_func=lambda x: "",
                index=default_index,
                key="row_selection",
                label_visibility="collapsed"
            )
            st.session_state.selected_row_index = selected
            st.markdown('</div>', unsafe_allow_html=True)
        
        with table_col:
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                height=min(280, 56 * (len(display_df) + 1))
            )
        
        st.caption(f"Showing {len(display_df)} result(s)")
else:
    st.markdown("""
    <div class="empty-state">
        <p>Enter search criteria above and click Search to find client records.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.markdown('<div class="section-header">Report Actions</div>', unsafe_allow_html=True)

has_selection = st.session_state.selected_row_index is not None and st.session_state.search_results is not None and not st.session_state.search_results.empty

col_preview, col_download, col_spacer = st.columns([1, 1, 8])

with col_preview:
    st.markdown('<div class="preview-btn">', unsafe_allow_html=True)
    preview_clicked = st.button(
        "Preview Report",
        type="secondary",
        disabled=not has_selection,
        key="preview_btn"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col_download:
    st.markdown('<div class="download-btn">', unsafe_allow_html=True)
    if has_selection:
        selected_idx = st.session_state.selected_row_index
        data_row = st.session_state.search_results.iloc[selected_idx]
        pdf_buffer = generate_pdf(data_row)
        client_name_for_file = safe_str(data_row.get('client_name', 'unknown')).replace(' ', '_')
        st.download_button(
            label="Download PDF",
            data=pdf_buffer,
            file_name=f"client_report_{client_name_for_file}_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            type="secondary",
            key="download_btn"
        )
    else:
        st.button(
            "Download PDF",
            type="secondary",
            disabled=True,
            key="download_btn_disabled"
        )
    st.markdown('</div>', unsafe_allow_html=True)

if preview_clicked and has_selection:
    selected_idx = st.session_state.selected_row_index
    st.session_state.report_data = st.session_state.search_results.iloc[selected_idx]

if st.session_state.report_data is not None:
    st.markdown("---")
    
    data_row = st.session_state.report_data
    client_name = safe_str(data_row.get('client_name', ''))
    
    st.markdown(f'<div class="section-header">Report Preview: {client_name}</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #161B22; padding: 10px; border-radius: 8px; text-align: center; margin-bottom: 10px; border: 1px solid #30363D;">
        <span style="color: #8B949E;">Report Preview</span>
    </div>
    """, unsafe_allow_html=True)
    
    render_report_preview(data_row)
