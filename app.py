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
    'housing_risk_flag': 'boolean',
    'impairment_risk_flag': 'boolean',
    'mmh_risk_flag': 'boolean'
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
    overall_risk = get_risk_level(housing_risk, impairment_risk, mmh_risk)
    risk_color = get_risk_color(overall_risk)
    
    title_style.textColor = risk_color
    
    story = []
    
    story.append(Paragraph(f"CLIENT BACKGROUND REPORT", title_style))
    story.append(Paragraph("Plunket AI Model Analysis", subtitle_style))
    
    story.append(HRFlowable(
        width="100%",
        thickness=2,
        color=risk_color,
        spaceBefore=4,
        spaceAfter=12
    ))
    
    story.append(Paragraph("CASE DETAILS", section_header_style))
    
    client_id = safe_str(data_row.get('koo_clientid', ''))
    contact_id = safe_str(data_row.get('koo_contactid', ''))
    created_on = data_row.get('createdon', '')
    if pd.notna(created_on):
        if hasattr(created_on, 'strftime'):
            date_str = created_on.strftime('%Y-%m-%d')
        else:
            date_str = str(created_on)[:10]
    else:
        date_str = "Not available"
    
    case_data = [
        ['Client ID:', client_id, 'Case Manager:', f"Nurse {contact_id}"],
        ['Assessment Date:', date_str, 'Generated:', datetime.now().strftime('%Y-%m-%d %H:%M')]
    ]
    
    case_table = Table(case_data, colWidths=[2.5*cm, 5*cm, 3*cm, 5*cm])
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
    
    story.append(Paragraph("RISK ASSESSMENT", section_header_style))
    
    risk_style = risk_high_style if overall_risk == "HIGH RISK" else risk_low_style
    story.append(Paragraph(f"<b>Overall Risk Level: {overall_risk}</b>", risk_style))
    story.append(Spacer(1, 8))
    
    risk_data = [
        ['Risk Category', 'Status'],
        ['Housing Risk', format_risk_flag(housing_risk)],
        ['Impairment Risk', format_risk_flag(impairment_risk)],
        ['Mental/Maternal Health Risk', format_risk_flag(mmh_risk)]
    ]
    
    risk_table = Table(risk_data, colWidths=[8*cm, 6*cm])
    risk_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F5F5F5")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#333333")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(risk_table)
    story.append(Spacer(1, 12))
    
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#DDDDDD"), spaceAfter=8))
    
    story.append(Paragraph("DISCUSSION TOPICS", section_header_style))
    
    story.append(Paragraph("Housing Topics:", subsection_style))
    housing_topics = safe_str(data_row.get('topic_tags_house', ''))
    story.append(Paragraph(f"&bull; {housing_topics}", body_style))
    
    story.append(Paragraph("Impairment Topics:", subsection_style))
    impairment_topics = safe_str(data_row.get('topic_tags_impairment', ''))
    story.append(Paragraph(f"&bull; {impairment_topics}", body_style))
    
    story.append(Paragraph("Mental/Maternal Health Topics:", subsection_style))
    mmh_topics = safe_str(data_row.get('topic_tags_mmh', ''))
    story.append(Paragraph(f"&bull; {mmh_topics}", body_style))
    
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#DDDDDD"), spaceAfter=8))
    
    story.append(Paragraph("DETAILED SUMMARIES", section_header_style))
    
    story.append(Paragraph("Housing Situation", subsection_style))
    housing_summary = safe_str(data_row.get('housing_summary', ''))
    story.append(Paragraph(housing_summary, body_style))
    
    story.append(Paragraph("Impairment Status", subsection_style))
    impairment_summary = safe_str(data_row.get('impairments_summary', ''))
    story.append(Paragraph(impairment_summary, body_style))
    
    story.append(Paragraph("Mental/Maternal Health", subsection_style))
    mmh_summary = safe_str(data_row.get('mmh_summary', ''))
    story.append(Paragraph(mmh_summary, body_style))
    
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#DDDDDD"), spaceAfter=8))
    
    story.append(Paragraph("Generated by Plunket AI Model.", footer_style))
    story.append(Paragraph("End of Report", footer_style))
    
    doc.build(story)
    buffer.seek(0)
    return buffer


def render_report_preview(data_row: pd.Series):
    """Render a preview of the report in Streamlit."""
    housing_risk = data_row.get('housing_risk_flag', False)
    impairment_risk = data_row.get('impairment_risk_flag', False)
    mmh_risk = data_row.get('mmh_risk_flag', False)
    overall_risk = get_risk_level(housing_risk, impairment_risk, mmh_risk)
    
    if overall_risk == "HIGH RISK":
        risk_color = "#C41E3A"
        bg_color = "#FFF5F5"
    elif overall_risk == "MODERATE RISK":
        risk_color = "#FF8C00"
        bg_color = "#FFFAF0"
    else:
        risk_color = "#228B22"
        bg_color = "#F0FFF0"
    
    st.markdown(f"""
    <div style="background-color: {bg_color}; padding: 20px; border-radius: 8px; border: 1px solid #ddd;">
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <div style="width: 30px; height: 30px; background-color: {risk_color}; border-radius: 50%; margin-right: 15px;"></div>
            <h2 style="color: {risk_color}; margin: 0;">Client Background Report ({overall_risk})</h2>
        </div>
        <hr style="border: 1px solid {risk_color}; margin: 10px 0;">
    """, unsafe_allow_html=True)
    
    client_id = safe_str(data_row.get('koo_clientid', ''))
    contact_id = safe_str(data_row.get('koo_contactid', ''))
    created_on = data_row.get('createdon', '')
    if pd.notna(created_on):
        if hasattr(created_on, 'strftime'):
            date_str = created_on.strftime('%Y-%m-%d')
        else:
            date_str = str(created_on)[:10]
    else:
        date_str = "Not available"
    
    st.markdown(f"""
        <p><strong>Client ID:</strong> {client_id} | <strong>Contact ID:</strong> {contact_id} | <strong>Date:</strong> {date_str}</p>
        <h3 style="margin-top: 20px;">SUMMARY</h3>
    """, unsafe_allow_html=True)
    
    if housing_risk:
        st.markdown(f"""
        <div style="background-color: #FFF0F0; padding: 10px; border-left: 4px solid #C41E3A; margin: 10px 0;">
            <span style="color: #C41E3A;"><strong>Housing Risk: HIGH RISK</strong></span>
        </div>
        """, unsafe_allow_html=True)
    
    if impairment_risk:
        st.markdown(f"""
        <div style="background-color: #FFF0F0; padding: 10px; border-left: 4px solid #C41E3A; margin: 10px 0;">
            <span style="color: #C41E3A;"><strong>Impairment Risk: HIGH RISK</strong></span>
        </div>
        """, unsafe_allow_html=True)
    
    if mmh_risk:
        st.markdown(f"""
        <div style="background-color: #FFF0F0; padding: 10px; border-left: 4px solid #C41E3A; margin: 10px 0;">
            <span style="color: #C41E3A;"><strong>Mental/Maternal Health Risk: HIGH RISK</strong></span>
        </div>
        """, unsafe_allow_html=True)
    
    housing_summary = safe_str(data_row.get('housing_summary', ''))
    st.markdown(f"""
        <p><strong>Housing Situation:</strong> {housing_summary}</p>
    """, unsafe_allow_html=True)
    
    st.markdown("<h3>MMH</h3>", unsafe_allow_html=True)
    mmh_summary = safe_str(data_row.get('mmh_summary', ''))
    st.markdown(f"""
        <p>{mmh_summary}</p>
    """, unsafe_allow_html=True)
    
    st.markdown("<h3>Impairment</h3>", unsafe_allow_html=True)
    impairment_summary = safe_str(data_row.get('impairments_summary', ''))
    st.markdown(f"""
        <p>{impairment_summary}</p>
        <p style="color: #999; text-align: right; margin-top: 30px;">Generated by Plunket AI Model.</p>
    </div>
    """, unsafe_allow_html=True)


st.set_page_config(
    page_title="Client Report Exporter",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>PDF</text></svg>",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
    }
    .pdf-icon {
        width: 40px;
        height: 40px;
        background-color: #C41E3A;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 4px;
        margin-right: 15px;
        font-weight: bold;
        font-size: 12px;
    }
    .stButton > button {
        background-color: #2563EB;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 6px;
        font-weight: 500;
    }
    .stButton > button:hover {
        background-color: #1D4ED8;
    }
    .download-btn > button {
        background-color: #C41E3A !important;
    }
    .download-btn > button:hover {
        background-color: #A01830 !important;
    }
    .input-card {
        background-color: white;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #E5E7EB;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <div class="pdf-icon">PDF</div>
    <div>
        <h1 style="margin: 0; font-size: 28px;">Client Report Exporter</h1>
        <p style="color: #666; margin: 0;">Select client details to generate and review the background report.</p>
    </div>
</div>
<hr style="margin: 20px 0;">
""", unsafe_allow_html=True)

user_token = st.context.headers.get('X-Forwarded-Access-Token')

st.markdown('<div class="input-card">', unsafe_allow_html=True)
st.subheader("Report Generation Inputs")

col1, col2, col3 = st.columns(3)

with col1:
    client_id = st.text_input("Client ID", placeholder="Enter client ID", key="client_id")

with col2:
    contact_id = st.text_input("Contact ID", placeholder="Enter contact ID", key="contact_id")

with col3:
    assessment_date = st.date_input("Report Date", value=date.today(), key="assessment_date")

st.markdown('</div>', unsafe_allow_html=True)

col_btn, col_spacer = st.columns([1, 3])
with col_btn:
    preview_clicked = st.button("Preview Report", type="primary", use_container_width=True)

if 'report_data' not in st.session_state:
    st.session_state.report_data = None

if preview_clicked:
    if not client_id or not contact_id:
        st.error("Please enter both Client ID and Contact ID.")
    else:
        with st.spinner("Loading client data..."):
            df = load_client_data(client_id, contact_id, assessment_date, user_token)
            
            if df.empty:
                st.warning(f"No data found for Client ID: {client_id}, Contact ID: {contact_id}, Date: {assessment_date}")
                st.session_state.report_data = None
            else:
                st.session_state.report_data = df.iloc[0]
                st.success(f"Data loaded successfully! Found {len(df)} record(s).")

if st.session_state.report_data is not None:
    data_row = st.session_state.report_data
    
    st.markdown("---")
    
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.subheader(f"Report Preview: Client #{safe_str(data_row.get('koo_clientid', ''))}")
    with header_col2:
        pdf_buffer = generate_pdf(data_row)
        st.download_button(
            label="Download PDF",
            data=pdf_buffer,
            file_name=f"client_report_{safe_str(data_row.get('koo_clientid', 'unknown'))}_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )
    
    st.markdown("""
    <div style="background-color: #F9FAFB; padding: 10px; border-radius: 8px; text-align: center; margin-bottom: 10px;">
        <span style="color: #9CA3AF;">Simulated PDF Viewer</span>
    </div>
    """, unsafe_allow_html=True)
    
    render_report_preview(data_row)
    
    with st.expander("View Raw Data"):
        st.dataframe(pd.DataFrame([data_row]), use_container_width=True)
