import os
import math
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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Image, Flowable
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.pdfgen import canvas

# Ensure environment variable is set correctly
assert os.getenv('DATABRICKS_WAREHOUSE_ID'), "DATABRICKS_WAREHOUSE_ID must be set in app.yaml."

# Databricks config
cfg = Config()

# Schema definition for the client data table (search table)
SCHEMA_SEARCH = {
    'koo_clientid': 'string',
    'koo_contactid': 'string',
    'client_name': 'string',
    'client_nhi': 'string',
    'createdon': 'date',
    'response_house': 'string',
    'response_impa': 'string',
    'response_mmh': 'string'
}

# Schema definition for the report table
SCHEMA_REPORT = {
    'koo_clientid': 'string',
    'client_name': 'string',
    'client_nhi': 'string',
    'dhb': 'string',
    'ethnicity': 'string',
    'domicile': 'string',
    'gender': 'string',
    'primary_caregiver': 'string',
    'well_child_level_of_need': 'string',
    'topic_tags_house': 'string',
    'topic_tags_impairment': 'string',
    'topic_tags_mmh': 'string',
    'housing_summary': 'string',
    'impairments_summary': 'string',
    'mmh_summary': 'string',
    'housing_risk_flag': 'string',
    'impairment_risk_flag': 'string',
    'mmh_risk_flag': 'string'
}

# Table configuration
TABLE_NAME_SEARCH = "dev_structured.analytics.all_measures"
TABLE_NAME_REPORT = "dev_structured.analytics.all_measures_with_ai"


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


def search_client_data(client_name: str = None, client_nhi: str = None, assessment_date: date = None, user_token: str = None) -> pd.DataFrame:
    """
    Search client data from Databricks table with partial matching.
    At least one search parameter must be provided.
    Returns: pandas DataFrame with search results or empty DataFrame if not found.
    """
    conditions = []
    
    if client_name and client_name.strip():
        escaped_name = client_name.strip().replace("'", "''")
        conditions.append(f"LOWER(client_name) LIKE LOWER('%{escaped_name}%')")
    
    if client_nhi and client_nhi.strip():
        escaped_nhi = client_nhi.strip().replace("'", "''")
        conditions.append(f"LOWER(client_nhi) LIKE LOWER('%{escaped_nhi}%')")
    
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
        client_nhi,
        createdon,
        response_house,
        response_impa,
        response_mmh
    FROM {TABLE_NAME_SEARCH}
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


def load_report_data(client_id: str, user_token: str = None) -> pd.DataFrame:
    """
    Load report data from Databricks table by koo_clientid.
    Returns: pandas DataFrame with report data or empty DataFrame if not found.
    """
    escaped_client_id = client_id.replace("'", "''")
    
    query = f"""
    SELECT 
        koo_clientid,
        client_name,
        client_nhi,
        dhb,
        ethnicity,
        domicile,
        gender,
        primary_caregiver,
        well_child_level_of_need,
        topic_tags_house,
        topic_tags_impairment,
        topic_tags_mmh,
        housing_summary,
        impairments_summary,
        mmh_summary,
        housing_risk_flag,
        impairment_risk_flag,
        mmh_risk_flag
    FROM {TABLE_NAME_REPORT}
    WHERE koo_clientid = '{escaped_client_id}'
    LIMIT 1
    """
    
    try:
        if user_token:
            df = sql_query_with_user_token(query, user_token)
        else:
            df = sql_query_with_service_principal(query)
        return df
    except Exception as e:
        st.error(f"Error loading report data: {str(e)}")
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


class PDFHeader(Flowable):
    """Custom flowable for PDF header with blue background, wave pattern, and logo."""
    
    def __init__(self, width, height, logo_path=None):
        Flowable.__init__(self)
        self.width = width
        self.height = height
        self.logo_path = logo_path
    
    def draw(self):
        c = self.canv
        
        c.setFillColor(colors.HexColor("#0099D8"))
        c.rect(0, 0, self.width, self.height, fill=1, stroke=0)
        
        c.setFillColor(colors.white)
        wave_path = c.beginPath()
        wave_path.moveTo(0, 0)
        
        wave_height = self.height * 0.25
        num_points = 100
        for i in range(num_points + 1):
            x = (i / num_points) * self.width
            y = wave_height * (0.5 + 0.5 * math.sin((i / num_points) * math.pi * 2 - math.pi / 2))
            if i == 0:
                wave_path.moveTo(x, y)
            else:
                wave_path.lineTo(x, y)
        
        wave_path.lineTo(self.width, 0)
        wave_path.lineTo(0, 0)
        wave_path.close()
        c.drawPath(wave_path, fill=1, stroke=0)
        
        if self.logo_path and os.path.exists(self.logo_path):
            try:
                logo_height = self.height * 0.6
                logo_width = logo_height * 1.2
                logo_x = self.width - logo_width - 15
                logo_y = (self.height - logo_height) / 2 + 5
                c.drawImage(self.logo_path, logo_x, logo_y, width=logo_width, height=logo_height, preserveAspectRatio=True, mask='auto')
            except Exception:
                pass
    
    def wrap(self, availWidth, availHeight):
        return (self.width, self.height)


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
        fontSize=16,
        textColor=colors.HexColor("#0099D8"),
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
        fontSize=12,
        textColor=colors.HexColor("#0099D8"),
        spaceBefore=14,
        spaceAfter=6,
        borderPadding=4
    )
    
    subsection_style = ParagraphStyle(
        'SubSection',
        parent=styles['Heading3'],
        fontSize=10,
        textColor=colors.HexColor("#444444"),
        spaceBefore=10,
        spaceAfter=4
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
    
    story = []
    
    page_width = A4[0] - 2*cm
    logo_path = os.path.join(os.path.dirname(__file__), 'assets', 'logo.png')
    header = PDFHeader(width=page_width, height=2.5*cm, logo_path=logo_path)
    story.append(header)
    story.append(Spacer(1, 12))
    
    story.append(Paragraph("CLIENT BACKGROUND REPORT", title_style))
    story.append(Paragraph("Based on Plunket AI Model Analysis", subtitle_style))
    
    story.append(HRFlowable(
        width="100%",
        thickness=1,
        color=colors.HexColor("#DEE2E6"),
        spaceBefore=4,
        spaceAfter=12
    ))
    
    # CLIENT INFORMATION section
    story.append(Paragraph("CLIENT INFORMATION", section_header_style))
    
    client_name = safe_str(data_row.get('client_name', ''))
    client_nhi = safe_str(data_row.get('client_nhi', ''))
    dhb = safe_str(data_row.get('dhb', ''))
    ethnicity = safe_str(data_row.get('ethnicity', ''))
    domicile = safe_str(data_row.get('domicile', ''))
    gender = safe_str(data_row.get('gender', ''))
    primary_caregiver = safe_str(data_row.get('primary_caregiver', ''))
    well_child_level_of_need = safe_str(data_row.get('well_child_level_of_need', ''))
    
    client_info_data = [
        ['Client Name:', client_name, 'Client NHI:', client_nhi],
        ['DHB:', dhb, 'Ethnicity:', ethnicity],
        ['Domicile:', domicile, 'Gender:', gender],
        ['Primary Caregiver:', primary_caregiver, 'Well Child Level of Need:', well_child_level_of_need],
        ['Generated:', datetime.now().strftime('%Y-%m-%d %H:%M'), '', '']
    ]
    
    client_info_table = Table(client_info_data, colWidths=[3.2*cm, 5.5*cm, 3.8*cm, 5.5*cm])
    client_info_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor("#666666")),
        ('TEXTCOLOR', (2, 0), (2, -1), colors.HexColor("#666666")),
        ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor("#333333")),
        ('TEXTCOLOR', (3, 0), (3, -1), colors.HexColor("#333333")),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica-Bold'),
        ('FONTNAME', (3, 0), (3, -1), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
    ]))
    story.append(client_info_table)
    story.append(Spacer(1, 12))
    
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#DEE2E6"), spaceAfter=8))
    
    story.append(Paragraph("DISCUSSION TOPICS", section_header_style))
    
    housing_topics = safe_str(data_row.get('topic_tags_house', ''))
    impairment_topics = safe_str(data_row.get('topic_tags_impairment', ''))
    mmh_topics = safe_str(data_row.get('topic_tags_mmh', ''))
    
    story.append(Paragraph(f"&bull; <b>Housing:</b> {housing_topics}", body_style))
    story.append(Paragraph(f"&bull; <b>Impairment:</b> {impairment_topics}", body_style))
    story.append(Paragraph(f"&bull; <b>Mental/Maternal Health:</b> {mmh_topics}", body_style))
    
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#DEE2E6"), spaceAfter=8))
    
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
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#DEE2E6"), spaceAfter=8))
    
    story.append(Paragraph("RISK FLAGS", section_header_style))
    
    # Use raw risk flag values from database columns
    housing_risk_raw = safe_str(data_row.get('housing_risk_flag', ''))
    impairment_risk_raw = safe_str(data_row.get('impairment_risk_flag', ''))
    mmh_risk_raw = safe_str(data_row.get('mmh_risk_flag', ''))
    
    story.append(Paragraph(f"&bull; <b>Housing:</b> {housing_risk_raw}", body_style))
    story.append(Paragraph(f"&bull; <b>Impairment:</b> {impairment_risk_raw}", body_style))
    story.append(Paragraph(f"&bull; <b>MMH:</b> {mmh_risk_raw}", body_style))
    
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#DEE2E6"), spaceAfter=8))
    
    story.append(Paragraph("Generated by Plunket AI Model.", footer_style))
    
    doc.build(story)
    buffer.seek(0)
    return buffer


def render_report_preview(data_row: pd.Series):
    """Render a preview of the report in Streamlit using native components."""
    # Client information
    client_name = safe_str(data_row.get('client_name', ''))
    client_nhi = safe_str(data_row.get('client_nhi', ''))
    dhb = safe_str(data_row.get('dhb', ''))
    ethnicity = safe_str(data_row.get('ethnicity', ''))
    domicile = safe_str(data_row.get('domicile', ''))
    gender = safe_str(data_row.get('gender', ''))
    primary_caregiver = safe_str(data_row.get('primary_caregiver', ''))
    well_child_level_of_need = safe_str(data_row.get('well_child_level_of_need', ''))
    
    # Discussion topics
    housing_topics = safe_str(data_row.get('topic_tags_house', ''))
    impairment_topics = safe_str(data_row.get('topic_tags_impairment', ''))
    mmh_topics = safe_str(data_row.get('topic_tags_mmh', ''))
    
    # Summaries
    housing_summary = safe_str(data_row.get('housing_summary', ''))
    impairment_summary = safe_str(data_row.get('impairments_summary', ''))
    mmh_summary = safe_str(data_row.get('mmh_summary', ''))
    
    # Risk flags
    housing_risk_raw = safe_str(data_row.get('housing_risk_flag', ''))
    impairment_risk_raw = safe_str(data_row.get('impairment_risk_flag', ''))
    mmh_risk_raw = safe_str(data_row.get('mmh_risk_flag', ''))
    
    # Use native Streamlit components for reliable rendering
    with st.container():
        st.subheader("Client Background Report")
        st.caption("Based on Plunket AI Model Analysis")
        
        st.divider()
        
        # CLIENT INFORMATION section
        st.markdown("### CLIENT INFORMATION")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Client Name:** {client_name}")
            st.markdown(f"**DHB:** {dhb}")
            st.markdown(f"**Domicile:** {domicile}")
            st.markdown(f"**Primary Caregiver:** {primary_caregiver}")
        with col2:
            st.markdown(f"**Client NHI:** {client_nhi}")
            st.markdown(f"**Ethnicity:** {ethnicity}")
            st.markdown(f"**Gender:** {gender}")
            st.markdown(f"**Well Child Level of Need:** {well_child_level_of_need}")
        
        st.divider()
        
        # DISCUSSION TOPICS section
        st.markdown("### DISCUSSION TOPICS")
        st.markdown(f"- **Housing:** {housing_topics}")
        st.markdown(f"- **Impairment:** {impairment_topics}")
        st.markdown(f"- **Mental/Maternal Health:** {mmh_topics}")
        
        st.divider()
        
        # SUMMARIES section
        st.markdown("### SUMMARIES")
        
        st.markdown(f"**Housing Situation:**")
        st.markdown(housing_summary)
        
        st.markdown(f"**Impairment Status:**")
        st.markdown(impairment_summary)
        
        st.markdown(f"**Mental/Maternal Health:**")
        st.markdown(mmh_summary)
        
        st.divider()
        
        # RISK FLAGS section
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
        background-color: #FFFFFF;
    }
    .main-header {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }
    .pdf-icon {
        width: 40px;
        height: 40px;
        background-color: #0099D8;
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
        color: #1A1A1A;
    }
    .section-header::before {
        content: '';
        width: 4px;
        height: 24px;
        background-color: #0099D8;
        margin-right: 12px;
        border-radius: 2px;
    }
    .stButton > button {
        background-color: #0099D8 !important;
        color: white !important;
        border: none;
        padding: 10px 20px;
        border-radius: 6px;
        font-weight: 500;
    }
    .stButton > button:hover {
        background-color: #007BB5 !important;
    }
    .stButton > button:disabled {
        background-color: #ADB5BD !important;
        color: #6C757D !important;
    }
    .card {
        background-color: #F8F9FA;
        border: 1px solid #DEE2E6;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 16px;
    }
    /* Text input styling */
    .stTextInput > div > div > input,
    [data-testid="stTextInput"] input {
        background-color: #FFFFFF !important;
        border: 1px solid #DEE2E6 !important;
        color: #1A1A1A !important;
    }
    
    /* Date input styling - comprehensive selectors */
    .stDateInput > div > div > input,
    [data-testid="stDateInput"] input,
    [data-testid="stDateInput"] > div,
    [data-testid="stDateInput"] > div > div {
        background-color: #FFFFFF !important;
        border-color: #DEE2E6 !important;
        color: #1A1A1A !important;
    }
    [data-testid="stDateInput"] {
        background-color: #FFFFFF !important;
    }
    
    h1, h2, h3, p, label {
        color: #1A1A1A !important;
    }
    
    /* Data table/editor styling - comprehensive light theme */
    .stDataFrame,
    [data-testid="stDataFrame"],
    [data-testid="stDataFrameResizable"] {
        background-color: #FFFFFF !important;
    }
    
    /* Table header styling */
    [data-testid="stDataFrame"] thead tr th,
    [data-testid="stDataFrameResizable"] thead tr th,
    .stDataFrame thead tr th {
        background-color: #F8F9FA !important;
        color: #1A1A1A !important;
        border-color: #DEE2E6 !important;
    }
    
    /* Table body rows */
    [data-testid="stDataFrame"] tbody tr,
    [data-testid="stDataFrameResizable"] tbody tr,
    .stDataFrame tbody tr {
        background-color: #FFFFFF !important;
        color: #1A1A1A !important;
    }
    
    /* Table cells */
    [data-testid="stDataFrame"] td,
    [data-testid="stDataFrameResizable"] td,
    .stDataFrame td {
        background-color: #FFFFFF !important;
        color: #1A1A1A !important;
        border-color: #DEE2E6 !important;
    }
    
    /* Selected row styling - light blue highlight */
    [data-testid="stDataFrame"] tbody tr[aria-selected="true"],
    [data-testid="stDataFrame"] tbody tr[data-selected="true"],
    [data-testid="stDataFrameResizable"] tbody tr[aria-selected="true"],
    .stDataFrame tbody tr.row-selected {
        background-color: #E7F1FF !important;
        color: #1A1A1A !important;
    }
    
    /* Glide data grid styling (used by st.data_editor) */
    .dvn-scroller,
    .gdg-style {
        background-color: #FFFFFF !important;
    }
    
    /* Data editor cells */
    [data-testid="stDataFrame"] [role="gridcell"],
    [data-testid="stDataFrameResizable"] [role="gridcell"] {
        background-color: #FFFFFF !important;
        color: #1A1A1A !important;
    }
    
    /* Data editor header cells */
    [data-testid="stDataFrame"] [role="columnheader"],
    [data-testid="stDataFrameResizable"] [role="columnheader"] {
        background-color: #F8F9FA !important;
        color: #1A1A1A !important;
    }
    
    .empty-state {
        text-align: center;
        padding: 40px;
        color: #6C757D;
    }
    
    /* Hide 'Press Enter to apply' hint */
    .stTextInput > div > div > div[data-testid="InputInstructions"],
    .stDateInput > div > div > div[data-testid="InputInstructions"] {
        display: none !important;
    }
    
    /* Preview button - blue with white text */
    .preview-btn button,
    .preview-btn [data-testid="baseButton-secondary"] {
        background-color: #0099D8 !important;
        color: #FFFFFF !important;
        max-width: 150px !important;
    }
    .preview-btn button:hover {
        background-color: #007BB5 !important;
    }
    .preview-btn button:disabled {
        background-color: #B8D4E8 !important;
        color: #FFFFFF !important;
    }
    
    /* Download button - light blue with white text */
    .download-btn button,
    .download-btn [data-testid="baseButton-secondary"],
    .download-btn a {
        background-color: #60A5FA !important;
        color: #FFFFFF !important;
        max-width: 150px !important;
        border-radius: 6px;
        padding: 10px 20px;
        text-decoration: none;
        display: inline-block;
    }
    .download-btn button:hover {
        background-color: #3B82F6 !important;
    }
    .download-btn button:disabled {
        background-color: #B8D4E8 !important;
        color: #FFFFFF !important;
    }
    
    /* Ensure all disabled buttons have white text */
    .stButton > button:disabled,
    button:disabled {
        color: #FFFFFF !important;
    }
    
    /* Checkbox styling in data editor */
    [data-testid="stDataFrame"] input[type="checkbox"],
    .stCheckbox input[type="checkbox"] {
        accent-color: #0099D8;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <div class="pdf-icon">PDF</div>
    <div>
        <h1 style="margin: 0; font-size: 28px; color: #1A1A1A;">Healthcare Dashboard</h1>
        <p style="color: #6C757D; margin: 0;">Search for client records and generate reports.</p>
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
if 'selection_mask' not in st.session_state:
    st.session_state.selection_mask = []

st.markdown('<div class="section-header">Search Criteria</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

with col1:
    client_name_input = st.text_input("Client Name", placeholder="Enter client name", key="client_name_input")

with col2:
    client_nhi_input = st.text_input("Client NHI", placeholder="Enter client NHI", key="client_nhi_input")

with col3:
    report_date_input = st.date_input("Report Date", value=None, key="report_date_input")

with col4:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    search_clicked = st.button("Search", type="primary", use_container_width=True, key="search_btn")

if search_clicked:
    if not client_name_input and not client_nhi_input and not report_date_input:
        st.error("Please enter at least one search criteria (Client Name, Client NHI, or Report Date).")
    else:
        with st.spinner("Searching..."):
            df = search_client_data(
                client_name=client_name_input if client_name_input else None,
                client_nhi=client_nhi_input if client_nhi_input else None,
                assessment_date=report_date_input if report_date_input else None,
                user_token=user_token
            )
            st.session_state.search_results = df
            st.session_state.selected_row_index = None
            st.session_state.report_data = None
            st.session_state.selection_mask = [False] * len(df) if not df.empty else []

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
            'Client NHI': df['client_nhi'].apply(safe_str),
            'Response House': df['response_house'].apply(safe_str),
            'Response Impa': df['response_impa'].apply(safe_str),
            'Response MMH': df['response_mmh'].apply(safe_str),
            'Create Date': df['createdon'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) and hasattr(x, 'strftime') else str(x)[:10] if pd.notna(x) else 'N/A')
        })
        
        n_rows = len(display_df)
        
        if len(st.session_state.selection_mask) != n_rows:
            st.session_state.selection_mask = [False] * n_rows
        
        previous_mask = st.session_state.selection_mask.copy()
        
        editor_df = display_df.copy()
        editor_df.insert(0, 'Select', previous_mask)
        
        edited_df = st.data_editor(
            editor_df,
            column_config={
                "Select": st.column_config.CheckboxColumn(
                    "Select",
                    help="Select a record (only one can be selected)",
                    default=False,
                    width="small"
                )
            },
            disabled=['Client Name', 'Client NHI', 'Response House', 'Response Impa', 'Response MMH', 'Create Date'],
            hide_index=True,
            use_container_width=True,
            height=min(280, 56 * (n_rows + 1)),
            key="results_editor"
        )
        
        new_mask = edited_df['Select'].tolist()
        
        if sum(new_mask) > 1:
            changed_indices = [i for i, (old, new) in enumerate(zip(previous_mask, new_mask)) if old != new]
            if changed_indices:
                selected_idx = changed_indices[-1]
            else:
                selected_idx = next(i for i, v in enumerate(new_mask) if v)
            new_mask = [i == selected_idx for i in range(n_rows)]
            st.session_state.selection_mask = new_mask
            st.session_state.selected_row_index = selected_idx
            st.rerun()
        elif sum(new_mask) == 1:
            selected_idx = next(i for i, v in enumerate(new_mask) if v)
            st.session_state.selection_mask = new_mask
            st.session_state.selected_row_index = selected_idx
        else:
            st.session_state.selection_mask = new_mask
            st.session_state.selected_row_index = None
        
        st.caption(f"Showing {n_rows} result(s)")
else:
    st.markdown("""
    <div class="empty-state">
        <p>Enter search criteria above and click Search to find client records.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.markdown('<div class="section-header">Report Actions</div>', unsafe_allow_html=True)

has_selection = st.session_state.selected_row_index is not None and st.session_state.search_results is not None and not st.session_state.search_results.empty

btn_container, spacer = st.columns([3, 7])

with btn_container:
    col_preview, col_download = st.columns(2)
    
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
            search_row = st.session_state.search_results.iloc[selected_idx]
            client_id = search_row.get('koo_clientid', '')
            report_df = load_report_data(client_id, user_token)
            if not report_df.empty:
                data_row = report_df.iloc[0]
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
                    key="download_btn_no_data"
                )
                st.caption("No report data found")
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
    search_row = st.session_state.search_results.iloc[selected_idx]
    client_id = search_row.get('koo_clientid', '')
    report_df = load_report_data(client_id, user_token)
    if not report_df.empty:
        new_data = report_df.iloc[0]
        if st.session_state.report_data is not None:
            st.session_state.report_data = None
        else:
            st.session_state.report_data = new_data
    else:
        st.error("No report data found for the selected client.")

if st.session_state.report_data is not None:
    st.markdown("---")
    
    data_row = st.session_state.report_data
    client_name = safe_str(data_row.get('client_name', ''))
    
    st.markdown(f'<div class="section-header">Report Preview: {client_name}</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #F8F9FA; padding: 10px; border-radius: 8px; text-align: center; margin-bottom: 10px; border: 1px solid #DEE2E6;">
        <span style="color: #6C757D;">Report Preview</span>
    </div>
    """, unsafe_allow_html=True)
    
    render_report_preview(data_row)
