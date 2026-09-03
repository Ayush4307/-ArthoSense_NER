"""
ArthoSense NER - Digital Medical Report Card Generator
Generates localized, printable digital clinical report cards in PDF and HTML format
for rural healthcare worker field handoffs and patient referrals.
"""

import os
from datetime import datetime
from typing import Dict, Any

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

def generate_html_report(data: Dict[str, Any]) -> str:
    """Generates an elegant, printable HTML report card."""
    patient_uid = data.get("patient_uid", "NER-OA-999")
    name = data.get("name", "Unknown Patient")
    age = data.get("age", 45)
    gender = data.get("gender", "Unspecified")
    district = data.get("district", "Jorhat, Assam")
    bmi = data.get("bmi", 24.5)
    risk_cat = data.get("risk_category", "Low Risk")
    rule_score = data.get("rule_risk_score", 15.0)
    ml_prob = data.get("ml_probability", 18.0)
    rom_angle = data.get("rom_angle", 130.0)
    vib_rms = data.get("vibration_rms", 0.12)
    recommendations = data.get("recommendations", "Maintain healthy activity.")
    date_str = datetime.now().strftime("%d-%b-%Y %H:%M")

    # Theme colors based on severity
    if "High" in str(risk_cat):
        badge_bg = "#ff4d4d"
        badge_border = "#cc0000"
        badge_text = "HIGH RISK"
        pathway_action = "Clinical Evaluation Recommended"
    elif "Moderate" in str(risk_cat):
        badge_bg = "#ffa500"
        badge_border = "#e67300"
        badge_text = "MODERATE RISK"
        pathway_action = "Assessment & Physiotherapy"
    else:
        badge_bg = "#2ecc71"
        badge_border = "#27ae60"
        badge_text = "LOW RISK"
        pathway_action = "Monitor / Routine Annual Check"

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>ArthoSense NER - OA Screening Report Card</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            color: #222;
            background-color: #fff;
        }}
        .report-card {{
            border: 2px solid #2b4c7e;
            border-radius: 12px;
            padding: 24px;
            max-width: 850px;
            margin: 0 auto;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #2b4c7e;
            padding-bottom: 15px;
            margin-bottom: 20px;
        }}
        .header-title {{
            color: #2b4c7e;
            margin: 0;
            font-size: 24px;
            font-weight: bold;
        }}
        .header-sub {{
            color: #555;
            font-size: 13px;
            margin-top: 4px;
        }}
        .badge-box {{
            background-color: {badge_bg};
            color: #fff;
            padding: 12px 22px;
            border-radius: 8px;
            text-align: center;
            font-weight: bold;
            font-size: 18px;
            border: 1px solid {badge_border};
        }}
        .section-title {{
            font-size: 15px;
            font-weight: bold;
            color: #2b4c7e;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 5px;
            margin-top: 15px;
            margin-bottom: 10px;
        }}
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 15px;
        }}
        .data-table td {{
            padding: 8px 12px;
            border: 1px solid #e2e8f0;
            font-size: 14px;
        }}
        .data-table td.label {{
            background-color: #f8fafc;
            font-weight: 600;
            width: 35%;
            color: #334155;
        }}
        .rec-box {{
            background-color: #f0f7ff;
            border-left: 5px solid #2b4c7e;
            padding: 15px;
            border-radius: 4px;
            font-size: 14px;
            line-height: 1.5;
            margin-top: 10px;
        }}
        .footer {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 30px;
            padding-top: 15px;
            border-top: 1px dashed #cbd5e1;
            font-size: 12px;
            color: #64748b;
        }}
        .sign-line {{
            width: 200px;
            border-top: 1px solid #475569;
            text-align: center;
            margin-top: 35px;
            font-size: 12px;
            color: #334155;
        }}
        @media print {{
            body {{ margin: 0; }}
            .report-card {{ border: none; box-shadow: none; padding: 0; }}
        }}
    </style>
</head>
<body>
    <div class="report-card">
        <div class="header">
            <div>
                <h1 class="header-title">ArthoSense NER</h1>
                <div class="header-sub">AI-Assisted Knee Osteoarthritis Screening | SIH26004</div>
                <div class="header-sub">North East Region Rural Healthcare Field Node</div>
            </div>
            <div class="badge-box">
                <div>{badge_text}</div>
                <div style="font-size: 12px; margin-top: 3px; font-weight: normal;">{pathway_action}</div>
            </div>
        </div>

        <div class="section-title">1. Patient Demographic Profile</div>
        <table class="data-table">
            <tr>
                <td class="label">Patient UID</td>
                <td><strong>{patient_uid}</strong></td>
                <td class="label">Screening Date</td>
                <td>{date_str}</td>
            </tr>
            <tr>
                <td class="label">Full Name</td>
                <td>{name}</td>
                <td class="label">Age / Gender</td>
                <td>{age} yrs / {gender}</td>
            </tr>
            <tr>
                <td class="label">District / State</td>
                <td>{district}</td>
                <td class="label">Calculated BMI</td>
                <td><strong>{bmi} kg/m²</strong></td>
            </tr>
        </table>

        <div class="section-title">2. Multimodal Diagnostic Findings</div>
        <table class="data-table">
            <tr>
                <td class="label">Evidence-Based Risk Score</td>
                <td><strong>{rule_score}% ({badge_text})</strong></td>
                <td class="label">Supervised ML Probability</td>
                <td><strong>{ml_prob}% Risk Probability</strong></td>
            </tr>
            <tr>
                <td class="label">Vision Kinematic ROM</td>
                <td><strong>{rom_angle}° Flexion Angle</strong></td>
                <td class="label">Piezo Crepitus RMS</td>
                <td><strong>{vib_rms} RMS Vibration</strong></td>
            </tr>
        </table>

        <div class="section-title">3. Actionable Clinical Pathway & Health Worker Guidance</div>
        <div class="rec-box">
            <strong>Recommended Care Protocol:</strong><br>
            {recommendations}
        </div>

        <div class="footer">
            <div>
                <strong>System Mode:</strong> 100% Local Edge Node (Offline SQLite)<br>
                <strong>Deployment:</strong> Sub-Centre / PHC Screening Outreach
            </div>
            <div>
                <div class="sign-line">Screening Healthcare Worker Signature</div>
            </div>
        </div>
    </div>
</body>
</html>"""
    return html

def generate_pdf_report(data: Dict[str, Any], output_path: str = "screening_report.pdf") -> str:
    """Generates a binary PDF report file using ReportLab."""
    if not REPORTLAB_AVAILABLE:
        html_path = output_path.replace(".pdf", ".html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(generate_html_report(data))
        return html_path

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#2b4c7e")
    )
    sub_style = ParagraphStyle(
        "ReportSub",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#555555")
    )
    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#2b4c7e"),
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#222222")
    )

    story = []

    # Header
    story.append(Paragraph("<b>ArthoSense NER</b>", title_style))
    story.append(Paragraph("AI-Assisted Knee Osteoarthritis Screening | SIH26004 | Offline Rural Diagnostic Card", sub_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#2b4c7e"), spaceBefore=5, spaceAfter=15))

    # Severity Banner Table
    risk_cat = data.get("risk_category", "Low Risk")
    rule_score = data.get("rule_risk_score", 15.0)
    ml_prob = data.get("ml_probability", 18.0)

    if "High" in str(risk_cat):
        banner_color = colors.HexColor("#ff4d4d")
        badge_title = "HIGH RISK - CLINICAL EVALUATION RECOMMENDED"
    elif "Moderate" in str(risk_cat):
        banner_color = colors.HexColor("#ffa500")
        badge_title = "MODERATE RISK - ASSESSMENT & PHYSIOTHERAPY"
    else:
        banner_color = colors.HexColor("#2ecc71")
        badge_title = "LOW RISK - MONITOR & ROUTINE ANNUAL FOLLOW-UP"

    banner_data = [
        [Paragraph(f"<b>SCREENING RESULT: {badge_title}</b>", ParagraphStyle('B', parent=body_style, textColor=colors.white, fontSize=12, alignment=1))],
        [Paragraph(f"Evidence-Based Rule Score: <b>{rule_score}%</b> | Supervised ML Predicted Risk: <b>{ml_prob}%</b>", ParagraphStyle('B2', parent=body_style, textColor=colors.white, fontSize=10, alignment=1))]
    ]
    banner_table = Table(banner_data, colWidths=[540])
    banner_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), banner_color),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(banner_table)
    story.append(Spacer(1, 15))

    # Demographics
    story.append(Paragraph("<b>1. Patient Demographic Profile</b>", h2_style))
    demo_table_data = [
        [Paragraph("<b>Patient UID:</b>", body_style), Paragraph(str(data.get("patient_uid", "NER-001")), body_style),
         Paragraph("<b>Screening Date:</b>", body_style), Paragraph(datetime.now().strftime("%d-%b-%Y"), body_style)],
        [Paragraph("<b>Full Name:</b>", body_style), Paragraph(str(data.get("name", "Unknown")), body_style),
         Paragraph("<b>Age / Gender:</b>", body_style), Paragraph(f"{data.get('age', 45)} yrs / {data.get('gender', 'M')}", body_style)],
        [Paragraph("<b>District / State:</b>", body_style), Paragraph(str(data.get("district", "Assam")), body_style),
         Paragraph("<b>Calculated BMI:</b>", body_style), Paragraph(f"<b>{data.get('bmi', 24.0)} kg/m²</b>", body_style)]
    ]
    demo_table = Table(demo_table_data, colWidths=[110, 160, 110, 160])
    demo_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor("#f8fafc")),
        ('BACKGROUND', (2,0), (2,-1), colors.HexColor("#f8fafc")),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(demo_table)
    story.append(Spacer(1, 15))

    # Multimodal findings
    story.append(Paragraph("<b>2. Multimodal Perception & Sensor Findings</b>", h2_style))
    findings_data = [
        [Paragraph("<b>Vision Kinematic ROM:</b>", body_style), Paragraph(f"<b>{data.get('rom_angle', 130.0)}° Flexion</b>", body_style),
         Paragraph("<b>Acoustic Crepitus RMS:</b>", body_style), Paragraph(f"<b>{data.get('vibration_rms', 0.15)} RMS</b>", body_style)],
        [Paragraph("<b>Tea Garden / Knee Load:</b>", body_style), Paragraph("Yes" if data.get("occupation_loading") else "No", body_style),
         Paragraph("<b>Morning Stiffness:</b>", body_style), Paragraph("Yes (>30 min)" if data.get("stiffness_symptom") else "No", body_style)],
        [Paragraph("<b>Knee Pain (VAS 0-10):</b>", body_style), Paragraph(f"{data.get('pain_score', 0)} / 10", body_style),
         Paragraph("<b>Previous Knee Injury:</b>", body_style), Paragraph("Yes" if data.get("previous_injury") else "No", body_style)]
    ]
    findings_table = Table(findings_data, colWidths=[140, 130, 140, 130])
    findings_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor("#f8fafc")),
        ('BACKGROUND', (2,0), (2,-1), colors.HexColor("#f8fafc")),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(findings_table)
    story.append(Spacer(1, 15))

    # Recommendations
    story.append(Paragraph("<b>3. Actionable Clinical Referral & Advice</b>", h2_style))
    rec_text = str(data.get("recommendations", "Follow standard clinical guidelines."))
    rec_data = [[Paragraph(f"<b>Care Pathway:</b><br/>{rec_text}", body_style)]]
    rec_table = Table(rec_data, colWidths=[540])
    rec_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f0f7ff")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#2b4c7e")),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(rec_table)
    story.append(Spacer(1, 25))

    # Signature
    sig_data = [
        [Paragraph("<b>Edge Device ID:</b> ArthoSense-NER-01 (Offline SQLite)", sub_style),
         Paragraph("________________________________________<br/><b>Health Worker Signature & Stamp</b>", ParagraphStyle('S', parent=sub_style, alignment=1))]
    ]
    sig_table = Table(sig_data, colWidths=[270, 270])
    story.append(sig_table)

    doc.build(story)
    return output_path
