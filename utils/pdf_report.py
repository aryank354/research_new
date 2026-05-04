# utils/pdf_report.py

import os
import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak, HRFlowable
)

def build_visual_pdf(results: list, output_path: str):
    """Generates a professional PDF report from the experiment results."""
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=1.5*cm, rightMargin=1.5*cm,
        topMargin=1.5*cm, bottomMargin=1.5*cm
    )
    styles = getSampleStyleSheet()
    
    # Styles
    title_style = ParagraphStyle('CustomTitle', parent=styles['Title'], fontSize=20, textColor=colors.HexColor('#1a237e'), spaceAfter=6)
    sub_style = ParagraphStyle('SubTitle', parent=styles['Normal'], fontSize=11, alignment=TA_CENTER, spaceAfter=15)
    section_style = ParagraphStyle('Section', parent=styles['Heading2'], fontSize=14, textColor=colors.HexColor('#2e7d32'), spaceAfter=6)
    caption_style = ParagraphStyle('Caption', parent=styles['Normal'], fontSize=9, textColor=colors.HexColor('#455a64'), alignment=TA_CENTER)

    story = []
    
    # Cover Page / Header
    story.append(Paragraph("Fragile Watermarking: Visual Evaluation Report", title_style))
    now = datetime.datetime.now().strftime("%B %d, %Y  %H:%M")
    story.append(Paragraph(f"Generated: {now} | Total Experiments: {len(results)}", sub_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#1a237e'), spaceAfter=15))

    IMG_W = 5.5 * cm
    IMG_H = 5.5 * cm

    def make_img(path):
        return RLImage(path, width=IMG_W, height=IMG_H, kind='proportional')

    # Helper function to safely format metrics (allows for "N/A" strings)
    def format_metric(val):
        if isinstance(val, str):
            return val
        return f"{val:.4f}"

    for idx, r in enumerate(results):
        story.append(Paragraph(f"Experiment {idx+1}: {r['Image']} — Attack: {r['Attack']}", section_style))
        
        # --- Image Grid (Row 1: Orig, Watermarked, Attacked | Row 2: True Map, Recovered, Blank) ---
        img_table = Table(
            [
                [make_img(r['paths']['orig']), make_img(r['paths']['watermarked']), make_img(r['paths']['tampered'])],
                [Paragraph("1. Original", caption_style), Paragraph("2. Watermarked", caption_style), Paragraph("3. Attacked", caption_style)],
                [make_img(r['paths']['true_map']), make_img(r['paths']['recovered']), ""],
                [Paragraph("4. True Tamper Map", caption_style), Paragraph("5. Recovered Image", caption_style), ""]
            ],
            colWidths=[IMG_W + 0.5*cm, IMG_W + 0.5*cm, IMG_W + 0.5*cm]
        )
        img_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ]))
        story.append(img_table)
        story.append(Spacer(1, 0.3*cm))

        # --- Metrics Table ---
        metrics_data = [
            ["Metric", "Value", "Metric", "Value"],
            ["Watermarked PSNR", f"{r['W-PSNR']:.2f} dB", "Recovered PSNR", f"{r['R-PSNR']:.2f} dB"],
            ["Watermarked SSIM", f"{r['W-SSIM']:.4f}", "Recovered SSIM", f"{r['R-SSIM']:.4f}"],
            ["True Positive Rate (TPR)", format_metric(r['TPR']), "Recovered NC", format_metric(r['R-NC'])],
            ["False Positive Rate (FPR)", format_metric(r['FPR']), "Tamper F1-Score", format_metric(r['F1-Score'])]
        ]
        
        m_table = Table(metrics_data, colWidths=[4.5*cm, 3.5*cm, 4.5*cm, 3.5*cm])
        m_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#37474f')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#eceff1'), colors.white]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#b0bec5')),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('ALIGN', (3, 0), (3, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(m_table)
        
        story.append(Spacer(1, 0.8*cm))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cfd8dc'), spaceAfter=15))

    doc.build(story)