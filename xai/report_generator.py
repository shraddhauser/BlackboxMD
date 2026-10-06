"""
BLACKBOX MD — Medical PDF Report Generator
Module 7 of the BlackBoxMD-XAI Diagnostic Pipeline.

Generates official, exportable clinical diagnostic PDF reports containing:
- Clinical case metadata & timestamp
- Original CT scan & Grad-CAM Heatmap side-by-side
- Diagnostic prediction & confidence metric
- Key quantitative spatial insights (Location, Area %, Intensity)
- Clinical NLP Explanation & Radiological Impression
- Full 4-class differential probability breakdown
"""

import io
import os
import base64
from datetime import datetime
from typing import Dict, Any, Optional, Union
from PIL import Image

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
    HRFlowable
)

def _image_to_pil(img_input: Union[Image.Image, str, bytes]) -> Image.Image:
    """Helper to convert path, base64 data URI, or bytes into a PIL Image."""
    if isinstance(img_input, Image.Image):
        return img_input
    if isinstance(img_input, (str, os.PathLike)):
        str_val = str(img_input)
        if str_val.startswith("data:image"):
            # Base64 data URI
            header, encoded = str_val.split(",", 1)
            data = base64.b64decode(encoded)
            return Image.open(io.BytesIO(data)).convert("RGB")
        elif os.path.exists(str_val):
            return Image.open(str_val).convert("RGB")
    elif isinstance(img_input, bytes):
        return Image.open(io.BytesIO(img_input)).convert("RGB")

    # Fallback blank image
    return Image.new("RGB", (256, 256), color=(20, 20, 20))

def _pil_to_reportlab_image(pil_img: Image.Image, width=2.4*inch, height=2.4*inch) -> RLImage:
    """Converts a PIL Image into a ReportLab flowable image using in-memory bytes."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return RLImage(buf, width=width, height=height)

def generate_pdf_report(
    original_image: Union[Image.Image, str, bytes],
    gradcam_image: Union[Image.Image, str, bytes],
    prediction_data: Dict[str, Any],
    features: Dict[str, Any],
    nlp_data: Dict[str, Any],
    scan_id: Optional[str] = None,
    patient_id: Optional[str] = None
) -> bytes:
    """
    Builds an end-to-end medical clinical diagnostic report in PDF format.
    Returns the PDF document as bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f172a")
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#64748b")
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#334155")
    )
    bold_body = ParagraphStyle(
        "DocBoldBody",
        parent=body_style,
        fontName="Helvetica-Bold"
    )
    narrative_style = ParagraphStyle(
        "Narrative",
        parent=body_style,
        fontName="Helvetica",
        fontSize=10,
        leading=15,
        textColor=colors.HexColor("#0f172a")
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#94a3b8")
    )

    story = []

    # 1. Header & Branding Banner
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    scan_ref = scan_id or f"CT-{datetime.now().strftime('%Y%m%d-%H%M')}"
    pat_ref = patient_id or "ANONYMIZED-STUDY"

    header_table_data = [
        [
            Paragraph("<b>BLACKBOX <font color='#0284c7'>MD</font></b><br/><font size='8' color='#64748b'>Comparative Explainable AI • Pulmonary Oncology Diagnostic System</font>", title_style),
            Paragraph(f"<b>Report ID:</b> {scan_ref}<br/><b>Study Date:</b> {now_str}<br/><b>Subject:</b> {pat_ref}", subtitle_style)
        ]
    ]
    header_table = Table(header_table_data, colWidths=[330, 210])
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceAfter=12))

    # 2. Primary Diagnostic Callout Box
    pred_class = nlp_data.get("metadata", {}).get("predicted_class", prediction_data.get("predicted_class", "Undetermined"))
    conf_pct = nlp_data.get("metadata", {}).get("confidence_percentage", prediction_data.get("confidence", 0.0) * 100)

    # Status color: green for normal, red/amber for malignancies
    if "normal" in pred_class.lower():
        status_bg = colors.HexColor("#ecfdf5")
        status_border = colors.HexColor("#10b981")
        status_text_color = "#047857"
    else:
        status_bg = colors.HexColor("#fef2f2")
        status_border = colors.HexColor("#ef4444")
        status_text_color = "#b91c1c"

    diag_box_data = [
        [
            Paragraph(f"<font size='9' color='#64748b'>PREDICTED CLINICAL DIAGNOSIS</font><br/><b><font size='16' color='{status_text_color}'>{pred_class.upper()}</font></b>", body_style),
            Paragraph(f"<font size='9' color='#64748b'>DIAGNOSTIC CONFIDENCE</font><br/><b><font size='16' color='#0284c7'>{conf_pct:.1f}%</font></b><br/><font size='8' color='#64748b'>DenseNet-121 Softmax</font>", subtitle_style)
        ]
    ]
    diag_box = Table(diag_box_data, colWidths=[360, 180])
    diag_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), status_bg),
        ("BOX", (0, 0), (-1, -1), 1, status_border),
        ("PADDING", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
    ]))
    story.append(diag_box)
    story.append(Spacer(1, 14))

    # 3. Visual Evidence (CT Scan + Grad-CAM Heatmap Side-by-Side)
    story.append(Paragraph("<b>1. Imaging Evidence & XAI Localization Map</b>", section_heading))

    pil_orig = _image_to_pil(original_image)
    pil_gradcam = _image_to_pil(gradcam_image)

    rl_orig = _pil_to_reportlab_image(pil_orig, width=2.4*inch, height=2.4*inch)
    rl_gradcam = _pil_to_reportlab_image(pil_gradcam, width=2.4*inch, height=2.4*inch)

    images_table_data = [
        [rl_orig, rl_gradcam],
        [
            Paragraph("<b>Figure 1A:</b> Raw Input Chest CT Slice (256×256)", subtitle_style),
            Paragraph("<b>Figure 1B:</b> Grad-CAM Activation Heatmap Overlay", subtitle_style)
        ]
    ]
    images_table = Table(images_table_data, colWidths=[270, 270])
    images_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 1), (-1, 1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(images_table)
    story.append(Spacer(1, 12))

    # 4. Quantitative Spatial Insights Table (Flowchart Step 5 / Step 7)
    story.append(Paragraph("<b>2. Heatmap Feature Extraction (Quantitative Insights)</b>", section_heading))

    location = features.get("location", "Upper Right Lung")
    activation_lvl = features.get("activation_level", "High")
    region_size_str = features.get("region_size_str", f"{features.get('region_size_pct', 18.0):.1f}% of lung area")
    intensity = features.get("activation_intensity", 0.82)
    concentration = features.get("concentration", "High (Focal)")

    insights_data = [
        [Paragraph("<b>Metric / Parameter</b>", bold_body), Paragraph("<b>Measurement / Value</b>", bold_body), Paragraph("<b>Clinical Significance</b>", bold_body)],
        [Paragraph("Anatomical Location", body_style), Paragraph(f"<b>{location}</b>", body_style), Paragraph("Primary lesion centroid in lung coordinate grid", subtitle_style)],
        [Paragraph("Activation Level", body_style), Paragraph(f"<b>{activation_lvl}</b>", body_style), Paragraph("Relative gradient magnitude on final dense layer", subtitle_style)],
        [Paragraph("Region Size", body_style), Paragraph(f"<b>{region_size_str}</b>", body_style), Paragraph("Proportion of segmented parenchyma involved", subtitle_style)],
        [Paragraph("Activation Intensity", body_style), Paragraph(f"<b>{intensity:.2f} ({activation_lvl})</b>", body_style), Paragraph("Peak normalized attribution score (0.0 – 1.0)", subtitle_style)],
        [Paragraph("Spatial Distribution", body_style), Paragraph(f"<b>{concentration}</b>", body_style), Paragraph("Dispersion cluster radius around lesion epicenter", subtitle_style)],
    ]
    insights_table = Table(insights_data, colWidths=[150, 160, 230])
    insights_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
    ]))
    story.append(insights_table)
    story.append(Spacer(1, 12))

    # 5. Clinical NLP Explanation (Flowchart Step 6 / Step 7)
    story.append(Paragraph("<b>3. Natural Language Explanation (NLP Module)</b>", section_heading))

    primary_narrative = nlp_data.get("primary_explanation", "")
    clinical_impression = nlp_data.get("clinical_impression", "")
    recommended_action = nlp_data.get("recommended_action", "")

    nlp_box_data = [
        [Paragraph("<b>Primary Finding:</b>", bold_body)],
        [Paragraph(f"<i>\"{primary_narrative}\"</i>", narrative_style)],
        [Spacer(1, 4)],
        [Paragraph("<b>Formal Radiological Impression:</b>", bold_body)],
        [Paragraph(clinical_impression, body_style)],
        [Spacer(1, 4)],
        [Paragraph("<b>Recommended Next Steps:</b>", bold_body)],
        [Paragraph(recommended_action, body_style)]
    ]
    nlp_box = Table(nlp_box_data, colWidths=[540])
    nlp_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#94a3b8")),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(nlp_box)
    story.append(Spacer(1, 12))

    # 6. Multi-Class Probability Distribution Table
    probs = prediction_data.get("probabilities", {})
    if probs:
        story.append(Paragraph("<b>4. Multi-Class Probability Breakdown</b>", section_heading))
        prob_headers = [Paragraph("<b>Class Name</b>", bold_body), Paragraph("<b>Probability</b>", bold_body), Paragraph("<b>Confidence Bar</b>", bold_body)]
        prob_rows = [prob_headers]
        for cls_name, p in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            bar_len = int(p * 20)
            bar_repr = "█" * bar_len + "░" * (20 - bar_len)
            clean_name = cls_name.replace(".", " ").replace("_", " ").title()
            prob_rows.append([
                Paragraph(clean_name, body_style),
                Paragraph(f"<b>{p * 100:.2f}%</b>", body_style),
                Paragraph(f"<font color='#0284c7'>{bar_repr}</font>", body_style)
            ])
        prob_table = Table(prob_rows, colWidths=[200, 110, 230])
        prob_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("PADDING", (0, 0), (-1, -1), 4),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ]))
        story.append(prob_table)
        story.append(Spacer(1, 14))

    # 7. Medical Disclaimer & Footer
    disclaimer_text = (
        "CONFIDENTIAL MEDICAL REPORT — BlackBoxMD-XAI is an artificial intelligence-assisted diagnostic research tool. "
        "All visual explanations and automated findings must be clinically verified by a licensed radiologist or oncologist "
        "before making diagnostic or therapeutic decisions."
    )
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceAfter=6))
    story.append(Paragraph(disclaimer_text, disclaimer_style))

    doc.build(story)
    return buffer.getvalue()


if __name__ == "__main__":
    print("=" * 65)
    print("BLACKBOX MD — Medical Report Generator Self-Test")
    print("=" * 65)

    # Create synthetic test images
    img1 = Image.new("RGB", (256, 256), color=(30, 40, 50))
    img2 = Image.new("RGB", (256, 256), color=(80, 20, 20))

    pred = {
        "predicted_class": "adenocarcinoma",
        "confidence": 0.914,
        "probabilities": {
            "adenocarcinoma": 0.914,
            "squamous.cell.carcinoma": 0.051,
            "large.cell.carcinoma": 0.021,
            "normal": 0.014
        }
    }
    feats = {
        "location": "Upper Right Lung",
        "activation_level": "High",
        "region_size_pct": 18.2,
        "region_size_str": "18.2% of lung area",
        "activation_intensity": 0.82,
        "concentration": "High (Focal)"
    }
    nlp = {
        "primary_explanation": "The model predicts Adenocarcinoma with 91.4% confidence. The prediction was primarily influenced by a highly activated region in the upper right portion of the lung, which occupies about 18% of the lung area.",
        "clinical_impression": "IMPRESSION: Findings consistent with Adenocarcinoma. Saliency mapping localizes the primary diagnostic determinant to the Upper Right Lung (occupying ~18.2% of segmented lung volume) with focal concentration and peak intensity 0.82.",
        "recommended_action": "Correlate with contrast-enhanced chest CT / PET-CT targeting the Upper Right Lung lesion.",
        "metadata": {
            "predicted_class": "Adenocarcinoma",
            "confidence_percentage": 91.4
        }
    }

    pdf_bytes = generate_pdf_report(img1, img2, pred, feats, nlp)
    output_path = "results/test_sample_report.pdf"
    os.makedirs("results", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(pdf_bytes)

    print(f"[OK] Generated PDF Report: {output_path} ({len(pdf_bytes)} bytes)")
    print("=" * 65)
