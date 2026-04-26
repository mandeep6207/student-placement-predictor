from __future__ import annotations

from io import BytesIO
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.shapes import Drawing, String


SUGGESTION_SEPARATOR = " | "


def _build_probability_pie(probability: float) -> Drawing:
    placed = max(0.0, min(100.0, probability))
    remaining = 100.0 - placed

    drawing = Drawing(360, 210)
    title = String(0, 190, "Placement Probability Breakdown", fontSize=12)
    drawing.add(title)

    pie = Pie()
    pie.x = 60
    pie.y = 10
    pie.width = 180
    pie.height = 180
    pie.data = [placed, remaining]
    pie.labels = [f"Placed: {placed:.2f}%", f"Not placed: {remaining:.2f}%"]
    pie.slices[0].fillColor = colors.HexColor("#2b9348")
    pie.slices[1].fillColor = colors.HexColor("#d9d9d9")
    pie.slices.strokeWidth = 0.6
    drawing.add(pie)
    return drawing


def _build_importance_table(feature_importance: dict[str, float]) -> Table:
    rows = [["Feature", "Importance"]]
    for feature, value in sorted(feature_importance.items(), key=lambda item: item[1], reverse=True):
        rows.append([feature, f"{value * 100:.2f}%"])

    table = Table(rows, colWidths=[8 * cm, 6 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e5f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c5ced3")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#f4f7fa")]),
                ("ALIGN", (1, 1), (1, -1), "RIGHT"),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def build_prediction_report(
    username: str,
    record: dict[str, Any],
    feature_importance: dict[str, float],
    model_name: str,
) -> BytesIO:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, title="Placement Report")

    styles = getSampleStyleSheet()
    heading = ParagraphStyle(
        "Heading",
        parent=styles["Heading1"],
        textColor=colors.HexColor("#1f4e5f"),
        spaceAfter=10,
    )

    story = []
    story.append(Paragraph("Student Placement Predictor Report", heading))
    story.append(Paragraph(f"Generated for: <b>{username}</b>", styles["BodyText"]))
    story.append(Paragraph(f"Timestamp: {record['created_at']}", styles["BodyText"]))
    story.append(Spacer(1, 10))

    input_rows = [
        ["CGPA", str(record["cgpa"])],
        ["Skills", str(record["skills"])],
        ["Internship", "Yes" if int(record["internship"]) == 1 else "No"],
        ["Projects", str(record["projects"])],
        ["Communication", str(record["communication"])],
        ["Model", model_name],
        ["Placement Probability", f"{record['probability']:.2f}%"],
        ["Result Label", record["label"]],
    ]

    input_table = Table([["Field", "Value"], *input_rows], colWidths=[7 * cm, 7 * cm])
    input_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f766e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c5ced3")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fbfb")]),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    story.append(Paragraph("Profile Summary", styles["Heading3"]))
    story.append(input_table)
    story.append(Spacer(1, 12))

    suggestion_items = [item.strip() for item in str(record["suggestions"]).split(SUGGESTION_SEPARATOR) if item.strip()]
    story.append(Paragraph("Personalized Suggestions", styles["Heading3"]))
    for item in suggestion_items:
        story.append(Paragraph(f"- {item}", styles["BodyText"]))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Feature Importance", styles["Heading3"]))
    story.append(_build_importance_table(feature_importance))
    story.append(Spacer(1, 12))
    story.append(_build_probability_pie(float(record["probability"])))

    doc.build(story)
    buffer.seek(0)
    return buffer
