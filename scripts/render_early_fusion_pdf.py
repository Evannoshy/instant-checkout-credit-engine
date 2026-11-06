"""Render the early-fusion Markdown report, tables and figures to PDF."""
from pathlib import Path
import re
import html
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, Preformatted, KeepTogether
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader
import reportlab

fonts = Path(reportlab.__file__).resolve().parent / "fonts"
pdfmetrics.registerFont(TTFont("ReportSans", str(fonts / "Vera.ttf")))
pdfmetrics.registerFont(TTFont("ReportSans-Bold", str(fonts / "VeraBd.ttf")))
pdfmetrics.registerFont(TTFont("ReportSans-Italic", str(fonts / "VeraIt.ttf")))
pdfmetrics.registerFont(TTFont("ReportSans-BoldItalic", str(fonts / "VeraBI.ttf")))
pdfmetrics.registerFontFamily("ReportSans", normal="ReportSans", bold="ReportSans-Bold", italic="ReportSans-Italic", boldItalic="ReportSans-BoldItalic")

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/early-fusion-desc-distress-report.md"
OUTPUT = ROOT / "reports/tabular/early_fusion_research_report.pdf"
WIDTH = A4[0] - 38 * mm
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="ResearchTitle", parent=styles["Title"], fontName="ReportSans-Bold", fontSize=17, leading=21, textColor=colors.HexColor("#173550"), spaceAfter=12, alignment=0))
styles.add(ParagraphStyle(name="ResearchH1", parent=styles["Heading1"], fontName="ReportSans-Bold", fontSize=13.5, leading=17, textColor=colors.HexColor("#173550"), spaceBefore=14, spaceAfter=7, keepWithNext=True))
styles.add(ParagraphStyle(name="ResearchH2", parent=styles["Heading2"], fontName="ReportSans-Bold", fontSize=10.5, leading=13.5, textColor=colors.HexColor("#315d8c"), spaceBefore=10, spaceAfter=5, keepWithNext=True))
styles.add(ParagraphStyle(name="ResearchBody", parent=styles["BodyText"], fontName="ReportSans", fontSize=8.8, leading=11.8, spaceAfter=4, alignment=TA_JUSTIFY))
styles.add(ParagraphStyle(name="Cell", fontName="ReportSans", fontSize=7.0, leading=9, spaceAfter=0))
styles.add(ParagraphStyle(name="CellHead", fontName="ReportSans-Bold", fontSize=7.0, leading=9, textColor=colors.white))
styles.add(ParagraphStyle(name="Caption", parent=styles["BodyText"], fontName="ReportSans-Italic", fontSize=8, leading=10, alignment=TA_CENTER, textColor=colors.HexColor("#536474"), spaceAfter=8))
styles.add(ParagraphStyle(name="CodeBlock", fontName="Courier", fontSize=7.5, leading=10, leftIndent=8, spaceAfter=8))


def inline(text):
    text = html.escape(text)
    text = re.sub(r'!?(?:\[([^\]]+)\])\(([^)]+)\)',
                  lambda match: '<link href="' + match.group(2) + '">' + match.group(1) + '</link>' if match.group(2).startswith("https://") else match.group(1), text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'`([^`]+)`', r'<font name="Courier">\1</font>', text)
    return text


def footer(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#b9c7d2")); canvas.line(19*mm, 15*mm, A4[0]-19*mm, 15*mm)
    canvas.setFont("ReportSans", 7.5); canvas.setFillColor(colors.HexColor("#536474"))
    canvas.drawString(19*mm, 10*mm, "Early feature fusion | Development analysis")
    canvas.drawRightString(A4[0]-19*mm, 10*mm, str(document.page))
    canvas.restoreState()


def main():
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    story = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1; continue
        if line.startswith("```"):
            i += 1; code = []
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i]); i += 1
            story.append(Preformatted("\n".join(code), styles["CodeBlock"])); i += 1; continue
        if line.startswith("!["):
            match = re.match(r'!\[([^]]+)\]\(([^)]+)\)', line)
            if match:
                path = (SOURCE.parent / match.group(2)).resolve()
                pixel_width, pixel_height = ImageReader(str(path)).getSize()
                img = Image(str(path), width=WIDTH, height=WIDTH * pixel_height / pixel_width)
                story.append(KeepTogether([img, Spacer(1, 4), Paragraph(inline(match.group(1)), styles["Caption"])]))
            i += 1; continue
        if line.startswith("| "):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [x.strip() for x in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r'[-: ]+', x or ' ') for x in cells): rows.append(cells)
                i += 1
            size = len(rows[0]); cells = [[Paragraph(inline(v), styles["CellHead"] if j == 0 else styles["Cell"]) for v in row] for j, row in enumerate(rows)]
            widths = ([WIDTH * .26] + [WIDTH * .74 / (size-1)] * (size-1)) if size > 1 else [WIDTH]
            table = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#315d8c")), ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f0f5f8")]), ("VALIGN", (0,0), (-1,-1), "TOP"), ("BOTTOMPADDING", (0,0), (-1,-1), 5), ("TOPPADDING", (0,0), (-1,-1), 5), ("LINEBELOW", (0,0), (-1,0), .5, colors.HexColor("#315d8c"))]))
            story.extend([KeepTogether(table), Spacer(1, 9)]); continue
        if line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), styles["ResearchTitle"]))
        elif line.startswith("## "):
            story.append(Paragraph(inline(line[3:]), styles["ResearchH1"]))
        elif line.startswith("### "):
            story.append(Paragraph(inline(line[4:]), styles["ResearchH2"]))
        else:
            story.append(Paragraph(inline(line), styles["ResearchBody"]))
        i += 1
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, rightMargin=19*mm, leftMargin=19*mm, topMargin=16*mm, bottomMargin=19*mm, title="Early feature fusion for credit risk prediction", author="Zheng Wei")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)

if __name__ == "__main__":
    main()
