"""Sponsor presentation (English) — CPC for manual material handling activity recognition.

Run:  uv run --with python-pptx python Presentacion/build_deck.py   (figures from make_figures.py)
Output: Presentacion/CPC_MMH_sponsors.pptx  (upload to Google Slides: File > Import slides)
All figures come from Articulo/articulo.tex.
"""
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_MARKER_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = Path(__file__).parent
REPO = HERE.parent
FIG = HERE / "figs"

# ---------- palette (poster_plantilla.tex + harmonised accents) ----------
def rgb(h):
    return RGBColor.from_string(h)

OLIVE = rgb("9AA542")       # section headers
OLIVE_DARK = rgb("1B2208")  # titles
OLIVE_LINK = rgb("6B7A1F")  # accents / CPC series
OLIVE_SOFT = rgb("E4E8C7")  # soft fills
OLIVE_MIST = rgb("F3F5E6")  # very soft card fill
INK = rgb("1A1A10")
MUTED = rgb("6B6B5E")
SLATE = rgb("3E5C76")       # supervised / RF series
SLATE_SOFT = rgb("DDE5EC")
TEAL = rgb("2E7D73")
TEAL_SOFT = rgb("D9ECE8")
AMBER = rgb("B9822A")       # sparing highlight
AMBER_SOFT = rgb("F4E7CF")
WHITE = rgb("FFFFFF")
GREY = rgb("B8B9AE")

HEAD = "Cambria"
BODY = "Calibri"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
SW, SH = 13.333, 7.5
M = 0.5  # margin

state = {"n": 0}


# ---------- helpers ----------
def tb(slide, x, y, w, h, text="", size=16, color=INK, bold=False, font=BODY, align=PP_ALIGN.LEFT,
       anchor=MSO_ANCHOR.TOP, italic=False, margin=0.0, name=None):
    """Text box. `text` may be a str or a list of paragraphs; a paragraph may be a list of runs
    (text, {opts})."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        box.name = name
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, f"margin_{side}", Inches(margin))
    paras = text if isinstance(text, list) else [text]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        runs = para if isinstance(para, list) else [(para, {})]
        for rt, opt in runs:
            r = p.add_run()
            r.text = rt
            f = r.font
            f.name = opt.get("font", font)
            f.size = Pt(opt.get("size", size))
            f.bold = opt.get("bold", bold)
            f.italic = opt.get("italic", italic)
            f.color.rgb = opt.get("color", color)
        if isinstance(para, list) and para and "space_after" in para[0][1]:
            p.space_after = Pt(para[0][1]["space_after"])
    return box


def bullets(slide, x, y, w, h, items, size=16, color=INK, space=8, bullet_color=OLIVE_LINK):
    """Bulleted list; each item str or list of runs."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, f"margin_{side}", Inches(0))
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(space)
        r0 = p.add_run()
        r0.text = "■  "
        r0.font.size = Pt(size - 6)
        r0.font.color.rgb = bullet_color
        r0.font.name = BODY
        runs = it if isinstance(it, list) else [(it, {})]
        for rt, opt in runs:
            r = p.add_run()
            r.text = rt
            r.font.name = BODY
            r.font.size = Pt(opt.get("size", size))
            r.font.bold = opt.get("bold", False)
            r.font.italic = opt.get("italic", False)
            r.font.color.rgb = opt.get("color", color)
    return box


def shape(slide, kind, x, y, w, h, fill=None, line=None, line_w=1.25, dash=False, name=None):
    s = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        s.name = name
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(line_w)
        if dash:
            from pptx.enum.dml import MSO_LINE_DASH_STYLE
            s.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    s.shadow.inherit = False
    if kind == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = 0.08
    return s


def shape_text(s, text, size=14, color=INK, bold=False, font=BODY, align=PP_ALIGN.CENTER,
               anchor=MSO_ANCHOR.MIDDLE, margin=0.08):
    tf = s.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, f"margin_{side}", Inches(margin))
    paras = text if isinstance(text, list) else [text]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        runs = para if isinstance(para, list) else [(para, {})]
        for rt, opt in runs:
            r = p.add_run()
            r.text = rt
            r.font.name = opt.get("font", font)
            r.font.size = Pt(opt.get("size", size))
            r.font.bold = opt.get("bold", bold)
            r.font.italic = opt.get("italic", False)
            r.font.color.rgb = opt.get("color", color)


def arrow(slide, x1, y1, x2, y2, color=OLIVE, w=2.5, dash=False):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color
    c.line.width = Pt(w)
    if dash:
        from pptx.enum.dml import MSO_LINE_DASH_STYLE
        c.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    ln = c.line._get_or_add_ln()
    from lxml import etree
    tail = etree.SubElement(ln, "{http://schemas.openxmlformats.org/drawingml/2006/main}tailEnd")
    tail.set("type", "triangle")
    tail.set("w", "med")
    tail.set("len", "med")
    return c


def new_slide(title, section=None, notes=""):
    state["n"] += 1
    s = prs.slides.add_slide(BLANK)
    if title:
        bar = shape(s, MSO_SHAPE.RECTANGLE, M, 0.4, SW - 2 * M, 0.78, fill=OLIVE, name="Title band")
        tb(s, M + 0.25, 0.4, 9.6, 0.78, title, size=28, color=WHITE, font=HEAD, anchor=MSO_ANCHOR.MIDDLE,
           name="Title")
        if section:
            tb(s, SW - M - 3.0, 0.4, 2.8, 0.78, section.upper(), size=12, color=OLIVE_SOFT, bold=True,
               align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE, name="Section tag")
        # footer
        tb(s, M, SH - 0.42, 9, 0.3, "Self-supervised CPC for manual material handling recognition  ·  "
           "Tecnológico de Monterrey × Scuola Superiore Sant'Anna", size=10, color=MUTED, name="Footer")
        tb(s, SW - M - 1, SH - 0.42, 1, 0.3, str(state["n"]), size=10, color=MUTED, align=PP_ALIGN.RIGHT,
           name="Slide number")
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def stat(slide, x, y, w, h, value, label, sub="", fill=OLIVE_MIST, vcolor=OLIVE_LINK, vsize=40):
    shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, fill=fill)
    tb(slide, x + 0.2, y + 0.12, w - 0.4, h * 0.5, value, size=vsize, color=vcolor, bold=True, font=HEAD,
       anchor=MSO_ANCHOR.BOTTOM)
    paras = [[(label, {"bold": True, "size": 15, "color": INK})]]
    if sub:
        paras.append([(sub, {"size": 12, "color": MUTED})])
    tb(slide, x + 0.2, y + h * 0.5 + 0.12, w - 0.4, h * 0.5 - 0.2, paras)


def style_chart(chart, legend=True, size=12):
    chart.font.name = BODY
    chart.font.size = Pt(size)
    chart.font.color.rgb = INK
    chart.has_legend = legend
    if legend:
        chart.legend.position = XL_LEGEND_POSITION.BOTTOM
        chart.legend.include_in_layout = False
        chart.legend.font.size = Pt(size)


def quiet_axes(chart, vmin=None, vmax=None, major=None, gridlines=True, num_fmt="0.0", hide_values=False):
    va = chart.value_axis
    if hide_values:
        va.visible = False
        gridlines = False
    if vmin is not None:
        va.minimum_scale = vmin
    if vmax is not None:
        va.maximum_scale = vmax
    if major:
        va.major_unit = major
    va.has_major_gridlines = gridlines
    if gridlines:
        va.major_gridlines.format.line.color.rgb = rgb("E6E7DD")
        va.major_gridlines.format.line.width = Pt(0.75)
    va.format.line.fill.background()
    va.tick_labels.font.color.rgb = MUTED
    va.tick_labels.number_format = num_fmt
    va.tick_labels.number_format_is_linked = False
    ca = chart.category_axis
    ca.format.line.color.rgb = GREY
    ca.tick_labels.font.color.rgb = INK
    ca.has_major_gridlines = False


# =====================================================================
# 1. Title
# =====================================================================
s = new_slide(None, notes=(
    "[Isaura · ~45 s]\n"
    "Good morning. We are Isaura, Paola and Dylan, from the Master in Applied Artificial Intelligence at "
    "Tecnológico de Monterrey. Thank you for sponsoring this project and for sharing the Manual Material "
    "Handling dataset. Today we present how self-supervised learning, specifically Contrastive Predictive "
    "Coding, can recognize manual material handling activities with far fewer labels. "
    "I will introduce the problem and the gap, Paola will present our approach and methodology, and Dylan "
    "will show the results and discuss them against the state of the art."))
s.shapes.add_picture(str(REPO / "ResumenEjecutivo/logo.jpeg"), Inches(M + 0.1), Inches(0.45), height=Inches(0.85))
s.shapes.add_picture(str(REPO / "ResumenEjecutivo/sssa.png"), Inches(SW - M - 0.1 - 3.05), Inches(0.5),
                     height=Inches(0.85))
shape(s, MSO_SHAPE.RECTANGLE, M, 1.75, SW - 2 * M, 0.08, fill=OLIVE)
tb(s, M + 0.3, 2.15, SW - 2 * M - 0.6, 1.9,
   "Self-supervised learning with Contrastive Predictive Coding to recognize manual material handling activities",
   size=36, color=OLIVE_DARK, font=HEAD, anchor=MSO_ANCHOR.MIDDLE)
tb(s, M + 0.3, 4.1, 10, 0.5, "Fewer labels, fewer sensors, evaluated on people the model has never seen",
   size=20, color=OLIVE_LINK, italic=True)
tb(s, M + 0.3, 4.95, 11, 1.4, [
    [("Isaura Yutsil Flores Escamilla   ·   Paola Alejandra León Guarneros   ·   Dylan Ulises Quiroz Hernández",
      {"size": 18, "color": INK})],
    [("Advisor: Dr. Gerardo Jesús Camacho González", {"size": 15, "color": MUTED})],
    [("Master in Applied Artificial Intelligence · Integrative Project · November 2026",
      {"size": 15, "color": MUTED})],
])
shape(s, MSO_SHAPE.RECTANGLE, M, 6.75, SW - 2 * M, 0.08, fill=OLIVE)

# =====================================================================
# 2. Roadmap
# =====================================================================
s = new_slide("Roadmap", None, notes=(
    "[Isaura · ~30 s]\n"
    "This is the path of the talk. First the problem: why recognizing activities matters for ergonomics, and "
    "why labels are the bottleneck. Then the gap in the state of the art, our proposed approach, the "
    "methodology, the results, and finally what they mean in relation to the state of the art."))
steps = [("1", "Problem", "Why activities, why labels hurt", OLIVE_LINK),
         ("2", "Gap", "What the state of the art leaves open", OLIVE),
         ("3", "Approach", "Learn first without labels (CPC)", TEAL),
         ("4", "Methodology", "Data, protocol, model choices", SLATE),
         ("5", "Results", "All labels vs. few labels", OLIVE_LINK),
         ("6", "Discussion", "Significance vs. the state of the art", OLIVE_DARK)]
cw, gap = 1.95, 0.12
for i, (num, name, desc, col) in enumerate(steps):
    x = M + i * (cw + gap)
    ch = shape(s, MSO_SHAPE.CHEVRON if i else MSO_SHAPE.PENTAGON, x, 2.1, cw + 0.25, 1.25, fill=col)
    shape_text(ch, [[(num, {"size": 26, "bold": True, "color": WHITE, "font": HEAD})]])
    tb(s, x + 0.1, 3.6, cw - 0.1, 0.5, name, size=20, bold=True, color=OLIVE_DARK, font=HEAD)
    tb(s, x + 0.1, 4.1, cw - 0.15, 1.0, desc, size=14, color=MUTED)
c = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, M, 5.45, SW - 2 * M, 1.05, fill=OLIVE_MIST)
shape_text(c, [[("Research question  ", {"bold": True, "color": OLIVE_LINK, "size": 17}),
                ("Can a model pretrained without labels recognize the activities of a new person as well as a "
                 "supervised model, and beat it when labels are scarce?", {"size": 17, "italic": True})]],
           align=PP_ALIGN.LEFT, margin=0.25)

# =====================================================================
# 3. Problem — what and when
# =====================================================================
s = new_slide("Ergonomic risk starts with knowing what and when", "1 · Problem", notes=(
    "[Isaura · ~1 min]\n"
    "Manual material handling — lifting, holding, carrying and placing loads by hand — is one of the "
    "industrial tasks most associated with musculoskeletal injuries. To assess that risk we first need to know "
    "WHAT the worker is doing and WHEN: lifting a load from the floor does not load the back like taking it "
    "from a table. In the dataset we use there are seven activities, shown here. Today this segmentation is "
    "done by hand by an annotator, a task that the dataset authors themselves describe as very time-demanding. "
    "Human activity recognition with wearable sensors can automate it."))
bullets(s, M, 1.65, 5.6, 3.2, [
    [("Manual material handling ", {"bold": True}), ("is among the tasks most linked to musculoskeletal injuries", {})],
    [("Risk depends on the activity: ", {"bold": True}), ("lifting from the floor ≠ lifting from a table", {})],
    [("Today an annotator segments every recording by hand — ", {}), ("slow and not scalable", {"bold": True})],
    [("Wearable sensors + ", {}), ("Human Activity Recognition (HAR)", {"bold": True}), (" can automate it", {})],
], size=17, space=12)
acts = [("N", "Neutral posture", OLIVE_SOFT), ("LF", "Lift from floor", AMBER_SOFT), ("K", "Hold the load", TEAL_SOFT),
        ("PT", "Place on table", SLATE_SOFT), ("LT", "Lift from table", SLATE_SOFT), ("W", "Walk with load", TEAL_SOFT),
        ("PF", "Place on floor", AMBER_SOFT)]
tb(s, 6.6, 1.6, 6.2, 0.4, "Seven activities to recognize", size=16, bold=True, color=OLIVE_DARK)
bw, bh = 1.92, 1.12
for i, (code, name, fill) in enumerate(acts):
    r, c_ = divmod(i, 3) if i < 6 else (2, 1)
    x = 6.6 + c_ * (bw + 0.18)
    y = 2.1 + r * (bh + 0.18)
    b = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, bw, bh, fill=fill)
    shape_text(b, [[(code, {"size": 26, "bold": True, "color": OLIVE_DARK, "font": HEAD})],
                   [(name, {"size": 13, "color": INK})]])
c = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, M, 5.35, 5.6, 1.3, fill=OLIVE_MIST)
shape_text(c, [[("Floor level ", {"bold": True, "color": AMBER}), ("vs. ", {}),
                ("table level ", {"bold": True, "color": SLATE}),
                ("is what matters most for the back: trunk flexion is very different.", {})]],
           size=15, align=PP_ALIGN.LEFT, margin=0.2)

# =====================================================================
# 4. Labels are the bottleneck
# =====================================================================
s = new_slide("Recording is cheap; labeling is the bottleneck", "1 · Problem", notes=(
    "[Isaura · ~1 min]\n"
    "Supervised models need large volumes of annotated data. In HAR, recording hours of movement is cheap, "
    "but labeling them instant by instant is expensive. In this dataset, about 29 minutes of labeled activity "
    "per person already means around 600 segments marked one by one, more than 8,300 for the 14 people. "
    "And the cost repeats for every new person, task or work environment. So the natural question is: can a "
    "model learn first from unlabeled data and then need only a few annotations?"))
stat(s, M, 1.75, 3.9, 2.0, "≈ 600", "segments marked by hand", "per recorded person (≈ 29 min of activity)")
stat(s, M + 4.2, 1.75, 3.9, 2.0, "8,337", "hand-marked segments", "14 people, ≈ 7 h labeled", fill=SLATE_SOFT,
     vcolor=SLATE)
stat(s, M + 8.4, 1.75, 3.93, 2.0, "× every", "new person, task or plant", "the labeling cost repeats",
     fill=AMBER_SOFT, vcolor=AMBER)
# cheap vs expensive flow
y0 = 4.35
b1 = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, M, y0, 3.7, 1.5, fill=TEAL_SOFT)
shape_text(b1, [[("Unlabeled movement", {"bold": True, "size": 17, "color": TEAL})],
                [("hours of recordings — cheap and abundant", {"size": 14})]])
b2 = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, M + 4.3, y0, 3.7, 1.5, fill=AMBER_SOFT)
shape_text(b2, [[("Labeled windows", {"bold": True, "size": 17, "color": AMBER})],
                [("instant-by-instant annotation — scarce and costly", {"size": 14})]])
b3 = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, M + 8.6, y0, 3.73, 1.5, fill=OLIVE)
shape_text(b3, [[("Can we learn from the first", {"bold": True, "size": 17, "color": WHITE})],
                [("and need only a few of the second?", {"size": 15, "color": WHITE})]])
arrow(s, M + 3.75, y0 + 0.75, M + 4.25, y0 + 0.75, color=MUTED)
arrow(s, M + 8.05, y0 + 0.75, M + 8.55, y0 + 0.75, color=MUTED)

# =====================================================================
# 5. State of the art
# =====================================================================
s = new_slide("State of the art: two lines of work that have not met", "2 · Gap", notes=(
    "[Isaura · ~1.5 min]\n"
    "Two lines of work frame this project. On the left, supervised deep learning on this very dataset: "
    "Bassani, Avizzano and Filippeschi, 2025, in Sensors. Their best network, a BiLSTM on ten signals — eight "
    "motion and two sEMG — at 240 hertz, reaches 0.903 macro F1 on a held-out person, and 95.7 percent on a "
    "random 70–30 split. This is our reference of what a supervised model achieves with these data. "
    "On the right, self-supervised learning: Contrastive Predictive Coding, by van den Oord and colleagues, "
    "learns a representation by predicting the future of a signal without labels. Haresamudram and "
    "colleagues applied it to HAR with accelerometers: with all labels it is on par with supervised training, "
    "and its advantage appears when labels are scarce. These two lines had not been brought together."))
colw = (SW - 2 * M - 0.5) / 2
for i, (head, col, soft, ref, pts) in enumerate([
    ("Supervised deep learning on MMH", SLATE, SLATE_SOFT, "Bassani, Avizzano & Filippeschi (2025), Sensors", [
        [("Same dataset and 7 activities", {"bold": True})],
        [("Best model: ", {}), ("BiLSTM", {"bold": True}), (" · 8 motion + 2 sEMG signals · 1 s at 240 Hz", {})],
        [("0.903", {"bold": True, "color": SLATE}), (" macro F1 on a held-out person (subject 11)", {})],
        [("0.957", {"bold": True, "color": SLATE}), (" on a random 70–30 split of windows", {})],
        [("Needs every window labeled", {"italic": True})],
    ]),
    ("Self-supervised learning: CPC", OLIVE_LINK, OLIVE_MIST, "van den Oord et al. (2018); Haresamudram et al. (2021)", [
        [("Learns by ", {}), ("predicting the future", {"bold": True}), (" of the signal — no labels", {})],
        [("Applied to HAR with ", {}), ("accelerometers", {"bold": True}), (", daily activities", {})],
        [("All labels: ", {}), ("on par", {"bold": True, "color": OLIVE_LINK}), (" with supervised training", {})],
        [("Few labels: ", {}), ("advantage", {"bold": True, "color": OLIVE_LINK}), (" over supervised", {})],
        [("Not tested on material handling", {"italic": True})],
    ])]):
    x = M + i * (colw + 0.5)
    shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, x, 1.6, colw, 5.0, fill=soft)
    hb = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, x, 1.6, colw, 0.75, fill=col)
    shape_text(hb, head, size=20, color=WHITE, bold=True, font=HEAD)
    bullets(s, x + 0.35, 2.7, colw - 0.7, 3.2, pts, size=18, space=14, bullet_color=col)
    tb(s, x + 0.35, 5.95, colw - 0.7, 0.4, ref, size=12, color=MUTED, italic=True)

# =====================================================================
# 6. The gap
# =====================================================================
s = new_slide("The gap we address", "2 · Gap", notes=(
    "[Isaura · ~1.5 min]\n"
    "From there we identified four open points. First, self-supervised learning had not been tested on "
    "manual material handling. Second, nobody had measured how many labels a new person, task or "
    "environment really needs — label efficiency. Third, for a wearable system we need to know which "
    "sensors are really necessary; a 17-sensor suit is not practical on a factory floor. And fourth, "
    "complementary to the held-out-subject result of the reference work, we wanted an estimate over every "
    "person in the dataset, so that we can see how much performance varies between people. "
    "That gives our research question: can a model pretrained without labels match a supervised model on "
    "new people, and beat it when labels are scarce?"))
gaps = [("1", "Self-supervision untested in MMH", "CPC has been shown for daily activities with accelerometers, not for load handling", OLIVE_LINK, OLIVE_MIST),
        ("2", "Label efficiency unknown", "How many labeled examples does a new person, task or plant need?", AMBER, AMBER_SOFT),
        ("3", "Which sensors are needed?", "A 17-sensor suit is impractical on a factory floor", TEAL, TEAL_SOFT),
        ("4", "Variation across people", "Complement a single held-out person with an estimate over every person", SLATE, SLATE_SOFT)]
gw = (SW - 2 * M - 3 * 0.3) / 4
for i, (n, h, d, col, soft) in enumerate(gaps):
    x = M + i * (gw + 0.3)
    shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, x, 1.65, gw, 3.15, fill=soft)
    circ = shape(s, MSO_SHAPE.OVAL, x + 0.3, 1.9, 0.75, 0.75, fill=col)
    shape_text(circ, n, size=22, bold=True, color=WHITE, font=HEAD, margin=0)
    tb(s, x + 0.3, 2.85, gw - 0.55, 0.85, h, size=18, bold=True, color=OLIVE_DARK, font=HEAD)
    tb(s, x + 0.3, 3.7, gw - 0.55, 1.05, d, size=14, color=INK)
q = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, M, 5.15, SW - 2 * M, 1.45, fill=OLIVE)
shape_text(q, [[("Research question", {"bold": True, "size": 15, "color": OLIVE_SOFT})],
               [("Can a model pretrained without labels (CPC) recognize the activities of a new person as well as a "
                 "supervised model — and beat it when labels are scarce?", {"size": 19, "color": WHITE, "font": HEAD})]],
           align=PP_ALIGN.LEFT, margin=0.3)

# =====================================================================
# 7. Proposed approach
# =====================================================================
s = new_slide("Our approach: learn from movement first, label later", "3 · Approach", notes=(
    "[Isaura · ~1.5 min]\n"
    "Our approach has two phases. In phase one, the network sees only unlabeled movement and solves a task "
    "that needs no labels: given the past four seconds, recognize which of many candidates is the real "
    "future, up to one second ahead. To do that well it has to learn how the body moves. "
    "In phase two we keep that representation and train a small classification head with labels — and the "
    "question is how few labels are enough. "
    "Around it we set four objectives: find the signal source that separates the activities, compare CPC with "
    "supervised models under the same person-independent protocol, find the minimum set of sensors, and "
    "measure performance with few labeled windows. Now Paola will explain how we did it."))
# Phase 1
p1 = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, M, 1.65, 5.9, 2.55, fill=TEAL_SOFT)
tb(s, M + 0.3, 1.8, 5.3, 0.45, "Phase 1 · no labels", size=20, bold=True, color=TEAL, font=HEAD)
tb(s, M + 0.3, 2.3, 5.3, 0.7, "Given the last 4 s, pick the real future among many candidates (up to 1 s ahead)",
   size=15)
# timeline mini-illustration
tx, ty = M + 0.35, 3.25
past = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, tx, ty, 3.0, 0.65, fill=TEAL)
shape_text(past, "past 4 s  →  context", size=13, color=WHITE, bold=True)
for k, (lab, fill) in enumerate([("real", OLIVE_LINK), ("✗", GREY), ("✗", GREY), ("✗", GREY)]):
    cbox = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, tx + 3.2 + k * 0.58, ty + 0.05, 0.5, 0.55, fill=fill)
    shape_text(cbox, lab, size=11 if lab == "real" else 14, color=WHITE, bold=True, margin=0)
# Phase 2
p2 = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, M + 6.43, 1.65, 5.9, 2.55, fill=AMBER_SOFT)
tb(s, M + 6.73, 1.8, 5.3, 0.45, "Phase 2 · few labels", size=20, bold=True, color=AMBER, font=HEAD)
tb(s, M + 6.73, 2.3, 5.3, 0.7, "Reuse the learned representation; train a linear head for the 7 activities",
   size=15)
for k in range(7):
    lb = shape(s, MSO_SHAPE.OVAL, M + 6.8 + k * 0.62, 3.3, 0.5, 0.5, fill=AMBER if k < 2 else WHITE,
               line=AMBER, line_w=1.5)
tb(s, M + 6.8, 3.85, 5, 0.3, "e.g. 1–10 labeled windows per activity", size=12, color=MUTED, italic=True)
arrow(s, M + 5.95, 2.92, M + 6.4, 2.92, color=MUTED)
# objectives
tb(s, M, 4.45, 6, 0.4, "Objectives", size=18, bold=True, color=OLIVE_DARK, font=HEAD)
objs = [("Source", "sEMG, motion or both?"), ("Protocol", "CPC vs. supervised on unseen people"),
        ("Sensors", "minimum set that keeps performance"), ("Labels", "CPC vs. supervised with few labels")]
ow = (SW - 2 * M - 3 * 0.25) / 4
for i, (h, d) in enumerate(objs):
    x = M + i * (ow + 0.25)
    b = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, x, 4.95, ow, 1.55, fill=OLIVE_MIST)
    circ = shape(s, MSO_SHAPE.OVAL, x + 0.2, 5.1, 0.5, 0.5, fill=OLIVE)
    shape_text(circ, str(i + 1), size=15, bold=True, color=WHITE, margin=0)
    tb(s, x + 0.85, 5.12, ow - 1.0, 0.45, h, size=17, bold=True, color=OLIVE_DARK, anchor=MSO_ANCHOR.MIDDLE)
    tb(s, x + 0.2, 5.75, ow - 0.4, 0.7, d, size=14)

# =====================================================================
# 8. Architecture
# =====================================================================
s = new_slide("The model: CPC with a causal encoder and a GRU", "3 · Approach", notes=(
    "[Paola · ~1.5 min]\n"
    "Thank you, Isaura. This is the network. A four-second window of ten movement variables enters a causal "
    "encoder — three 1D convolutions padded only on the left, so it can never look at the future. A two-layer "
    "GRU summarizes the history into a context vector. During pretraining, a predictor uses that context to "
    "score the real future against 127 negatives with the InfoNCE loss, for horizons of up to one second. "
    "Three design decisions were key. First, negatives come from other windows of the SAME person: otherwise "
    "the network could solve the task by recognizing who the person is instead of how they move. Second, "
    "for fine-tuning we use LP-FT: first train only the head on the frozen network, then the whole network; "
    "without this the random head deformed the pretrained representation. Third, the whole model is causal, "
    "so it can classify in real time every half second. It has about 725 thousand parameters and trains in "
    "4 to 7 minutes on a laptop."))
y = 1.95
bx = [("Window", "10 variables × 48 steps\n(4 s at 12 Hz)", WHITE, OLIVE_DARK, GREY),
      ("Causal encoder", "3 × Conv1d, left padding\n→ z_t ∈ ℝ¹²⁸", TEAL_SOFT, TEAL, None),
      ("GRU · 2 layers", "summarises the history\n→ c_t ∈ ℝ²⁵⁶", TEAL_SOFT, TEAL, None)]
for i, (h, d, fill, col, line) in enumerate(bx):
    x = M + i * 2.75
    b = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, 2.35, 1.55, fill=fill, line=line)
    shape_text(b, [[(h, {"bold": True, "size": 16, "color": col})], [(d, {"size": 13})]])
    if i:
        arrow(s, x - 0.38, y + 0.78, x - 0.03, y + 0.78, color=MUTED)
# branch
pred = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, M + 8.55, 1.55, 3.75, 1.2, fill=WHITE, line=MUTED, dash=True)
shape_text(pred, [[("Phase 1 · Predictor W₁…W₁₂", {"bold": True, "size": 15, "color": MUTED})],
                  [("InfoNCE: real future vs. 127 negatives (discarded after)", {"size": 12, "color": MUTED})]])
head = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, M + 8.55, 2.95, 3.75, 1.2, fill=AMBER_SOFT)
shape_text(head, [[("Phase 2 · Linear head on c_T", {"bold": True, "size": 15, "color": AMBER})],
                  [("→ 7 activities, trained with LP-FT", {"size": 12})]])
arrow(s, M + 7.85, y + 0.78, M + 8.5, 2.15, color=MUTED, dash=True)
arrow(s, M + 7.85, y + 0.78, M + 8.5, 3.55, color=AMBER)
# design decisions
tb(s, M, 4.45, 8, 0.4, "Three design decisions that mattered", size=18, bold=True, color=OLIVE_DARK, font=HEAD)
dec = [("Same-subject negatives", "the network must learn the movement, not recognize the person", OLIVE_LINK, OLIVE_MIST),
       ("LP-FT fine-tuning", "train the head first, then the whole net — protects the learned representation", AMBER, AMBER_SOFT),
       ("Causal & light", "real-time label every 0.5 s · 725k parameters · 4–7 min per round on a laptop", TEAL, TEAL_SOFT)]
dw = (SW - 2 * M - 2 * 0.3) / 3
for i, (h, d, col, soft) in enumerate(dec):
    x = M + i * (dw + 0.3)
    shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, x, 4.95, dw, 1.6, fill=soft)
    tb(s, x + 0.25, 5.08, dw - 0.5, 0.45, h, size=17, bold=True, color=col)
    tb(s, x + 0.25, 5.55, dw - 0.5, 0.95, d, size=14)

# =====================================================================
# 9. Data and windows
# =====================================================================
s = new_slide("Data: the Manual Material Handling dataset", "4 · Methodology", notes=(
    "[Paola · ~1.5 min]\n"
    "We used the Manual Material Handling dataset published by Bassani, Filippeschi and Avizzano on Zenodo: "
    "14 people, 37 trials each, loads from 2 to 12 kilograms, recorded with an Xsens suit of 17 inertial "
    "sensors at 240 hertz and 8 channels of surface EMG on the upper limb. "
    "We synchronized both systems — their clocks differed by about one hour — using the label events, with a "
    "typical residual of half a millisecond. Signals were low-pass filtered and resampled to 12 hertz, which "
    "performed like 25 hertz at a third of the cost. "
    "Each example is a 4-second window labeled with the activity at its LAST instant, as in the figure: the "
    "task is 'what is happening now, given the last four seconds', which allows real-time use."))
st = [("14", "people", "12 men, 2 women, 25–38 y"), ("37", "trials each", "loads of 2–12 kg"),
      ("17", "inertial sensors", "Xsens suit, 240 Hz"), ("8", "sEMG channels", "upper limb")]
sw_ = (SW - 2 * M - 3 * 0.25) / 4
for i, (v, l, sub) in enumerate(st):
    stat(s, M + i * (sw_ + 0.25), 1.55, sw_, 1.6, v, l, sub, vsize=34,
         fill=[OLIVE_MIST, OLIVE_MIST, TEAL_SOFT, SLATE_SOFT][i], vcolor=[OLIVE_LINK, OLIVE_LINK, TEAL, SLATE][i])
s.shapes.add_picture(str(REPO / "Poster/fig_ventana_en.png"), Inches(M), Inches(3.35), width=Inches(8.4))
bullets(s, M + 8.75, 3.5, 3.6, 3.0, [
    [("4 s windows", {"bold": True}), (" (48 steps at 12 Hz)", {})],
    [("Label = activity at the ", {}), ("last instant", {"bold": True}), (" → real-time task", {})],
    [("One labeled window every 0.5 s", {}), (" (40,944 for subjects 1–12)", {"color": MUTED})],
    [("sEMG and motion synchronized", {}), (" (≈ 0.5 ms residual)", {"color": MUTED})],
], size=15, space=9)

# =====================================================================
# 10. Evaluation protocol
# =====================================================================
s = new_slide("Always evaluated on people the model has never seen", "4 · Methodology", notes=(
    "[Paola · ~1.5 min]\n"
    "Every experiment uses leave-one-subject-out cross-validation over subjects 1 to 12. In each of the 12 "
    "rounds, one person is evaluated, the next person in the list is used for internal validation — early "
    "stopping and hyperparameters — and the remaining ten train the model. So the evaluated person never "
    "takes part in any training decision. Subjects 13 and 14 were held out during the whole development for "
    "a single final test. "
    "Our metric is macro F1 per subject, which weighs the seven activities equally despite the imbalance — "
    "neutral posture is 44 percent of the windows. We report the mean, the spread between people and the "
    "worst person, and compare models paired by subject with the Wilcoxon test and Holm's correction."))
tb(s, M, 1.55, 8, 0.4, "Leave-one-subject-out (LOSO): 12 rounds", size=18, bold=True, color=OLIVE_DARK, font=HEAD)
cell, gp = 0.42, 0.07
x0, y0 = M + 1.35, 2.25
for r in range(4):
    rnd = [0, 1, 2, 11][r]
    yy = y0 + r * (cell + 0.14) if r < 3 else y0 + 3 * (cell + 0.14) + 0.25
    tb(s, M, yy, 1.3, cell, f"Round {rnd + 1}", size=13, color=MUTED, anchor=MSO_ANCHOR.MIDDLE)
    for j in range(14):
        if j >= 12:
            fill, line = WHITE, AMBER
        elif j == rnd:
            fill, line = OLIVE_LINK, None
        elif j == (rnd + 1) % 12:
            fill, line = AMBER_SOFT, AMBER
        else:
            fill, line = SLATE_SOFT, None
        b = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, x0 + j * (cell + gp), yy, cell, cell, fill=fill, line=line,
                  dash=(j >= 12))
        shape_text(b, str(j + 1), size=10, color=WHITE if j == rnd else INK, margin=0)
tb(s, x0 + 4.2, y0 + 3 * (cell + 0.14) - 0.12, 2, 0.35, "⋮", size=18, color=MUTED, align=PP_ALIGN.CENTER)
# legend
ly = 5.35
for i, (fill, line, lab) in enumerate([(OLIVE_LINK, None, "evaluated person"), (AMBER_SOFT, AMBER, "internal validation"),
                                       (SLATE_SOFT, None, "training (10 people)"), (WHITE, AMBER, "held out for final test")]):
    x = M + [0, 2.0, 4.15, 6.4][i]
    shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, x, ly, 0.32, 0.32, fill=fill, line=line, dash=(i == 3))
    tb(s, x + 0.42, ly - 0.03, 1.6, 0.4, lab, size=12, color=INK, anchor=MSO_ANCHOR.MIDDLE)
tb(s, M, 5.95, 8.6, 0.7, "The evaluated person never takes part in any training decision.", size=16,
   italic=True, color=OLIVE_LINK)
# metric card
shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, M + 9.05, 1.55, 3.28, 5.05, fill=OLIVE_MIST)
tb(s, M + 9.3, 1.7, 2.8, 0.45, "How we score", size=18, bold=True, color=OLIVE_DARK, font=HEAD)
bullets(s, M + 9.3, 2.3, 2.85, 4.2, [
    [("Macro F1 per person", {"bold": True}), (": the 7 activities weigh the same (N = 44 % of windows)", {})],
    [("Mean, spread and ", {}), ("worst person", {"bold": True})],
    [("Paired by person: ", {}), ("Wilcoxon + Holm", {"bold": True})],
    [("Chance level: ", {}), ("0.143", {"bold": True})],
], size=14, space=10)

# =====================================================================
# 11. From muscle to movement
# =====================================================================
s = new_slide("First finding: movement carries the information", "4 · Methodology", notes=(
    "[Paola · ~1.5 min]\n"
    "The project started from surface EMG. But with sEMG alone, five very different classical models all "
    "stopped between 0.40 and 0.49 macro F1 — when very different models agree, the limit is in the signal, "
    "not in the model. "
    "Here the same network — an sLSTM with CPC — is trained on the same trials and rounds, changing only the "
    "input. sEMG alone: 0.43. Adding movement: 0.88, better in all 12 people. Movement alone: 0.885 — adding "
    "sEMG to movement did not help measurably. "
    "The reason is physical: sEMG tells how MUCH effort the muscle makes, movement tells WHERE the body is — "
    "and floor versus table is exactly what sEMG cannot see. So we refocused on movement, which also removes "
    "the need to synchronize two systems."))
cd = CategoryChartData()
cd.categories = ["sEMG only", "sEMG + motion", "Motion only"]
cd.add_series("Macro F1", (0.433, 0.880, 0.885))
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(M), Inches(1.6), Inches(7.2), Inches(4.0), cd)
ch = gf.chart
style_chart(ch, legend=False, size=14)
ch.has_title = True
ch.chart_title.text_frame.text = "Same network (CPC-sLSTM), different input · macro F1, LOSO 12 people"
ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(13)
ch.chart_title.text_frame.paragraphs[0].runs[0].font.color.rgb = MUTED
ch.chart_title.text_frame.paragraphs[0].runs[0].font.bold = False
pl = ch.plots[0]
pl.gap_width = 55
pl.has_data_labels = True
pl.data_labels.number_format = "0.000"
pl.data_labels.number_format_is_linked = False
pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
pl.data_labels.font.size = Pt(15)
pl.data_labels.font.bold = True
ser = pl.series[0]
for i, col in enumerate([SLATE, TEAL, OLIVE_LINK]):
    pt = ser.points[i]
    pt.format.fill.solid()
    pt.format.fill.fore_color.rgb = col
quiet_axes(ch, 0, 1.0, 0.2, hide_values=True)
ch.category_axis.reverse_order = True
ch.category_axis.tick_labels.font.size = Pt(15)
tb(s, M, 5.75, 7.2, 0.8, "5 classical models on sEMG all stopped at 0.40–0.49: the limit was the signal, not the model.",
   size=14, italic=True, color=MUTED)
for i, (h, d, col, soft) in enumerate([
    ("sEMG → how much effort", "Separates rest from effort and grows with the load (×3 from 2 to 12 kg)", SLATE, SLATE_SOFT),
    ("Motion → where the body is", "Tells floor from table — what the muscles alone cannot see", OLIVE_LINK, OLIVE_MIST),
]):
    y = 1.7 + i * 1.75
    shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, M + 7.6, y, 4.73, 1.5, fill=soft)
    tb(s, M + 7.85, y + 0.15, 4.3, 0.45, h, size=17, bold=True, color=col)
    tb(s, M + 7.85, y + 0.62, 4.3, 0.8, d, size=14)
c = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, M + 7.6, 5.25, 4.73, 1.3, fill=OLIVE)
shape_text(c, [[("+0.45 F1", {"size": 24, "bold": True, "color": WHITE, "font": HEAD})],
               [("by adding motion · better in 12/12 people (p < 0.001)", {"size": 13, "color": WHITE})]])

# =====================================================================
# 12. Choosing the network
# =====================================================================
s = new_slide("Choosing the context network", "4 · Methodology", notes=(
    "[Paola · ~1.5 min]\n"
    "With the 30 motion variables we compared different model families on the same windows and rounds. "
    "The Random Forest, on five statistics per variable, was the strongest at 0.894. Among networks, the GRU "
    "with CPC and LP-FT was the best: 0.878, with the lowest spread between people and the best worst person, "
    "and it is the fastest to pretrain. The xLSTM cells, sLSTM and mLSTM, were within 0.016 — smaller than the "
    "variation between people — so their two-to-three-times higher cost was not justified. "
    "An important signal: with the CPC-GRU frozen and only a linear probe on top, F1 was already 0.82, "
    "versus 0.63 for the same network without pretraining. The representation learned without labels "
    "already separates the activities."))
cd = CategoryChartData()
models = ["Random Forest", "GRU + CPC (LP-FT)", "sLSTM from scratch", "sLSTM + CPC", "mLSTM from scratch",
          "LSTM from scratch", "1D CNN from scratch"]
vals = (0.894, 0.878, 0.875, 0.871, 0.862, 0.854, 0.801)
cd.categories = models
cd.add_series("Macro F1", vals)
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(M), Inches(1.55), Inches(7.4), Inches(5.0), cd)
ch = gf.chart
style_chart(ch, legend=False, size=13)
ch.has_title = True
ch.chart_title.text_frame.text = "30 motion variables · macro F1, LOSO 12 people"
r_ = ch.chart_title.text_frame.paragraphs[0].runs[0]
r_.font.size = Pt(13); r_.font.color.rgb = MUTED; r_.font.bold = False
pl = ch.plots[0]
pl.gap_width = 45
pl.has_data_labels = True
pl.data_labels.number_format = "0.000"
pl.data_labels.number_format_is_linked = False
pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
pl.data_labels.font.size = Pt(13)
ser = pl.series[0]
for i in range(len(models)):
    pt = ser.points[i]
    pt.format.fill.solid()
    pt.format.fill.fore_color.rgb = SLATE if i == 0 else (OLIVE_LINK if i == 1 else GREY)
quiet_axes(ch, 0.7, 0.95, 0.05, num_fmt="0.00", hide_values=True)
ch.category_axis.reverse_order = True
stat(s, M + 7.8, 1.6, 4.53, 2.3, "0.820 vs 0.631", "frozen CPC-GRU + linear probe",
     "vs. the same network without pretraining: the label-free representation already separates activities",
     vsize=30)
bullets(s, M + 7.8, 4.25, 4.5, 2.4, [
    [("GRU + CPC: ", {"bold": True, "color": OLIVE_LINK}), ("best network, lowest spread, best worst person", {})],
    [("xLSTM cells within 0.016 ", {"bold": True}), ("— less than the variation between people, at 2–3× the cost", {})],
    [("Tuning hyperparameters ", {}), ("did not change the order", {"bold": True})],
], size=14, space=9)

# =====================================================================
# 13. 11 of 17 sensors
# =====================================================================
s = new_slide("11 of 17 sensors are enough", "4 · Methodology", notes=(
    "[Paola · ~1.5 min]\n"
    "A 17-sensor suit is impractical in a plant, so we searched for the minimum. We grouped the 30 variables "
    "by the sensors they depend on and removed groups one at a time — backward elimination — guided by the "
    "Random Forest and confirmed with the CPC-GRU. "
    "The result is a subset we call EML: flexion of elbows, wrists, hips and knees on both sides, plus the "
    "speed of each hand — ten variables measurable with 11 sensors: pelvis and, on each side, upper arm, "
    "forearm, hand, thigh and lower leg. It keeps the performance of all 30 variables with no significant "
    "loss. Elbows are the one thing we cannot remove. "
    "Because Xsens computes angles with the full-body model, we also recomputed them using ONLY the 11 "
    "sensor orientations: correlation of at least 0.997 and no loss for the CPC-GRU, 0.870. This 11-sensor, "
    "10-variable model is our final model. Now Dylan will present the results."))
s.shapes.add_picture(str(FIG / "body_sensors.png"), Inches(M + 0.2), Inches(1.6), height=Inches(4.6))
for i, (fill, line, lab) in enumerate([(OLIVE_LINK, None, "kept (11)"), (WHITE, MUTED, "removed (6)")]):
    x = M + 0.15 + i * 1.6
    shape(s, MSO_SHAPE.OVAL, x, 6.35, 0.3, 0.3, fill=fill, line=line, line_w=2)
    tb(s, x + 0.38, 6.3, 1.2, 0.4, lab, size=13, anchor=MSO_ANCHOR.MIDDLE)
cd = CategoryChartData()
cd.categories = ["All\n17 sensors", "EML\n11 sensors", "Elbows + legs\n9 sensors", "Elbows only\n4 sensors"]
cd.add_series("Random Forest", (0.894, 0.892, 0.860, 0.714))
cd.add_series("CPC-GRU", (0.878, 0.873, 0.861, 0.740))
gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(M + 3.3), Inches(1.55), Inches(5.6), Inches(5.1), cd)
ch = gf.chart
style_chart(ch, legend=True, size=12)
ch.has_title = True
ch.chart_title.text_frame.text = "Backward elimination of sensor groups · macro F1"
r_ = ch.chart_title.text_frame.paragraphs[0].runs[0]
r_.font.size = Pt(13); r_.font.color.rgb = MUTED; r_.font.bold = False
for ser, col, mk in zip(ch.plots[0].series, [SLATE, OLIVE_LINK], [XL_MARKER_STYLE.SQUARE, XL_MARKER_STYLE.CIRCLE]):
    ser.format.line.color.rgb = col
    ser.format.line.width = Pt(3)
    ser.smooth = False
    ser.marker.style = mk
    ser.marker.size = 10
    ser.marker.format.fill.solid()
    ser.marker.format.fill.fore_color.rgb = col
    ser.marker.format.line.color.rgb = WHITE
pl = ch.plots[0]
quiet_axes(ch, 0.65, 0.95, 0.05, num_fmt="0.00")
ch.category_axis.tick_labels.font.size = Pt(12)
stat(s, M + 9.15, 1.6, 3.18, 1.9, "0.873", "CPC-GRU with EML", "vs. 0.878 with all 17 sensors (p = 0.79)",
     vsize=34)
stat(s, M + 9.15, 3.7, 3.18, 1.9, "≥ 0.997", "angle correlation", "recomputed from the 11 sensors only; F1 0.870",
     fill=TEAL_SOFT, vcolor=TEAL, vsize=34)
tb(s, M + 9.15, 5.8, 3.18, 0.85, "Elbows are irreplaceable: without them F1 drops to ≈ 0.81.", size=13,
   italic=True, color=MUTED)

# =====================================================================
# 14. Results with all labels
# =====================================================================
s = new_slide("All labels: near supervised, on par with the reference", "5 · Results", notes=(
    "[Dylan · ~1.5 min]\n"
    "Thank you, Paola. First, the scenario with ALL labels. Over the 12 unseen people, our final CPC-GRU "
    "reaches 0.873 macro F1, close to the Random Forest with the same input, 0.892 — the Random Forest is "
    "consistently a little better here, in 11 of 12 people. This agrees with what Haresamudram and "
    "colleagues reported: with all labels, CPC does not beat supervised learning. "
    "The two people held out during the whole development confirmed the estimate: 0.884 and 0.898, inside "
    "the range predicted by the cross-validation and never below its mean. "
    "To compare with the reference work under conditions close to theirs, we replicated their protocol — "
    "training on 13 people, with their ten signals. On subject 11, the one they report, the CPC-GRU obtained "
    "0.910 versus 0.903 for their BiLSTM, and 0.883 on average over the 14 people. Since that difference is "
    "much smaller than the variation between people, we read it as being at the same level, not above."))
cards = [("Cross-validation", "12 unseen people (LOSO)", "0.873", "± 0.036", "Random Forest 0.892 ± 0.029", OLIVE_LINK, OLIVE_MIST),
         ("Final test", "2 people held out all along", "0.884 · 0.898", "subjects 13 · 14", "Random Forest 0.885 · 0.906", TEAL, TEAL_SOFT),
         ("Reference protocol", "13 people → subject 11", "0.910", "mean of 14 people: 0.883", "BiLSTM (Bassani et al., 2025): 0.903", SLATE, SLATE_SOFT)]
cw3 = (SW - 2 * M - 2 * 0.3) / 3
for i, (h, sub, v, vs, ref, col, soft) in enumerate(cards):
    x = M + i * (cw3 + 0.3)
    shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, x, 1.6, cw3, 3.6, fill=soft)
    tb(s, x + 0.3, 1.75, cw3 - 0.6, 0.45, h, size=20, bold=True, color=col, font=HEAD)
    tb(s, x + 0.3, 2.2, cw3 - 0.6, 0.4, sub, size=14, color=MUTED)
    tb(s, x + 0.3, 2.65, cw3 - 0.6, 0.95, v, size=44 if len(v) < 8 else 34, bold=True, color=OLIVE_DARK, font=HEAD, anchor=MSO_ANCHOR.MIDDLE)
    tb(s, x + 0.3, 3.6, cw3 - 0.6, 0.4, f"CPC-GRU  {vs}", size=14, color=INK)
    rb = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, x + 0.3, 4.2, cw3 - 0.6, 0.7, fill=WHITE)
    shape_text(rb, ref, size=13, color=col, bold=True)
tb(s, M, 5.45, SW - 2 * M, 1.1, [
    [("Reading: ", {"bold": True, "color": OLIVE_LINK}),
     ("with all labels CPC does not beat the supervised model — as Haresamudram et al. (2021) found for daily "
      "activities — but it reaches the level of the reference network with 11 sensors, no sEMG and 12 Hz.", {})]],
   size=16)

# =====================================================================
# 15. Errors
# =====================================================================
s = new_slide("Where the remaining errors are", "5 · Results", notes=(
    "[Dylan · ~1.5 min]\n"
    "Where does the model fail? Accuracy per activity goes from 79 percent for lift-from-table to 96 percent "
    "for walking. Errors concentrate in a few pairs, highlighted: lift and place on table confused with "
    "neutral posture — they start and end in neutral and are the shortest actions — and place-on-table "
    "versus hold, both with the load in front of the body at mid height. "
    "For ergonomics, the costliest error would be confusing a floor-level activity with a table-level one, "
    "because trunk flexion would be underestimated. That confusion is low: at most 4 percent. "
    "So errors are mostly boundary errors between consecutive activities: the model knows WHAT happens but "
    "may shift the change by a fraction of a second. And the ceiling seems to be in the data, not the "
    "classifier: a 15-times larger head or attention pooling did not help, and almost 90 percent of the "
    "4-second windows cross a transition, because actions last 1.6 to 4.4 seconds."))
s.shapes.add_picture(str(FIG / "confusion_final.png"), Inches(M), Inches(1.5), height=Inches(5.15))
items = [("Boundary errors", "LT → N 9 %, PT → N 7 %: short actions that start and end in neutral", SLATE, SLATE_SOFT),
         ("Similar postures", "PT ↔ K ≈ 5 %: load held in front at mid height", TEAL, TEAL_SOFT),
         ("Floor vs. table ≤ 4 %", "the costliest ergonomic error is rare; walking is never taken for standing", OLIVE_LINK, OLIVE_MIST),
         ("Ceiling in the data", "89.6 % of 4 s windows cross a transition; bigger heads or attention did not help", AMBER, AMBER_SOFT)]
for i, (h, d, col, soft) in enumerate(items):
    y = 1.55 + i * 1.3
    shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, M + 6.3, y, 6.03, 1.12, fill=soft)
    tb(s, M + 6.55, y + 0.1, 5.6, 0.4, h, size=17, bold=True, color=col)
    tb(s, M + 6.55, y + 0.5, 5.6, 0.6, d, size=14)

# =====================================================================
# 16. Few labels — main finding
# =====================================================================
s = new_slide("Main finding: with few labels, CPC wins", "5 · Results", notes=(
    "[Dylan · ~2 min]\n"
    "This is the central result. The design was fixed before seeing the results: CPC is pretrained in each "
    "round only with UNLABELED windows of the training people; then we draw 1 to 100 labeled windows per "
    "activity, ten repetitions per size, and every model gets exactly the same windows. Superiority required "
    "a higher mean, p below 0.05 after Holm correction over 21 comparisons, and improvement in at least 9 of "
    "12 people. "
    "With 10 labeled windows per activity or fewer — between 7 and 70 windows in total, less than 0.2 percent "
    "of the training labels — the CPC-GRU beats the Random Forest by 0.045 to 0.107 F1, in 11 or 12 of the 12 "
    "people. To match what CPC achieves with 5 and 10 windows per activity, the Random Forest needs about 11 "
    "and 22: roughly 2.2 times more labels. From 25 windows on, the advantage shrinks and stops being "
    "significant, and with all labels the order reverses."))
cd = CategoryChartData()
cd.categories = ["1", "2", "5", "10", "25", "50", "100"]
cd.add_series("CPC-GRU (pretrained without labels)", (0.368, 0.473, 0.620, 0.707, 0.769, 0.800, 0.827))
cd.add_series("Random Forest (supervised)", (0.323, 0.407, 0.513, 0.611, 0.722, 0.777, 0.816))
gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(M), Inches(1.5), Inches(8.1), Inches(5.15), cd)
ch = gf.chart
style_chart(ch, legend=True, size=13)
ch.has_title = True
ch.chart_title.text_frame.text = "Macro F1 on a new person vs. labeled windows per activity"
r_ = ch.chart_title.text_frame.paragraphs[0].runs[0]
r_.font.size = Pt(13); r_.font.color.rgb = MUTED; r_.font.bold = False
for ser, col, mk in zip(ch.plots[0].series, [OLIVE_LINK, SLATE], [XL_MARKER_STYLE.CIRCLE, XL_MARKER_STYLE.SQUARE]):
    ser.format.line.color.rgb = col
    ser.format.line.width = Pt(3.25)
    ser.smooth = False
    ser.marker.style = mk
    ser.marker.size = 10
    ser.marker.format.fill.solid()
    ser.marker.format.fill.fore_color.rgb = col
    ser.marker.format.line.color.rgb = WHITE
cser = ch.plots[0].series[0]
cser.data_labels.show_value = True
cser.data_labels.number_format = "0.00"
cser.data_labels.number_format_is_linked = False
cser.data_labels.position = XL_LABEL_POSITION.ABOVE
cser.data_labels.font.size = Pt(12)
cser.data_labels.font.bold = True
cser.data_labels.font.color.rgb = OLIVE_LINK
quiet_axes(ch, 0.3, 0.9, 0.1, num_fmt="0.0")
ch.category_axis.has_title = True
ch.category_axis.axis_title.text_frame.text = "Labeled windows per activity"
ch.category_axis.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(12)
ch.category_axis.axis_title.text_frame.paragraphs[0].runs[0].font.bold = False
# significance shading note
tb(s, M + 0.9, 1.95, 3.6, 0.35, "★ significant (Holm) for 1, 2, 5 and 10", size=12, color=AMBER, bold=True)
stat(s, M + 8.45, 1.5, 3.88, 1.6, "+0.11 F1", "over Random Forest", "with 5 labeled windows per activity", vsize=32)
stat(s, M + 8.45, 3.25, 3.88, 1.6, "≈ 2.2×", "fewer labels", "for the same performance (5–10 per activity)",
     fill=AMBER_SOFT, vcolor=AMBER, vsize=32)
stat(s, M + 8.45, 5.0, 3.88, 1.6, "11–12 / 12", "people improve", "with ≤ 10 labeled windows per activity",
     fill=TEAL_SOFT, vcolor=TEAL, vsize=32)

# =====================================================================
# 17. Why — controls
# =====================================================================
s = new_slide("The advantage comes from pretraining", "5 · Results", notes=(
    "[Dylan · ~1.5 min]\n"
    "Is the advantage due to the architecture or to what CPC learns without labels? We used two controls. "
    "The same network trained from scratch with the same 10 windows per activity lands on the Random Forest "
    "curve: 0.615 versus 0.611, while the pretrained CPC-GRU reaches 0.707. And against a frozen random "
    "encoder with the same heads, CPC wins by 0.07 to 0.16 from two windows on, in all 12 people. So the "
    "architecture alone gives no advantage; what CPC learns without labels does. "
    "To exploit it, freezing the network is enough: a linear head, an MLP, or full fine-tuning differ by at "
    "most 0.016. "
    "The gain is not uniform: it concentrates in hold, place-on-table and lift-from-table, activities defined "
    "by how the movement evolves more than by average posture. For walk and lift-from-floor it is a tie, and "
    "for place-on-floor the Random Forest is slightly better."))
cd = CategoryChartData()
cd.categories = ["CPC-GRU (pretrained)", "Same network, from scratch", "Random Forest"]
cd.add_series("Macro F1 @ 10 windows/activity", (0.707, 0.615, 0.611))
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(M), Inches(1.55), Inches(5.9), Inches(4.3), cd)
ch = gf.chart
style_chart(ch, legend=False, size=13)
ch.has_title = True
ch.chart_title.text_frame.text = "Macro F1 with 10 labeled windows per activity"
r_ = ch.chart_title.text_frame.paragraphs[0].runs[0]
r_.font.size = Pt(13); r_.font.color.rgb = MUTED; r_.font.bold = False
pl = ch.plots[0]
pl.gap_width = 60
pl.has_data_labels = True
pl.data_labels.number_format = "0.000"
pl.data_labels.number_format_is_linked = False
pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
pl.data_labels.font.size = Pt(15)
pl.data_labels.font.bold = True
for i, col in enumerate([OLIVE_LINK, GREY, SLATE]):
    pt = pl.series[0].points[i]
    pt.format.fill.solid()
    pt.format.fill.fore_color.rgb = col
quiet_axes(ch, 0.5, 0.75, 0.05, num_fmt="0.00", hide_values=True)
tb(s, M, 5.95, 5.9, 0.7, "vs. a frozen random encoder: CPC wins by 0.07–0.16 from 2 windows, in 12/12 people",
   size=14, italic=True, color=MUTED)
# per-activity gain: diverging bars drawn with shapes (renders identically everywhere)
tb(s, M + 6.3, 1.5, 6.03, 0.4, "Per-activity F1 gain of CPC over RF (10 windows/activity)", size=13, color=MUTED,
   align=PP_ALIGN.CENTER)
acts_ = ["K · hold", "PT · place table", "LT · lift table", "W · walk", "LF · lift floor", "PF · place floor"]
gains = (0.26, 0.22, 0.13, 0.0, 0.0, -0.04)
zero_x, scale = M + 9.05, 11.0  # inches per F1 point
for i, (a, g) in enumerate(zip(acts_, gains)):
    yy = 2.0 + i * 0.52
    tb(s, M + 6.3, yy, 2.2, 0.42, a, size=13, align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)
    if g > 0:
        shape(s, MSO_SHAPE.RECTANGLE, zero_x, yy + 0.04, g * scale, 0.34, fill=OLIVE_LINK)
        tb(s, zero_x + g * scale + 0.08, yy, 0.8, 0.42, f"+{g:.2f}", size=13, bold=True, color=OLIVE_LINK,
           anchor=MSO_ANCHOR.MIDDLE)
    elif g < 0:
        shape(s, MSO_SHAPE.RECTANGLE, zero_x + g * scale, yy + 0.04, -g * scale, 0.34, fill=SLATE)
        tb(s, zero_x + 0.08, yy, 1.5, 0.42, f"−{-g:.2f} (RF better)", size=13, bold=True, color=SLATE,
           anchor=MSO_ANCHOR.MIDDLE)
    else:
        tb(s, zero_x + 0.08, yy, 0.8, 0.42, "tie", size=13, bold=True, color=MUTED, anchor=MSO_ANCHOR.MIDDLE)
ln_ = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(zero_x), Inches(1.95), Inches(zero_x), Inches(5.1))
ln_.line.color.rgb = GREY
ln_.line.width = Pt(1.25)
c = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, M + 6.3, 5.3, 6.03, 1.35, fill=OLIVE_MIST)
shape_text(c, [[("Freezing is enough: ", {"bold": True, "color": OLIVE_LINK}),
                ("linear head, MLP or full fine-tuning differ by ≤ 0.016. The gain lives in activities defined by "
                 "how movement evolves, not by average posture.", {})]], size=14, align=PP_ALIGN.LEFT, margin=0.22)

# =====================================================================
# 18. Discussion vs SoA
# =====================================================================
s = new_slide("What this adds to the state of the art", "6 · Discussion", notes=(
    "[Dylan · ~2 min]\n"
    "Let us place the results against the state of the art. "
    "Haresamudram and colleagues showed, for daily activities with accelerometers, that CPC is on par with "
    "supervised learning with all labels and better with few. We confirm both statements in a new domain — "
    "manual material handling with full-body motion — and we quantify the label efficiency: about 2.2 times "
    "fewer labels, consistently across people. "
    "Bassani and colleagues established the supervised reference on this dataset. Under their protocol, our "
    "model performs at the same level on their test person, 0.910 versus 0.903, while using a label-free "
    "pretraining, 11 inertial sensors without sEMG, and 12 hertz instead of 240. Our evaluation is "
    "complementary to theirs: we report the mean over all people, 0.883, and show how much it varies between "
    "people, from 0.80 to 0.92. Their random 70–30 split answers a different question — how well people "
    "already seen in training are recognized; with that same split our models also rise to 0.90–0.92. "
    "Both views together give a fuller picture of how such a system would behave on a new worker."))
rows = [("What was known", "What we found", "Significance"),
        ("CPC ≈ supervised with all labels; better with few — daily activities, accelerometers (Haresamudram et al., 2021)",
         "Confirmed in manual material handling with full-body motion: 0.873 vs. 0.892 with all labels; +0.05 to +0.11 F1 with ≤ 10 windows",
         "Self-supervision transfers to ergonomics; ≈ 2.2× fewer labels, measured per person"),
        ("Supervised BiLSTM: 0.903 on a held-out person; 8 motion + 2 sEMG at 240 Hz (Bassani et al., 2025)",
         "Same level under their protocol: 0.910 on subject 11; 0.883 mean over all 14 people",
         "Comparable accuracy with label-free pretraining, 11 IMUs, no sEMG, 12 Hz"),
        ("Random 70–30 split of windows: 0.957 (Bassani et al., 2025)",
         "With that split our models also reach 0.90–0.92; across people F1 ranges 0.80–0.92",
         "Complementary views: seen people vs. new people; person-level spread now quantified"),
        ("sEMG + motion as input",
         "Motion alone carries the information (sEMG adds −0.006, n.s.)",
         "Simpler acquisition: one system, no synchronization")]
colx = [M, M + 4.35, M + 8.55]
colw_ = [4.15, 4.0, 3.78]
heads_col = [SLATE, OLIVE_LINK, TEAL]
y = 1.55
for ri, row in enumerate(rows):
    h = 0.5 if ri == 0 else 1.12
    for ci, txt in enumerate(row):
        if ri == 0:
            b = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, colx[ci], y, colw_[ci], h, fill=heads_col[ci])
            shape_text(b, txt, size=16, bold=True, color=WHITE)
        else:
            fill = [SLATE_SOFT, OLIVE_MIST, TEAL_SOFT][ci]
            b = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, colx[ci], y, colw_[ci], h, fill=fill)
            shape_text(b, txt, size=12.5, color=INK, align=PP_ALIGN.LEFT, margin=0.14)
    y += h + 0.1

# =====================================================================
# 19. Significance & limitations
# =====================================================================
s = new_slide("Significance and limitations", "6 · Discussion", notes=(
    "[Dylan · ~1.5 min]\n"
    "In terms of significance: to our knowledge this is the first evaluation of contrastive self-supervised "
    "learning for manual material handling recognition; it shows that labels — the expensive part of bringing "
    "such a system to a plant — can be cut roughly in half for a new person; it identifies a practical "
    "11-sensor set; and it uses a person-independent protocol with paired statistics across 12 people, plus a "
    "held-out test. "
    "We are also clear about the limits. The final test has only two people. Most network comparisons use a "
    "single seed. The population is 14 young adults in a laboratory, only two women. The few-label scenario is "
    "idealized — balanced and drawn from ten people. And the 11 sensors were simulated from the full suit, "
    "so a real reduced kit may be noisier. These limitations define the next steps."))
half = (SW - 2 * M - 0.4) / 2
shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, M, 1.55, half, 5.05, fill=OLIVE_MIST)
hb = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, M, 1.55, half, 0.7, fill=OLIVE_LINK)
shape_text(hb, "Why it matters", size=19, bold=True, color=WHITE, font=HEAD)
bullets(s, M + 0.35, 2.5, half - 0.7, 4.0, [
    [("First evaluation of CPC for material handling", {"bold": True}), (" (to our knowledge)", {"color": MUTED})],
    [("≈ half the labels", {"bold": True}), (" to adapt to a new person, task or plant", {})],
    [("11 inertial sensors", {"bold": True}), (" instead of a 17-sensor suit + sEMG", {})],
    [("Person-independent evidence: ", {}), ("12-fold LOSO, paired tests, held-out final test", {"bold": True})],
    [("Reusable pretrained model", {"bold": True}), (" and reproducible pipeline", {})],
], size=15, space=11)
x2 = M + half + 0.4
shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, x2, 1.55, half, 5.05, fill=AMBER_SOFT)
hb = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, x2, 1.55, half, 0.7, fill=AMBER)
shape_text(hb, "Limitations", size=19, bold=True, color=WHITE, font=HEAD)
bullets(s, x2 + 0.35, 2.5, half - 0.7, 4.0, [
    [("Final test with only ", {}), ("2 people", {"bold": True})],
    [("Most network comparisons with ", {}), ("one seed", {"bold": True}), (" (±0.01 may be noise)", {})],
    [("14 young adults in a laboratory", {"bold": True}), (", only 2 women", {})],
    [("Few-label scenario ", {}), ("idealized", {"bold": True}), (": balanced, drawn from 10 people", {})],
    [("11 sensors ", {}), ("simulated", {"bold": True}), (" from the full suit; a real kit may be noisier", {})],
], size=15, space=11, bullet_color=AMBER)

# =====================================================================
# 20. Conclusions and next steps
# =====================================================================
s = new_slide("Conclusions and next steps", "6 · Discussion", notes=(
    "[Dylan · ~1.5 min]\n"
    "To conclude, the answer to our research question is yes, in both parts. With all labels, CPC comes close "
    "to the supervised model and reaches the level of the reference network; with few labels, it wins. "
    "Three take-home messages: movement, not muscle activity, carries the information to tell these "
    "activities apart; 11 sensors are enough; and with few labels, CPC wins — so CPC pays off exactly when "
    "there is plenty of recorded movement and few annotations: a new person, task or environment. "
    "As next steps, we propose to validate with workers and a real 11-sensor kit, to move from activities to "
    "harmful movements by combining this segmentation with validated ergonomic indices, and to explore "
    "shorter windows of about two seconds and Contrastive Active Inference as an alternative self-supervised "
    "method."))
concl = [("Motion beats sEMG", "0.885 vs. 0.433 with the same network", SLATE),
         ("11 sensors suffice", "10 joint variables, no significant loss", TEAL),
         ("Few labels: CPC wins", "up to +0.11 F1, ≈ 2.2× fewer labels", OLIVE_LINK)]
for i, (h, d, col) in enumerate(concl):
    cx = M + 0.4 + i * 4.15
    circ = shape(s, MSO_SHAPE.OVAL, cx, 1.5, 2.6, 2.6, fill=col)
    shape_text(circ, [[(h, {"size": 21, "bold": True, "color": WHITE, "font": HEAD})]], margin=0.3)
    tb(s, cx - 0.55, 4.2, 3.7, 0.5, d, size=15, color=INK, align=PP_ALIGN.CENTER)
tb(s, M, 4.8, 4, 0.4, "Next steps", size=18, bold=True, color=OLIVE_DARK, font=HEAD)
nx = [("Validate with workers", "and a real 11-sensor kit"), ("From activities", "to harmful movements (ergonomic indices)"),
      ("Shorter windows (≈ 2 s)", "to reduce boundary errors"), ("Contrastive Active Inference", "as an alternative to CPC")]
nw = (SW - 2 * M - 3 * 0.4) / 4
for i, (h, d) in enumerate(nx):
    x = M + i * (nw + 0.4)
    b = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, x, 5.3, nw, 1.25, fill=[OLIVE_MIST, OLIVE_SOFT, OLIVE_MIST, OLIVE_SOFT][i])
    shape_text(b, [[(h, {"bold": True, "size": 15, "color": OLIVE_DARK})], [(d, {"size": 13})]], margin=0.15)
    if i:
        arrow(s, x - 0.36, 5.92, x - 0.05, 5.92, color=OLIVE)

# =====================================================================
# 21. Thank you + references
# =====================================================================
s = new_slide(None, notes=(
    "[All · Q&A]\n"
    "Thank you very much for your attention and for your support of this project. We are happy to take your "
    "questions. (Backup material: confusion matrices by source, protocol comparison and per-activity results "
    "on subject 11 are in the article appendices.)"))
s.shapes.add_picture(str(REPO / "ResumenEjecutivo/logo.jpeg"), Inches(M + 0.1), Inches(0.45), height=Inches(0.85))
s.shapes.add_picture(str(REPO / "ResumenEjecutivo/sssa.png"), Inches(SW - M - 0.1 - 3.05), Inches(0.5),
                     height=Inches(0.85))
band = shape(s, MSO_SHAPE.RECTANGLE, M, 1.75, SW - 2 * M, 2.2, fill=OLIVE)
shape_text(band, [[("Thank you · Grazie", {"size": 44, "color": WHITE, "font": HEAD})],
                  [("Questions?", {"size": 22, "color": OLIVE_SOFT, "italic": True})]])
tb(s, M, 4.25, 4, 0.4, "References", size=16, bold=True, color=OLIVE_DARK, font=HEAD)
refs = [
    "Bassani, G., Avizzano, C. A., & Filippeschi, A. (2025). Deep learning algorithms for human activity recognition in manual material handling tasks. Sensors, 25(21), 6705.",
    "Bassani, G., Filippeschi, A., & Avizzano, C. A. (2021). Manual material handling dataset for biomechanical and ergonomics analysis [Data set]. Zenodo.",
    "Haresamudram, H., Essa, I., & Plötz, T. (2021). Contrastive predictive coding for human activity recognition. Proc. ACM IMWUT, 5(2), 1–26.",
    "van den Oord, A., Li, Y., & Vinyals, O. (2018). Representation learning with contrastive predictive coding. arXiv:1807.03748.",
    "Kumar, A., et al. (2022). Fine-tuning can distort pretrained features and underperform out-of-distribution. ICLR.",
]
tb(s, M, 4.75, SW - 2 * M, 2.3, [[(r, {"size": 13, "color": MUTED, "space_after": 3})] for r in refs])

out = HERE / "CPC_MMH_sponsors.pptx"
prs.save(out)
print("saved", out, "slides:", len(prs.slides))
