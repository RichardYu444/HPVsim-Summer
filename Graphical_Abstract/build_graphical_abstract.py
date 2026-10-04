"""
build_graphical_abstract.py
===========================

Builds HPVsim_graphical_abstract.pptx by driving PowerPoint (Windows, via pywin32), following
Elsevier's graphical-abstract guidance: 2.5:1 (500 x 200) aspect, read left to right
(context -> method -> outcome), Arial, large enough to survive a 200-px-high thumbnail.
Colours follow the 2026 M4P poster (2026M4P_NetworksHPVsim_poster_retitle.pptx): purple
773F9B (theme accent 6) and its 40%-lighter tint AF82CC for the header pills.

Slides
------
1  Draft A  three panels (Elsevier's own template structure); the findings panel holds two
            findings: a native column chart and a stats card
2  Draft B  methods | network comparison | the same two findings, with title and conclusion bands
3  Blank    the slide-1 frame with bracketed placeholders, for reuse

Inputs: icons/*.svg (Health Icons, CC0, healthicons.org), img/net_*.png
(make_network_schematics.py; run that first if img/ is empty) and img/oxford_logo.png (the
University of Oxford mark, cropped from the Mathematical Institute logo on the poster).

Usage
-----
    python Graphical_Abstract/build_graphical_abstract.py
"""
import math
import pathlib
import time

import pywintypes
import win32com.client

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / 'HPVsim_graphical_abstract.pptx'
ICONS = HERE / 'icons'
IMG = HERE / 'img'
PREVIEWS = HERE / 'previews'

# 14 x 5.6 in = 2.5:1. PowerPoint's default 96-dpi picture export then gives 1344 x 538 px,
# above Elsevier's 1328 x 531 minimum.
W, H = 14 * 72, 5.6 * 72
M = 24  # outer margin, pt
LOGO = 46  # Oxford logo side, pt

TITLE = 'Importance of Network Structure in ABMs of STIs: an HPVsim Study'
FONT = 'Arial'

PURPLE = '773F9B'      # poster accent 6: titles, headings, key numbers
LILAC = 'AF82CC'       # accent 6, 40% lighter: the poster's section-header pills
LILAC_LIGHT = 'E4D5EE'  # accent 6, 80% lighter
LILAC_TINT = 'F3ECF8'  # panel fill
INK = '2B2233'
MUTED = '6E6378'
GRID = 'E2D9EA'
WHITE = 'FFFFFF'
ORANGE = 'EB6834'      # women in the network plots (network.py)
BLUE = '2A78D6'        # men
TARGET_RED = 'C0392B'

# Draft numbers: median cervical cancer incidence per 100,000 women in 2040, 50 runs each,
# results setting B (mean degree 1.4). Replace with the final values; STATS are each network
# against the default.
CHART_DATA = [('Default', 6.9, '9C94A6'), ('Gamma', 13.5, PURPLE), ('Power law', 9.0, LILAC)]
STATS = [('+96%', 'Gamma'), ('+30%', 'Power law')]
UK_TARGET = 4

# Office enums
MSO_TRUE, MSO_FALSE = -1, 0
RECT, ROUNDED, OVAL, CHEVRON = 1, 5, 9, 52
ALIGN = {'l': 1, 'c': 2, 'r': 3}
ANCHOR = {'t': 1, 'm': 3, 'b': 4}
XL_COLUMN_CLUSTERED, XL_LINE = 51, 4
XL_CATEGORY, XL_VALUE = 1, 2


def rgb(hex6):
    """Office COM colours are BGR integers."""
    r, g, b = (int(hex6[i:i + 2], 16) for i in (0, 2, 4))
    return r + g * 256 + b * 65536


# ---------------------------------------------------------------------------------------------
# drawing helpers
# ---------------------------------------------------------------------------------------------

def box(slide, x, y, w, h, fill, name, shape=RECT, radius=None, line=None, dash=False):
    s = slide.Shapes.AddShape(shape, x, y, w, h)
    s.Name = name
    if fill is None:
        s.Fill.Visible = MSO_FALSE
    else:
        s.Fill.ForeColor.RGB = rgb(fill)
    if line is None:
        s.Line.Visible = MSO_FALSE
    else:
        s.Line.ForeColor.RGB = rgb(line)
        s.Line.Weight = 1.25
        if dash:
            s.Line.DashStyle = 4  # msoLineDash
    if radius is not None:
        s.Adjustments.SetItem(1, radius)
    s.Shadow.Visible = MSO_FALSE
    s.TextFrame.TextRange.Text = ''
    return s


def text(slide, x, y, w, h, runs, name, size=14, color=INK, bold=False, italic=False,
         align='l', anchor='t', space_after=0, shape=None):
    """Text box. `runs` is a string, or a list of paragraphs, each a string or a list of
    (text, {bold, color, size, italic}) runs. Pass `shape` to write into an existing shape."""
    new = shape is None
    if new:
        shape = slide.Shapes.AddTextbox(1, x, y, w, h)
        shape.Name = name
    tf = shape.TextFrame
    tf.WordWrap = MSO_TRUE
    tf.AutoSize = 0
    if new:  # a new text box shrinks to one line before AutoSize is switched off
        shape.Top, shape.Height = y, h
    tf.MarginLeft = tf.MarginRight = tf.MarginTop = tf.MarginBottom = 0
    tf.VerticalAnchor = ANCHOR[anchor]

    paragraphs = [runs] if isinstance(runs, str) else runs
    paragraphs = [[(p, {})] if isinstance(p, str) else p for p in paragraphs]
    full = '\r'.join(''.join(t for t, _ in p) for p in paragraphs)
    tr = tf.TextRange
    tr.Text = full
    tr.Font.Name = FONT
    tr.Font.Size = size
    tr.Font.Bold = MSO_TRUE if bold else MSO_FALSE
    tr.Font.Italic = MSO_TRUE if italic else MSO_FALSE
    tr.Font.Color.RGB = rgb(color)
    tr.ParagraphFormat.Alignment = ALIGN[align]
    tr.ParagraphFormat.SpaceAfter = space_after
    tr.ParagraphFormat.SpaceBefore = 0

    pos = 1  # COM character positions are 1-based; paragraphs joined by one '\r'
    for p in paragraphs:
        for t, fmt in p:
            if fmt and t:
                c = tr.Characters(pos, len(t))
                if 'bold' in fmt:
                    c.Font.Bold = MSO_TRUE if fmt['bold'] else MSO_FALSE
                if 'italic' in fmt:
                    c.Font.Italic = MSO_TRUE if fmt['italic'] else MSO_FALSE
                if 'color' in fmt:
                    c.Font.Color.RGB = rgb(fmt['color'])
                if 'size' in fmt:
                    c.Font.Size = fmt['size']
            pos += len(t)
        pos += 1
    return shape


def pill(slide, x, y, label, name, size=15):
    """The poster's section header: white bold text on a light-purple pill sized to the text."""
    s = box(slide, x, y, 100, 28, LILAC, name, shape=ROUNDED, radius=0.5)
    text(slide, 0, 0, 0, 0, label, name, size=size, color=WHITE, bold=True, align='c',
         anchor='m', shape=s)
    tf = s.TextFrame
    tf.MarginLeft = tf.MarginRight = 12
    tf.MarginTop = tf.MarginBottom = 4
    tf.WordWrap = MSO_FALSE
    tf.AutoSize = 1  # ppAutoSizeShapeToFitText: sets the width to the label...
    tf.AutoSize = 0  # ...then fix the height, which auto-size would make too tall
    s.Left, s.Top, s.Height = x, y, 28
    return s


def icon(slide, stem, x, y, size, color, name):
    """Insert a Health Icons SVG recoloured to `color` (the originals use currentColor)."""
    out = IMG / 'icons' / f'{stem}_{color}.svg'
    out.parent.mkdir(parents=True, exist_ok=True)
    svg = (ICONS / f'{stem}.svg').read_text(encoding='utf-8')
    out.write_text(svg.replace('currentColor', f'#{color}'), encoding='utf-8')
    s = slide.Shapes.AddPicture(str(out), MSO_FALSE, MSO_TRUE, x, y, size, size)
    s.Name = name
    return s


def hpv_icon(slide, x, y, size, color, light, name):
    """A non-enveloped, knobbly capsid built from native circles, as HPV looks under EM. The
    Health Icons virus is spiked like a coronavirus, which HPV is not."""
    r = size / 2
    cx, cy = x + r, y + r
    body_r, bump_d, inner_d = 0.8 * r, 0.22 * size, 0.18 * size
    names = []
    for k in range(12):
        a = 2 * math.pi * k / 12
        names.append(box(slide, cx + body_r * math.cos(a) - bump_d / 2,
                         cy + body_r * math.sin(a) - bump_d / 2, bump_d, bump_d, color,
                         f'{name} rim {k + 1}', shape=OVAL).Name)
    names.append(box(slide, cx - body_r, cy - body_r, 2 * body_r, 2 * body_r, color,
                     f'{name} body', shape=OVAL).Name)
    centres = [(0, 0)] + [(0.5 * body_r * math.cos(2 * math.pi * k / 6 + math.pi / 6),
                           0.5 * body_r * math.sin(2 * math.pi * k / 6 + math.pi / 6))
                          for k in range(6)]
    for k, (dx, dy) in enumerate(centres):
        names.append(box(slide, cx + dx - inner_d / 2, cy + dy - inner_d / 2, inner_d, inner_d,
                         light, f'{name} capsomere {k + 1}', shape=OVAL).Name)
    return group(slide, names, name)


def picture(slide, path, x, y, w, h, name):
    s = slide.Shapes.AddPicture(str(path), MSO_FALSE, MSO_TRUE, x, y, w, h)
    s.Name = name
    return s


def oxford_logo(slide, y):
    return picture(slide, IMG / 'oxford_logo.png', W - M - LOGO, y, LOGO, LOGO,
                   'University of Oxford logo')


def arrow(slide, x1, y1, x2, y2, color, name, weight=1.75):
    s = slide.Shapes.AddLine(x1, y1, x2, y2)
    s.Name = name
    s.Line.ForeColor.RGB = rgb(color)
    s.Line.Weight = weight
    s.Line.EndArrowheadStyle = 2  # msoArrowheadTriangle
    return s


def wait_for(ready, timeout=15):
    start = time.time()
    while time.time() - start < timeout:
        try:
            if ready():
                return
        except pywintypes.com_error:
            pass
        time.sleep(0.25)
    raise TimeoutError('PowerPoint chart data did not refresh')


def notes(slide, body):
    slide.NotesPage.Shapes.Placeholders(2).TextFrame.TextRange.Text = body


def group(slide, names, name):
    g = slide.Shapes.Range(names).Group()
    g.Name = name
    return g


def chart_with_data(slide, x, y, w, h):
    """Column chart holding CHART_DATA plus a constant UK-target series."""
    shp = slide.Shapes.AddChart2(-1, XL_COLUMN_CLUSTERED, x, y, w, h, True)
    chart = shp.Chart
    chart.ChartData.Activate()
    wb = chart.ChartData.Workbook
    try:
        wb.Application.Visible = False
        ws = wb.Worksheets(1)
        ws.Range('A1:D5').ClearContents()
        rows = [['', 'Incidence 2040', f'UK target ({UK_TARGET})']]
        rows += [[label, value, UK_TARGET] for label, value, _ in CHART_DATA]
        for r, row in enumerate(rows, start=1):
            for c, v in enumerate(row, start=1):
                ws.Cells(r, c).Value = v
        if ws.ListObjects.Count:  # older chart sheets hold the data in a table
            ws.ListObjects(1).Resize(ws.Range(f'A1:C{len(rows)}'))
        chart.SetSourceData(f"='{ws.Name}'!$A$1:$C${len(rows)}", 2)  # xlColumns
        wait_for(lambda: chart.SeriesCollection().Count == 2
                 and chart.SeriesCollection(1).Points().Count == len(CHART_DATA), timeout=5)
    finally:
        wb.Close()
    return shp


def cancer_chart(slide, x, y, w, h, name):
    """Native, editable column chart (right-click > Edit Data) with the UK target as a line.
    No chart title: the card heading above it says what is plotted."""
    # Now and then PowerPoint leaves the series unlinked from the data sheet
    # (=SERIES("Series1",,,1)); starting the chart again fixes it.
    for attempt in range(4):
        try:
            shp = chart_with_data(slide, x, y, w, h)
            break
        except TimeoutError:
            slide.Shapes(slide.Shapes.Count).Delete()
    else:
        raise TimeoutError('PowerPoint chart data did not link after 4 attempts')
    shp.Name = name
    chart = shp.Chart

    chart.HasTitle = False
    chart.ChartArea.Format.Fill.Visible = MSO_FALSE
    chart.ChartArea.Format.Line.Visible = MSO_FALSE
    chart.PlotArea.Format.Fill.Visible = MSO_FALSE
    chart.ChartArea.Format.TextFrame2.TextRange.Font.Name = FONT

    bars = chart.SeriesCollection(1)
    for i, (_, _, colour) in enumerate(CHART_DATA, start=1):
        bars.Points(i).Format.Fill.ForeColor.RGB = rgb(colour)
    bars.HasDataLabels = True
    labels = bars.DataLabels()
    labels.NumberFormat = '0.0'
    labels.Position = 2  # xlLabelPositionOutsideEnd
    lf = labels.Format.TextFrame2.TextRange.Font
    lf.Size = 13
    lf.Bold = MSO_TRUE
    lf.Fill.ForeColor.RGB = rgb(INK)
    chart.ChartGroups(1).GapWidth = 55

    target = chart.SeriesCollection(2)
    target.ChartType = XL_LINE
    target.Format.Line.ForeColor.RGB = rgb(TARGET_RED)
    target.Format.Line.Weight = 2
    target.Format.Line.DashStyle = 4  # msoLineDash
    target.MarkerStyle = -4142  # xlMarkerStyleNone

    # Legend keyed to the target line only: every bar is above 4, so a label on the line would
    # sit on a bar.
    chart.HasLegend = True
    chart.Legend.Position = -4160  # xlLegendPositionTop
    chart.Legend.LegendEntries(1).Delete()
    lg = chart.Legend.Format.TextFrame2.TextRange.Font
    lg.Size = 10.5
    lg.Bold = MSO_TRUE
    lg.Fill.ForeColor.RGB = rgb(TARGET_RED)

    vax = chart.Axes(XL_VALUE)
    vax.MinimumScale = 0
    vax.MaximumScale = 16
    vax.MajorUnit = 4
    vax.MajorGridlines.Format.Line.ForeColor.RGB = rgb(GRID)
    vax.Format.Line.Visible = MSO_FALSE
    vf = vax.TickLabels.Font
    vf.Size = 10
    vf.Color = rgb(MUTED)
    cax = chart.Axes(XL_CATEGORY)
    cax.Format.Line.ForeColor.RGB = rgb(GRID)
    cf = cax.TickLabels.Font
    cf.Size = 11
    cf.Bold = True
    cf.Color = rgb(INK)
    return shp


def card_heading(slide, x, y, w, tag, heading, name):
    text(slide, x, y, w, 30, [[(tag, {'color': PURPLE, 'size': 9.5})], [(heading, {})]], name,
         size=11, bold=True)


def findings_cards(slide, x, y, w, h, stats_w=122):
    """Two white cards: finding 1, the chart; finding 2, the change against the default."""
    aw = w - stats_w - 8
    box(slide, x, y, aw, h, WHITE, 'Finding 1 card', shape=ROUNDED, radius=0.05)
    card_heading(slide, x + 10, y + 7, aw - 20, 'FINDING 1',
                 'Cervical cancer per 100,000 women, 2040', 'Finding 1 heading')
    cancer_chart(slide, x + 4, y + 40, aw - 8, h - 44, 'Cancer chart')

    bx = x + aw + 8
    box(slide, bx, y, stats_w, h, WHITE, 'Finding 2 card', shape=ROUNDED, radius=0.07)
    card_heading(slide, bx + 10, y + 7, stats_w - 20, 'FINDING 2', 'Change vs default',
                 'Finding 2 heading')
    step = (h - 40 - 24) / len(STATS)
    for k, (value, network) in enumerate(STATS):
        sy = y + 44 + k * step
        text(slide, bx + 10, sy, stats_w - 20, 36, value, f'Stat {k + 1}', size=30,
             color=PURPLE, bold=True)
        text(slide, bx + 10, sy + 36, stats_w - 20, 16, network, f'Stat {k + 1} network',
             size=12, bold=True)
    text(slide, bx + 10, y + h - 22, stats_w - 20, 14, '2040 medians, 50 runs',
         'Stats footnote', size=9.5, color=MUTED)


# ---------------------------------------------------------------------------------------------
# slides
# ---------------------------------------------------------------------------------------------

PY, PH = 58, 272                       # panel top and height (slides 1 and 3)
PWS = [232, 288, 395]                  # context | method | findings (two cards)
GAP = (W - 2 * M - sum(PWS)) / 2
PX = [M, M + PWS[0] + GAP, M + PWS[0] + PWS[1] + 2 * GAP]
BAND_Y, BAND_H = PY + PH + 8, 34       # conclusion band
FOOT_Y = BAND_Y + BAND_H + 6

ELSEVIER_NOTES = (
    'Elsevier graphical abstract checklist '
    '(elsevier.com/en-gb/researcher/author/tools-and-resources/graphical-abstract):\n'
    '- Minimum 1328 x 531 px (w x h), or proportionally more, keeping the 500 x 200 (2.5:1) ratio. '
    'This slide is 14 x 5.6 in, so even a default 96-dpi export is 1344 x 538 px.\n'
    '- Preferred files: TIFF, EPS, PDF or MS Office. Safest: File > Export > Create PDF (vector), '
    'exporting only the chosen slide.\n'
    '- Fonts: Times, Arial, Courier or Symbol, large enough to read when shrunk to 200 px high.\n'
    '- Clear start and end, reading left to right or top to bottom. No "Graphical abstract" heading, '
    'no unnecessary white space, no extra text outside the image.\n'
    '- Icons: Health Icons (healthicons.org), CC0, no credit required. Oxford logo: cropped from '
    'the Mathematical Institute logo on the M4P poster (212 px; swap in an official higher-'
    'resolution file if you have one). Network schematics: make_network_schematics.py '
    '(illustrations, not model output).'
)

DATA_NOTES = (
    'Findings: draft values are setting B (mean degree 1.4) medians of 50 runs for cervical cancer '
    'incidence per 100,000 women in 2040: Default 6.9, Gamma (shape 2) 13.5 (+96%), Power law '
    '(alpha 3.5) 9.0 (+30%). Note the Gamma caveat in results_figures.tex (run 1900-2050, fitted '
    'differently). Chart: right-click > Edit Data. Update the stats card and any "2x" wording to '
    'match the final values.'
)


def add_title_and_panels(slide, title, headers):
    text(slide, M, 8, W - 2 * M - LOGO - 16, 44, title, 'Title', size=24, color=PURPLE,
         bold=True, anchor='m')
    oxford_logo(slide, 6)
    for i, (x, w, header) in enumerate(zip(PX, PWS, headers), start=1):
        box(slide, x, PY, w, PH, LILAC_TINT, f'Panel {i}', shape=ROUNDED, radius=0.04)
        pill(slide, x + 12, PY + 10, f'{i}  {header}', f'Panel {i} heading')
    for i in range(2):
        cx = PX[i] + PWS[i] + GAP / 2
        s = box(slide, cx - 6, PY + PH / 2 - 13, 12, 26, LILAC, f'Arrow {i + 1}', shape=CHEVRON)
        s.Adjustments.SetItem(1, 0.55)


def add_band_and_footer(slide, label, message, footer):
    band = box(slide, M, BAND_Y, W - 2 * M, BAND_H, PURPLE, 'Conclusion band', shape=ROUNDED,
               radius=0.2)
    text(slide, M + 16, BAND_Y, W - 2 * M - 32, BAND_H,
         [[(label, {'bold': True, 'color': LILAC_LIGHT}), (message, {})]],
         'Conclusion text', size=14, color=WHITE, anchor='m')
    text(slide, M, FOOT_Y, W - 2 * M, 16, footer, 'Footer', size=10, color=MUTED)
    return band


def network_legend_runs(sep='   '):
    return [('●', {'color': ORANGE}), (f' women{sep}', {}), ('■', {'color': BLUE}),
            (f' men{sep}', {})]


def slide_draft_a(pres):
    slide = pres.Slides.Add(1, 12)  # ppLayoutBlank
    add_title_and_panels(slide, TITLE, ['The problem', 'Our approach', 'What we found'])

    # 1 - context
    x, w = PX[0], PWS[0]
    cx = x + w / 2
    icon(slide, 'i-groups_perspective_crowd', cx - 70, PY + 50, 62, PURPLE, 'Icon people')
    hpv_icon(slide, cx + 8, PY + 50, 62, PURPLE, LILAC_LIGHT, 'Icon HPV')
    text(slide, x + 16, PY + 128, w - 32, 136, [
        [('Most STI models assume everyone has a ', {}), ('similar number', {'bold': True}),
         (' of partners.', {})],
        [('Real behaviour (Natsal-3) is ', {}),
         ('highly skewed', {'bold': True, 'color': PURPLE}),
         (': a few people have many partners, many have none.', {})],
    ], 'Panel 1 text', size=13, space_after=8)

    # 2 - method
    x, w = PX[1], PWS[1]
    icon(slide, 'laptop', x + 14, PY + 50, 34, PURPLE, 'Icon model')
    text(slide, x + 56, PY + 46, w - 70, 42,
         [[('HPVsim', {'bold': True}),
           (' agent-based model (ABM) on three partnership networks', {})]],
         'Model text', size=13, anchor='m')
    s = 84
    gap = (w - 28 - 3 * s) / 2
    for j, (stem, label) in enumerate([('default', 'Default'), ('gamma', 'Gamma'),
                                       ('powerlaw', 'Power law')]):
        nx_ = x + 14 + j * (s + gap)
        picture(slide, IMG / f'net_{stem}.png', nx_, PY + 96, s, s, f'Network {label}')
        text(slide, nx_ - 8, PY + 182, s + 16, 18, label, f'Label {label}', size=12, bold=True,
             align='c')
    arrow(slide, x + 28, PY + 208, x + w - 28, PY + 208, PURPLE, 'Heterogeneity arrow')
    text(slide, x + 14, PY + 211, w - 28, 16, 'increasing heterogeneity', 'Heterogeneity label',
         size=11, color=MUTED, italic=True, align='c')
    text(slide, x + 14, PY + 234, w - 28, 32, [
        [('Same mean: 1.4 partners/year (Natsal-3)', {'bold': True})],
        network_legend_runs() + [('size = no. of partners', {})],
    ], 'Network legend', size=11, align='c', space_after=2)

    # 3 - two findings
    x, w = PX[2], PWS[2]
    icon(slide, 'cervical-cancer', x + w - 42, PY + 9, 30, PURPLE, 'Icon cervical cancer')
    findings_cards(slide, x + 12, PY + 46, w - 24, 184)
    text(slide, x + 14, PY + 236, w - 28, 32,
         [[('Same average behaviour, ', {}),
           ('up to 2× more cervical cancer', {'bold': True, 'color': PURPLE}),
           (' projected on heterogeneous networks', {})]],
         'Findings summary', size=12.5)

    add_band_and_footer(
        slide, 'Take-home: ',
        'network structure, not just average partner numbers, shapes STI projections and '
        'intervention decisions.',
        'Yu R, et al.  ·  [Journal], [Year]  ·  doi:[10.xxxx/xxxxx]')
    notes(slide,
          'DRAFT A: three panels, following the structure of Elsevier\'s own template '
          '(context -> method -> outcome), in the M4P poster colours.\n\n'
          + DATA_NOTES + '\n\n' + ELSEVIER_NOTES)
    return slide


def slide_draft_b(pres):
    slide = pres.Slides.Add(2, 12)
    band_h = 54
    box(slide, 0, 0, W, band_h, LILAC, 'Title band')
    text(slide, M, 0, W - 2 * M - LOGO - 16, band_h, TITLE, 'Title', size=24, color=WHITE,
         bold=True, anchor='m')
    oxford_logo(slide, (band_h - LOGO) / 2)
    bottom_y = 342
    col_w = 254  # leaves the findings cards wide enough for one-line headings
    box(slide, 0, band_h, col_w, bottom_y - band_h, LILAC_TINT, 'Methods column')

    # methods
    text(slide, M, band_h + 12, col_w - 2 * M, 24, 'Methods', 'Methods heading', size=16,
         color=PURPLE, bold=True)
    rows = [
        ('laptop', [('HPVsim', {'bold': True}),
                    (' agent-based model (ABM) with 100,000 agents', {})]),
        ('i-groups_perspective_crowd', [('Partnership networks fitted to ', {}),
                                        ('Natsal-3', {'bold': True}),
                                        (' (1.4 partners/year)', {})]),
        ('syringe-vaccine', [('NHS cervical screening and HPV vaccination', {})]),
    ]
    for k, (stem, runs) in enumerate(rows):
        ry = band_h + 52 + k * 76
        icon(slide, stem, M, ry, 38, PURPLE, f'Methods icon {k + 1}')
        text(slide, M + 50, ry - 8, col_w - M - 50 - 12, 54, [runs], f'Methods text {k + 1}',
             size=12.5, anchor='m')

    # network comparison
    cx0, cw = col_w + 16, 330
    text(slide, cx0, band_h + 12, cw, 24, 'Same mean, different spread', 'Networks heading',
         size=16, color=PURPLE, bold=True, align='c')
    s = 100
    gap = (cw - 3 * s) / 2
    for j, (stem, label, sub) in enumerate([('default', 'Default', 'similar for all'),
                                            ('gamma', 'Gamma', 'some hubs'),
                                            ('powerlaw', 'Power law', 'a few big hubs')]):
        nx_ = cx0 + j * (s + gap)
        picture(slide, IMG / f'net_{stem}.png', nx_, band_h + 44, s, s, f'Network {label}')
        text(slide, nx_ - 10, band_h + 152, s + 20, 18, label, f'Label {label}', size=14,
             bold=True, align='c')
        text(slide, nx_ - 10, band_h + 172, s + 20, 16, sub, f'Sublabel {label}', size=11,
             color=MUTED, align='c')
    arrow(slide, cx0 + 20, band_h + 202, cx0 + cw - 20, band_h + 202, PURPLE,
          'Heterogeneity arrow')
    text(slide, cx0, band_h + 206, cw, 16, 'increasing heterogeneity', 'Heterogeneity label',
         size=11, color=MUTED, italic=True, align='c')
    text(slide, cx0, band_h + 238, cw, 16,
         [network_legend_runs('    ') + [('node size = number of partners', {})]],
         'Network legend', size=11, align='c')

    # findings: the findings area gets its own tinted backdrop so the white cards stand out
    fx = cx0 + cw + 16
    fw = W - fx - M
    box(slide, fx - 8, band_h + 8, fw + 16, bottom_y - band_h - 16, LILAC_TINT,
        'Findings backdrop', shape=ROUNDED, radius=0.04)
    text(slide, fx + 4, band_h + 14, fw - 44, 24, 'Findings', 'Findings heading', size=16,
         color=PURPLE, bold=True)
    icon(slide, 'cervical-cancer', fx + fw - 32, band_h + 12, 28, PURPLE, 'Icon cervical cancer')
    findings_cards(slide, fx, band_h + 44, fw, bottom_y - band_h - 60)

    # conclusion band
    box(slide, 0, bottom_y, W, H - bottom_y, PURPLE, 'Conclusion band')
    text(slide, M, bottom_y, 640, H - bottom_y,
         [[('Conclusion: ', {'bold': True, 'color': LILAC_LIGHT}),
           ('STI models that assume homogeneous partnership networks can misjudge disease burden '
            'and intervention impact; fit network structure to behavioural data.', {})]],
         'Conclusion text', size=13, color=WHITE, anchor='m')
    text(slide, 700, bottom_y, W - 700 - M, H - bottom_y,
         ['Yu R, et al.  [Journal], [Year]', 'doi:[10.xxxx/xxxxx]'],
         'Reference', size=10, color=LILAC_LIGHT, align='r', anchor='m')
    notes(slide,
          'DRAFT B: methods | network comparison | findings, with a title band and conclusion band '
          '(closest to the Kidney Medicine example on the Elsevier page), in the M4P poster '
          'colours.\n\n' + DATA_NOTES + '\n\n' + ELSEVIER_NOTES)
    return slide


def slide_blank(pres):
    slide = pres.Slides.Add(3, 12)
    add_title_and_panels(slide, '[Title: one line, Arial 24 pt]',
                         ['Context', 'Method', 'Findings'])

    def placeholder(x, y, w, h, label, name):
        ph = box(slide, x, y, w, h, None, name, shape=ROUNDED, radius=0.06, line=LILAC,
                 dash=True)
        text(slide, 0, 0, 0, 0, label, name, size=12, color=MUTED, italic=True, align='c',
             anchor='m', shape=ph)

    prompts = ['[Introduce the context of your research]',
               '[Showcase your methodology]',
               '[One-line summary of the two findings]']
    for i, (x, w, prompt) in enumerate(zip(PX, PWS, prompts), start=1):
        if i < 3:
            placeholder(x + 16, PY + 50, w - 32, 150, '[icon, network or diagram]',
                        f'Image placeholder {i}')
        text(slide, x + 16, PY + 212 if i < 3 else PY + 236, w - 32, 50, prompt,
             f'Panel {i} text', size=13)
    x, w = PX[2], PWS[2]
    stats_w = 122
    placeholder(x + 12, PY + 46, w - 24 - stats_w - 8, 184,
                '[Finding 1: graphical comparison]', 'Finding 1 placeholder')
    placeholder(x + w - 12 - stats_w, PY + 46, stats_w, 184, '[Finding 2: key statistics]',
                'Finding 2 placeholder')
    add_band_and_footer(slide, 'Take-home: ', '[one sentence]',
                        '[Authors]  ·  [Journal], [Year]  ·  doi:[...]')
    notes(slide, 'BLANK: the Draft A frame with placeholders, including the two findings slots. '
                 'Duplicate this slide to try other versions; delete unused slides before '
                 'exporting.\n\n' + ELSEVIER_NOTES)
    return slide


def main():
    app = win32com.client.Dispatch('PowerPoint.Application')
    already_open = app.Presentations.Count
    pres = app.Presentations.Add(MSO_TRUE)  # AddChart2 fails on a windowless presentation
    try:
        pres.Windows(1).WindowState = 2  # ppWindowMinimized
        pres.PageSetup.SlideWidth = W
        pres.PageSetup.SlideHeight = H
        slide_draft_a(pres)
        slide_draft_b(pres)
        slide_blank(pres)
        if OUT.exists():
            OUT.unlink()
        pres.SaveAs(str(OUT))
        PREVIEWS.mkdir(exist_ok=True)
        for i in range(1, pres.Slides.Count + 1):
            pres.Slides(i).Export(str(PREVIEWS / f'slide{i}.png'), 'PNG', 2016, 806)
        print(f'Saved {OUT}')
    finally:
        pres.Close()
        if already_open == 0 and app.Presentations.Count == 0:
            app.Quit()


if __name__ == '__main__':
    main()
