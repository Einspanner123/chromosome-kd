from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / 'figures'


def wrap_tspans(text, x, start_y, line_gap=18):
    lines = text.split('\n')
    parts = []
    for idx, line in enumerate(lines):
        dy = 0 if idx == 0 else line_gap
        parts.append(f'<tspan x="{x}" dy="{dy}">{line}</tspan>')
    return ''.join(parts)


def svg_header(width, height, title):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs>',
        '<marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">',
        '<path d="M0,0 L0,6 L9,3 z" fill="#34495e"/>',
        '</marker>',
        '</defs>',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width / 2}" y="36" font-size="24" text-anchor="middle" font-family="Arial">{title}</text>',
    ]


def add_box(parts,
            x,
            y,
            width,
            height,
            text,
            fill='#f3f6fb',
            stroke='#2c3e50'):
    parts.append(
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="10" ry="10" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
    )
    parts.append(
        f'<text x="{x + width / 2}" y="{y + height / 2 - 10}" font-size="18" text-anchor="middle" font-family="Arial">'
        f'{wrap_tspans(text, x + width / 2, y + height / 2 - 10)}'
        '</text>')


def add_arrow(parts, x1, y1, x2, y2):
    parts.append(
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#34495e" stroke-width="2.5" marker-end="url(#arrow)"/>'
    )


def save_svg(filename, parts):
    parts.append('</svg>')
    (FIG_DIR / filename).write_text('\n'.join(parts), encoding='utf-8')


def figure_overview():
    parts = svg_header(1100, 600, 'Figure 1. Paper Overview')
    add_box(parts, 70, 120, 220, 120, 'Low-dimensional\nchromosome detection')
    add_box(parts, 360, 120, 220, 120, 'Deterministic OT\ncoupling')
    add_box(
        parts,
        650,
        120,
        290,
        120,
        'Diversity collapse\n+ argmax control loss',
        fill='#fdecea',
        stroke='#c0392b')
    add_box(
        parts,
        360,
        360,
        220,
        120,
        'Stochastic\ncoupling',
        fill='#eafaf1',
        stroke='#1e8449')
    add_box(
        parts,
        650,
        360,
        290,
        120,
        'Recovered diversity\nand better mAP',
        fill='#eafaf1',
        stroke='#1e8449')
    add_arrow(parts, 290, 180, 360, 180)
    add_arrow(parts, 580, 180, 650, 180)
    add_arrow(parts, 470, 240, 470, 360)
    add_arrow(parts, 580, 420, 650, 420)
    parts.append(
        '<text x="70" y="545" font-size="16" font-family="Arial">Replace with the final polished teaser figure after quantitative plots are fixed.</text>'
    )
    save_svg('figure1_overview.svg', parts)


def figure_coupling():
    parts = svg_header(1180, 600, 'Figure 2. Coupling Mechanisms')
    labels = [
        ('Random', '#f7f9f9'),
        ('Hard OT', '#fef5e7'),
        ('Sinkhorn + argmax', '#fdecea'),
        ('Sinkhorn + sample', '#eafaf1'),
    ]
    x_positions = [40, 320, 600, 880]
    for (label, color), x in zip(labels, x_positions):
        add_box(parts, x, 100, 220, 400, label, fill=color)
        parts.append(
            f'<text x="{x + 110}" y="150" font-size="16" text-anchor="middle" font-family="Arial">noise states</text>'
        )
        parts.append(
            f'<text x="{x + 110}" y="470" font-size="16" text-anchor="middle" font-family="Arial">GT boxes</text>'
        )
        for idx in range(3):
            y1 = 185 + idx * 55
            y2 = 395 - idx * 40
            parts.append(
                f'<line x1="{x + 60}" y1="{y1}" x2="{x + 160}" y2="{y2}" stroke="#7f8c8d" stroke-width="2"/>'
            )
    parts.append(
        '<text x="40" y="555" font-size="16" font-family="Arial">Replace line patterns with real assignment heatmaps or matrix visualizations.</text>'
    )
    save_svg('figure2_coupling.svg', parts)


def figure_epsilon():
    parts = svg_header(1100, 560, 'Figure 3. Epsilon Regimes')
    parts.append(
        '<rect x="70" y="100" width="180" height="330" fill="#fdecea" opacity="0.95"/>'
    )
    parts.append(
        '<rect x="250" y="100" width="230" height="330" fill="#eafaf1" opacity="0.95"/>'
    )
    parts.append(
        '<rect x="480" y="100" width="520" height="330" fill="#fef5e7" opacity="0.95"/>'
    )
    parts.append(
        '<text x="160" y="130" font-size="18" text-anchor="middle" font-family="Arial">OT-dominated</text>'
    )
    parts.append(
        '<text x="365" y="130" font-size="18" text-anchor="middle" font-family="Arial">Sweet spot</text>'
    )
    parts.append(
        '<text x="740" y="130" font-size="18" text-anchor="middle" font-family="Arial">Bias-dominated</text>'
    )
    points_map = [(80, 410), (120, 360), (230, 270), (300, 225), (430, 185),
                  (520, 210), (760, 315), (980, 385)]
    points_div = [(80, 425), (120, 395), (230, 315), (300, 235), (430, 185),
                  (520, 165), (760, 150), (980, 150)]
    parts.append(
        '<polyline fill="none" stroke="#2c3e50" stroke-width="3" points="' +
        ' '.join(f'{x},{y}' for x, y in points_map) + '"/>')
    parts.append(
        '<polyline fill="none" stroke="#1e8449" stroke-width="3" points="' +
        ' '.join(f'{x},{y}' for x, y in points_div) + '"/>')
    for x, y in points_map:
        parts.append(f'<circle cx="{x}" cy="{y}" r="5" fill="#2c3e50"/>')
    for x, y in points_div:
        parts.append(
            f'<rect x="{x - 4}" y="{y - 4}" width="8" height="8" fill="#1e8449"/>'
        )
    parts.append(
        '<text x="720" y="470" font-size="16" font-family="Arial">Black line: placeholder mAP</text>'
    )
    parts.append(
        '<text x="720" y="495" font-size="16" font-family="Arial">Green line: placeholder diversity</text>'
    )
    parts.append(
        '<text x="70" y="520" font-size="16" font-family="Arial">Replace with real curves after the epsilon scan is finalized.</text>'
    )
    save_svg('figure3_epsilon.svg', parts)


def figure_dataset():
    parts = svg_header(1100, 600, 'Figure 4. Dataset Overview')
    add_box(parts, 80, 120, 300, 120, 'Dataset A\nMain chromosome benchmark')
    add_box(parts, 720, 120, 300, 120, 'Dataset B\nsingle_chromosomes_object')
    add_box(
        parts,
        80,
        320,
        300,
        160,
        'Representative image\nplaceholder',
        fill='#f7f9f9')
    add_box(
        parts, 400, 320, 300, 160, 'Stats panel\nplaceholder', fill='#f7f9f9')
    add_box(
        parts,
        720,
        320,
        300,
        160,
        'Representative image\nplaceholder',
        fill='#f7f9f9')
    add_arrow(parts, 230, 240, 230, 320)
    add_arrow(parts, 870, 240, 870, 320)
    parts.append(
        '<text x="80" y="545" font-size="16" font-family="Arial">Replace with actual image crops, annotation overlays, and dataset statistics.</text>'
    )
    save_svg('figure4_dataset.svg', parts)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figure_overview()
    figure_coupling()
    figure_epsilon()
    figure_dataset()
    print(f'Generated placeholder figures in {FIG_DIR}')


if __name__ == '__main__':
    main()
