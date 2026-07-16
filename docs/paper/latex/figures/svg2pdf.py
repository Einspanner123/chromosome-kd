#!/usr/bin/env python3
"""Convert SVG figures to PDF format.

This script converts all SVG files in the current directory to PDF format
using cairosvg library.

Requirements:
    pip install cairosvg

Usage:
    python svg2pdf.py
    python svg2pdf.py --output-dir ./pdf_output
    python svg2pdf.py --input-dir ./svg_input --output-dir ./pdf_output
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import cairosvg
except ImportError:
    print("Error: cairosvg is not installed.")
    print("Install it with: pip install cairosvg")
    sys.exit(1)


def convert_svg_to_pdf(svg_path: Path, pdf_path: Path) -> bool:
    """Convert a single SVG file to PDF.

    Args:
        svg_path: Path to input SVG file
        pdf_path: Path to output PDF file

    Returns:
        True if conversion successful, False otherwise
    """
    try:
        cairosvg.svg2pdf(
            url=str(svg_path),
            write_to=str(pdf_path),
            output_width=None,  # Keep original size
            output_height=None,
        )
        return True
    except Exception as e:
        print(f"Error converting {svg_path.name}: {e}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert SVG files to PDF format"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory containing SVG files (default: script directory)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for PDF files (default: same as input)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed conversion information",
    )

    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir if args.output_dir else input_dir

    # Validate input directory
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return 1

    if not input_dir.is_dir():
        print(f"Error: Input path is not a directory: {input_dir}")
        return 1

    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all SVG files
    svg_files = sorted(input_dir.glob("*.svg"))

    if not svg_files:
        print(f"No SVG files found in {input_dir}")
        return 0

    print(f"Found {len(svg_files)} SVG file(s) in {input_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Convert each SVG to PDF
    success_count = 0
    for svg_path in svg_files:
        pdf_path = output_dir / f"{svg_path.stem}.pdf"

        if args.verbose:
            print(f"Converting: {svg_path.name} -> {pdf_path.name}")

        if convert_svg_to_pdf(svg_path, pdf_path):
            success_count += 1
            if args.verbose:
                print(f"  ✓ Success")
        else:
            print(f"  ✗ Failed")

    print()
    print(f"Conversion complete: {success_count}/{len(svg_files)} files converted")

    return 0 if success_count == len(svg_files) else 1


if __name__ == "__main__":
    sys.exit(main())
