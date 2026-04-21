#!/usr/bin/env python3
"""
Process a PDF with Azure Content Understanding (prebuilt-documentSearch).

Usage
-----
    python src/process_document.py path/to/document.pdf
    python src/process_document.py path/to/document.pdf --output-dir results/

Outputs (under <output-dir>)
----------------------------
    result.json          Full raw API response
    document.md          Full document markdown (text, tables, figure refs)
    tables.md            All tables rendered as markdown
    figures/             Per-figure description files + optional image download
        figure_<id>.md   AI description, caption, Chart.js / Mermaid data
        figure_<id>.png  Cropped figure image (when available from service)

Auth
----
    Uses DefaultAzureCredential (az login / managed identity — no keys in code).
    Endpoint is read from AZURE_CONTENT_UNDERSTANDING_ENDPOINT.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import cast

from azure.ai.contentunderstanding import ContentUnderstandingClient
from azure.ai.contentunderstanding.models import (
    AnalysisResult,
    DocumentChartFigure,
    DocumentContent,
    DocumentFigure,
    DocumentMermaidFigure,
)
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------


def _fmt_elapsed(secs: float) -> str:
    """Format seconds as '1m 23s' or '45.23s'."""
    if secs >= 60:
        m, s = divmod(secs, 60)
        return f"{int(m)}m {s:.0f}s"
    return f"{secs:.2f}s"


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


def _build_client() -> ContentUnderstandingClient:
    load_dotenv()
    endpoint = os.environ.get("AZURE_CONTENT_UNDERSTANDING_ENDPOINT")
    if not endpoint:
        raise EnvironmentError(
            "AZURE_CONTENT_UNDERSTANDING_ENDPOINT is not set. "
            "Check your .env file or environment."
        )
    return ContentUnderstandingClient(
        endpoint=endpoint,
        credential=DefaultAzureCredential(),
    )


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _analyze(
    client: ContentUnderstandingClient,
    pdf_path: Path,
) -> tuple[AnalysisResult, str, dict[str, float], dict | None]:
    """Submit the PDF for analysis and return (result, operation_id, timings, usage)."""
    t0 = time.perf_counter()
    print(f"Reading {pdf_path} ...")
    with open(pdf_path, "rb") as fh:
        pdf_bytes = fh.read()
    read_time = time.perf_counter() - t0
    print(f"  {len(pdf_bytes):,} bytes read in {_fmt_elapsed(read_time)}")

    t1 = time.perf_counter()
    print("Submitting to prebuilt-documentSearch ...")
    poller = client.begin_analyze_binary(
        analyzer_id="prebuilt-documentSearch",
        binary_input=pdf_bytes,
    )
    operation_id: str = poller.operation_id
    submit_time = time.perf_counter() - t1
    print(f"  Submitted in {_fmt_elapsed(submit_time)}  —  Operation ID: {operation_id}  (polling ...)")

    t2 = time.perf_counter()
    result: AnalysisResult = poller.result()
    poll_time = time.perf_counter() - t2
    print(f"Analysis complete  ({_fmt_elapsed(poll_time)}).")

    usage = _extract_usage(poller)

    return result, operation_id, {"read": read_time, "submit": submit_time, "poll": poll_time}, usage


def _extract_usage(poller) -> dict | None:
    """Pull usage details from the poller's final polling response.

    The SDK deserializes only the ``result`` field; ``usage`` sits at the
    operation-status level so we read it from the raw HTTP response.
    """
    try:
        response_json = (
            poller.polling_method()._pipeline_response.http_response.json()
        )
        return response_json.get("usage")
    except Exception:  # noqa: BLE001
        return None


def _print_usage(usage: dict | None) -> None:
    """Print a usage summary block."""
    if not usage:
        return

    _SEP = "\u2500" * 50
    print(f"\n{_SEP}")
    print("  Usage details")
    print(_SEP)

    # ---- Document pages ----
    pages_min = usage.get("documentPagesMinimal")
    pages_basic = usage.get("documentPagesBasic")
    pages_std = usage.get("documentPagesStandard")
    if any(v is not None for v in (pages_min, pages_basic, pages_std)):
        print("  Document pages:")
        if pages_min is not None:
            print(f"    Minimal            : {pages_min:,}")
        if pages_basic is not None:
            print(f"    Basic              : {pages_basic:,}")
        if pages_std is not None:
            print(f"    Standard           : {pages_std:,}")

    # ---- Audio / Video ----
    audio_h = usage.get("audioHours")
    video_h = usage.get("videoHours")
    if audio_h is not None:
        print(f"  Audio                : {audio_h:.3f} hours")
    if video_h is not None:
        print(f"  Video                : {video_h:.3f} hours")

    # ---- Contextualization tokens (API may return singular or plural key) ----
    ctx_tokens = usage.get("contextualizationTokens") or usage.get("contextualizationToken") or 0

    # ---- Per-model tokens ----
    tokens = usage.get("tokens") or {}
    if ctx_tokens or tokens:
        print("  Tokens:")
        if ctx_tokens:
            print(f"    {'Contextualization':<33s}: {ctx_tokens:>10,}")
        for key, count in sorted(tokens.items()):
            print(f"    {key:<33s}: {count:>10,}")
        total_tokens = sum(tokens.values()) + ctx_tokens
        print(f"    {'TOTAL':<33s}: {total_tokens:>10,}")
    print(_SEP)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _save_json(result: AnalysisResult, output_dir: Path) -> None:
    path = output_dir / "result.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result.as_dict(), fh, indent=2, ensure_ascii=False, default=str)
    print(f"  [saved] {path}")


def _save_markdown(doc: DocumentContent, output_dir: Path) -> None:
    path = output_dir / "document.md"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc.markdown or "")
    print(f"  [saved] {path}")


def _save_tables(doc: DocumentContent, output_dir: Path) -> None:
    if not doc.tables:
        print("  No structured tables found; skipping tables.md")
        return

    path = output_dir / "tables.md"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Extracted Tables\n\n")
        for idx, table in enumerate(doc.tables, start=1):
            caption_text = table.caption.content if table.caption else f"Table {idx}"
            fh.write(f"## {caption_text}\n\n")
            fh.write(f"*{table.row_count} rows \u00d7 {table.column_count} columns*\n\n")

            # Build a sparse grid then render as a markdown table
            grid: dict[tuple[int, int], str] = {}
            for cell in table.cells or []:
                content = cell.content.replace("|", "\\|").replace("\n", " ")
                grid[(cell.row_index, cell.column_index)] = content

            cols = range(table.column_count)
            header = [grid.get((0, c), "") for c in cols]
            fh.write("| " + " | ".join(header) + " |\n")
            fh.write("| " + " | ".join(["---"] * table.column_count) + " |\n")
            for row in range(1, table.row_count):
                cells = [grid.get((row, c), "") for c in cols]
                fh.write("| " + " | ".join(cells) + " |\n")
            fh.write("\n")

    print(f"  [saved] {path}  ({len(doc.tables)} table(s))")


def _save_figures(
    client: ContentUnderstandingClient,
    doc: DocumentContent,
    operation_id: str,
    output_dir: Path,
) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    if not doc.figures:
        print("  No figures found; skipping figures/")
        return

    print(f"  Processing {len(doc.figures)} figure(s) ...")
    for figure in doc.figures:
        t_fig = time.perf_counter()
        _save_figure_description(figure, figures_dir)
        _try_download_figure_image(client, operation_id, figure, figures_dir)
        print(f"    [time]  {figure.id}: {_fmt_elapsed(time.perf_counter() - t_fig)}")


def _safe_stem(figure_id: str) -> str:
    """Convert a figure ID to a safe filename stem."""
    return figure_id.replace("/", "_").replace(":", "_").replace(".", "_")


def _save_figure_description(figure: DocumentFigure, figures_dir: Path) -> None:
    stem = _safe_stem(figure.id)
    path = figures_dir / f"figure_{stem}.md"

    lines: list[str] = [f"# Figure `{figure.id}`\n\n"]

    if figure.caption:
        lines.append(f"**Caption:** {figure.caption.content}\n\n")

    if figure.description:
        lines.append(f"## AI Description\n\n{figure.description}\n\n")

    if isinstance(figure, DocumentChartFigure):
        lines.append("## Chart Data (Chart.js)\n\n```json\n")
        lines.append(json.dumps(figure.content, indent=2))
        lines.append("\n```\n\n")
    elif isinstance(figure, DocumentMermaidFigure):
        lines.append("## Diagram (Mermaid)\n\n```mermaid\n")
        lines.append(figure.content)
        lines.append("\n```\n\n")

    if figure.source:
        lines.append(f"**Source coordinates:** `{figure.source}`\n")

    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(lines)

    print(f"    [saved] {path.name}")


def _try_download_figure_image(
    client: ContentUnderstandingClient,
    operation_id: str,
    figure: DocumentFigure,
    figures_dir: Path,
) -> None:
    """
    Attempt to retrieve the figure as a cropped image via get_result_file.

    For document PDFs the service may not return image bytes for all figure
    types. When the call fails the per-figure .md file is the authoritative
    record of the figure content (description, chart data, diagram syntax).
    """
    try:
        response = client.get_result_file(
            operation_id=operation_id,
            path=f"figures/{figure.id}",
        )
        image_bytes = b"".join(response)
        if not image_bytes:
            return
        stem = _safe_stem(figure.id)
        out_path = figures_dir / f"figure_{stem}.png"
        with open(out_path, "wb") as fh:
            fh.write(image_bytes)
        print(f"    [saved] {out_path.name}  ({len(image_bytes):,} bytes)")
    except Exception as exc:  # noqa: BLE001
        print(f"    [info]  Image not available for figure {figure.id}: {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process a PDF with Azure Content Understanding "
                    "(prebuilt-documentSearch).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("pdf_path", help="Path to the input PDF file.")
    parser.add_argument(
        "--output-dir", "-o",
        default="output",
        help="Directory for output files (default: ./output).",
    )
    parser.add_argument(
        "--save-extras",
        action="store_true",
        default=False,
        help="Also save tables (tables.md) and figures (figures/*). "
             "Disabled by default; only result.json and document.md are saved.",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path).resolve()
    if not pdf_path.is_file():
        print(f"Error: file not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    client = _build_client()
    result, operation_id, analysis_times, usage = _analyze(client, pdf_path)

    if not result.contents:
        print("Error: empty response from service.", file=sys.stderr)
        sys.exit(1)

    doc = cast(DocumentContent, result.contents[0])
    print(
        f"\nDocument: {doc.mime_type or 'unknown'}  "
        f"pages {doc.start_page_number}\u2013{doc.end_page_number}\n"
        f"Writing output to: {output_dir}\n"
    )

    timings: list[tuple[str, float]] = [
        ("PDF read",   analysis_times["read"]),
        ("API submit", analysis_times["submit"]),
        ("API poll",   analysis_times["poll"]),
    ]

    t = time.perf_counter()
    _save_json(result, output_dir)
    timings.append(("Save JSON", time.perf_counter() - t))

    t = time.perf_counter()
    _save_markdown(doc, output_dir)
    timings.append(("Save markdown", time.perf_counter() - t))

    if args.save_extras:
        t = time.perf_counter()
        _save_tables(doc, output_dir)
        timings.append(("Save tables", time.perf_counter() - t))

        t = time.perf_counter()
        _save_figures(client, doc, operation_id, output_dir)
        timings.append(("Save figures", time.perf_counter() - t))

    total = time.perf_counter() - t_start
    _SEP = "\u2500" * 50
    print(f"\n{_SEP}")
    print("  Timing summary")
    print(_SEP)
    for label, elapsed in timings:
        print(f"  {label:<20s}  {elapsed:9.2f}s  ({_fmt_elapsed(elapsed)})")
    print(_SEP)
    print(f"  {'Total':<20s}  {total:9.2f}s  ({_fmt_elapsed(total)})")
    print(_SEP)

    _print_usage(usage)

    print("\nDone.")


if __name__ == "__main__":
    main()
