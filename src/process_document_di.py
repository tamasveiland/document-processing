#!/usr/bin/env python3
"""
Process a PDF with Azure Document Intelligence (prebuilt-layout).

Usage
-----
    python src/process_document_di.py path/to/document.pdf
    python src/process_document_di.py path/to/document.pdf --output-dir results/
    python src/process_document_di.py path/to/document.pdf --model-id prebuilt-read

Outputs (under <output-dir>)
----------------------------
    result.json          Full raw API response
    document.md          Full document markdown content
    tables.md            All tables rendered as markdown (when --save-extras)

Auth
----
    Uses DefaultAzureCredential (az login / managed identity — no keys in code).
    Endpoint is read from AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import (
    AnalyzeDocumentRequest,
    AnalyzeOutputOption,
    AnalyzeResult,
    DocumentContentFormat,
    DocumentAnalysisFeature,
)
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------


def _fmt_elapsed(secs: float) -> str:
    if secs >= 60:
        m, s = divmod(secs, 60)
        return f"{int(m)}m {s:.0f}s"
    return f"{secs:.2f}s"


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


def _build_client() -> DocumentIntelligenceClient:
    load_dotenv()
    endpoint = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    if not endpoint:
        raise EnvironmentError(
            "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT is not set. "
            "Check your .env file or environment."
        )
    return DocumentIntelligenceClient(
        endpoint=endpoint,
        credential=DefaultAzureCredential(),
    )


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _analyze(
    client: DocumentIntelligenceClient,
    pdf_path: Path,
    model_id: str,
) -> tuple[AnalyzeResult, str | None, dict[str, float]]:
    """Return (result, result_id, timings)."""
    t0 = time.perf_counter()
    print(f"Reading {pdf_path} ...")
    with open(pdf_path, "rb") as fh:
        pdf_bytes = fh.read()
    read_time = time.perf_counter() - t0
    print(f"  {len(pdf_bytes):,} bytes read in {_fmt_elapsed(read_time)}")

    t1 = time.perf_counter()
    print(f"Submitting to {model_id} ...")
    poller = client.begin_analyze_document(
        model_id,
        pdf_bytes,
        output_content_format=DocumentContentFormat.MARKDOWN,
        features=[DocumentAnalysisFeature.FORMULAS],
        output=[AnalyzeOutputOption.FIGURES],
    )
    submit_time = time.perf_counter() - t1
    print(f"  Submitted in {_fmt_elapsed(submit_time)}  (polling ...)")

    t2 = time.perf_counter()
    result: AnalyzeResult = poller.result()
    poll_time = time.perf_counter() - t2
    print(f"Analysis complete  ({_fmt_elapsed(poll_time)}).")

    result_id = _extract_result_id(poller)

    return result, result_id, {"read": read_time, "submit": submit_time, "poll": poll_time}


def _extract_result_id(poller) -> str | None:
    """Extract the result ID from the poller's operation-location URL.

    The URL looks like:
    .../documentModels/{modelId}/analyzeResults/{resultId}?api-version=...
    """
    try:
        op_url = poller.polling_method()._initial_response.http_response.headers.get(
            "Operation-Location", ""
        )
        # Extract the segment after "analyzeResults/"
        marker = "analyzeResults/"
        idx = op_url.find(marker)
        if idx == -1:
            return None
        rest = op_url[idx + len(marker):]
        return rest.split("?")[0].split("/")[0]
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _save_json(result: AnalyzeResult, output_dir: Path) -> None:
    path = output_dir / "result.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result.as_dict(), fh, indent=2, ensure_ascii=False, default=str)
    print(f"  [saved] {path}")


def _save_markdown(result: AnalyzeResult, output_dir: Path) -> None:
    path = output_dir / "document.md"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(result.content or "")
    print(f"  [saved] {path}")


def _save_tables(result: AnalyzeResult, output_dir: Path) -> None:
    if not result.tables:
        print("  No structured tables found; skipping tables.md")
        return

    path = output_dir / "tables.md"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Extracted Tables\n\n")
        for idx, table in enumerate(result.tables, start=1):
            fh.write(f"## Table {idx}\n\n")
            fh.write(f"*{table.row_count} rows \u00d7 {table.column_count} columns*\n\n")

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

    print(f"  [saved] {path}  ({len(result.tables)} table(s))")


def _save_figures(
    client: DocumentIntelligenceClient,
    result: AnalyzeResult,
    model_id: str,
    result_id: str | None,
    output_dir: Path,
) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    if not result.figures:
        print("  No figures found; skipping figures/")
        return

    if not result_id:
        print("  [warn] Could not determine result_id; skipping figure image downloads.")

    print(f"  Processing {len(result.figures)} figure(s) ...")
    for figure in result.figures:
        t_fig = time.perf_counter()
        _save_figure_description(figure, figures_dir)
        if result_id:
            _try_download_figure_image(client, model_id, result_id, figure, figures_dir)
        print(f"    [time]  {figure.id}: {_fmt_elapsed(time.perf_counter() - t_fig)}")


def _safe_stem(figure_id: str) -> str:
    """Convert a figure ID to a safe filename stem."""
    return figure_id.replace("/", "_").replace(":", "_").replace(".", "_")


def _save_figure_description(figure, figures_dir: Path) -> None:
    stem = _safe_stem(figure.id)
    path = figures_dir / f"figure_{stem}.md"

    lines: list[str] = [f"# Figure `{figure.id}`\n\n"]

    if figure.caption:
        lines.append(f"**Caption:** {figure.caption.content}\n\n")

    if figure.bounding_regions:
        for region in figure.bounding_regions:
            lines.append(
                f"**Page {region.page_number}** — "
                f"bounding polygon: `{region.polygon}`\n\n"
            )

    if figure.footnotes:
        lines.append("## Footnotes\n\n")
        for fn in figure.footnotes:
            lines.append(f"- {fn.content}\n")
        lines.append("\n")

    if figure.elements:
        lines.append(f"**Elements:** {len(figure.elements)} content element(s)\n")

    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(lines)

    print(f"    [saved] {path.name}")


def _try_download_figure_image(
    client: DocumentIntelligenceClient,
    model_id: str,
    result_id: str,
    figure,
    figures_dir: Path,
) -> None:
    """Download the cropped figure image via get_analyze_result_figure."""
    try:
        response = client.get_analyze_result_figure(
            model_id=model_id,
            result_id=result_id,
            figure_id=figure.id,
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
        description="Process a PDF with Azure Document Intelligence.",
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
        "--model-id", "-m",
        default="prebuilt-layout",
        help="Document Intelligence model ID (default: prebuilt-layout). "
             "Other options: prebuilt-read, prebuilt-invoice, prebuilt-receipt, etc.",
    )
    parser.add_argument(
        "--save-extras",
        action="store_true",
        default=False,
        help="Also save tables (tables.md). "
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
    result, result_id, analysis_times = _analyze(client, pdf_path, args.model_id)

    page_count = len(result.pages) if result.pages else 0
    table_count = len(result.tables) if result.tables else 0
    figure_count = len(result.figures) if result.figures else 0
    print(
        f"\nDocument: {page_count} page(s), {table_count} table(s), {figure_count} figure(s)\n"
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
    _save_markdown(result, output_dir)
    timings.append(("Save markdown", time.perf_counter() - t))

    if args.save_extras:
        t = time.perf_counter()
        _save_tables(result, output_dir)
        timings.append(("Save tables", time.perf_counter() - t))

        t = time.perf_counter()
        _save_figures(client, result, args.model_id, result_id, output_dir)
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

    print("\nDone.")


if __name__ == "__main__":
    main()
