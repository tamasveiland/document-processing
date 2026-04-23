#!/usr/bin/env python3
"""
Three-pass PDF processing with Azure Content Understanding.

Pass 1 — Text extraction (prebuilt-read) on the full document.
          Fast OCR / text-layer extraction.  Also used to detect which
          pages carry tables and which carry figures.

Pass 2 — Table extraction (prebuilt-layout) on table-bearing pages only.
          Returns detailed cell-level table data.

Pass 3 — Figure analysis (prebuilt-documentSearch) on figure-bearing
          pages only.  Provides AI descriptions, Chart.js charts, and
          Mermaid diagrams.

The final output merges text (pass 1), tables (pass 2), and figures
(pass 3), giving the best of each analyzer while minimizing cost and
latency.

Usage
-----
    python process_document_threepass.py report.pdf
    python process_document_threepass.py report.pdf --workers 8
    python process_document_threepass.py report.pdf --save-extras

Auth
----
    Uses DefaultAzureCredential.  Endpoint from
    AZURE_CONTENT_UNDERSTANDING_ENDPOINT.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import cast

import pypdfium2 as pdfium  # PDF operations (Apache-2.0)
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
# Helpers
# ---------------------------------------------------------------------------


def _fmt_elapsed(secs: float) -> str:
    if secs >= 60:
        m, s = divmod(secs, 60)
        return f"{int(m)}m {s:.0f}s"
    return f"{secs:.2f}s"


# ---------------------------------------------------------------------------
# Pass 1 — Text extraction (prebuilt-read, full document)
# ---------------------------------------------------------------------------


def _pass1_text(
    client: ContentUnderstandingClient,
    pdf_bytes: bytes,
) -> tuple[AnalysisResult, str, float, dict | None]:
    """Run prebuilt-read on the full document for fast text extraction.

    Returns (result, operation_id, elapsed_secs, usage).
    """
    t0 = time.perf_counter()
    print("Pass 1 — submitting full document to prebuilt-read (text) ...")

    poller = client.begin_analyze_binary(
        analyzer_id="prebuilt-read",
        binary_input=pdf_bytes,
    )
    op_id: str = poller.operation_id
    print(f"  Operation {op_id}  (polling ...)")

    result: AnalysisResult = poller.result()
    elapsed = time.perf_counter() - t0
    print(f"  Pass 1 complete in {_fmt_elapsed(elapsed)}.")

    usage = _extract_usage(poller)
    return result, op_id, elapsed, usage


# ---------------------------------------------------------------------------
# Page detection helpers
# ---------------------------------------------------------------------------

# Markdown table pattern: lines that look like  | ... | ... |
_TABLE_ROW_RE = re.compile(r"^\s*\|.+\|", re.MULTILINE)
# Separator row: | --- | --- |  (confirms a real table)
_TABLE_SEP_RE = re.compile(r"^\s*\|[\s\-:]+\|", re.MULTILINE)

# Figure references in prebuilt-read output
_FIGURE_REF_RE = re.compile(
    r":figure:\d+|<!-- *Figure|!\[.*?\]\(figures/",
    re.IGNORECASE,
)


def _split_markdown_by_page(markdown: str) -> list[tuple[int, str]]:
    """Split markdown into (page_number, section_text) tuples.

    Handles common page break markers produced by Content Understanding.
    """
    # Common markers: <!-- PageBreak --> or <!-- PageNumber="N" -->
    parts = re.split(r"(<!-- *PageBreak *-->|<!-- *PageNumber=\"\d+\" *-->)", markdown)

    pages: list[tuple[int, str]] = []
    current_page = 1
    current_text = ""

    for part in parts:
        if not part:
            continue
        pn_match = re.search(r"PageNumber=\"(\d+)\"", part)
        if pn_match:
            # Save previous section
            if current_text.strip():
                pages.append((current_page, current_text))
            current_page = int(pn_match.group(1))
            current_text = ""
        elif re.match(r"<!-- *PageBreak *-->", part):
            if current_text.strip():
                pages.append((current_page, current_text))
            current_page += 1 if pages else 1
            current_text = ""
        else:
            current_text += part

    if current_text.strip():
        pages.append((current_page, current_text))

    return pages


def _detect_table_pages_from_markdown(doc: DocumentContent) -> set[int]:
    """Detect pages containing tables by scanning pass-1 markdown for table syntax."""
    pages: set[int] = set()

    # First, try structured tables with page references
    if doc.tables:
        for table in doc.tables:
            if hasattr(table, "bounding_regions") and table.bounding_regions:
                for region in table.bounding_regions:
                    if hasattr(region, "page_number") and region.page_number:
                        pages.add(region.page_number)
            # Also check cells for page info
            for cell in table.cells or []:
                if hasattr(cell, "bounding_regions") and cell.bounding_regions:
                    for region in cell.bounding_regions:
                        if hasattr(region, "page_number") and region.page_number:
                            pages.add(region.page_number)

    # Fallback: scan markdown per-page sections for table row patterns
    if not pages and doc.markdown:
        page_sections = _split_markdown_by_page(doc.markdown)
        for page_num, section in page_sections:
            # A table needs both data rows and a separator row
            rows = _TABLE_ROW_RE.findall(section)
            seps = _TABLE_SEP_RE.findall(section)
            if len(rows) >= 3 and seps:
                pages.add(page_num)

    return pages


def _detect_table_pages_pymupdf(pdf_path: Path) -> set[int]:
    """Detect pages likely containing tables using text heuristics."""
    doc = pdfium.PdfDocument(pdf_path)
    pages: set[int] = set()
    for idx in range(len(doc)):
        page = doc[idx]
        textpage = page.get_textpage()
        text = textpage.get_text_range()
        textpage.close()
        page.close()
        lines = text.split("\n")
        tab_lines = sum(1 for ln in lines if ln.count("\t") >= 2 or ln.count("  ") >= 4)
        if tab_lines >= 3:
            pages.add(idx + 1)
    doc.close()
    return pages


def _detect_figure_pages(doc: DocumentContent) -> set[int]:
    """Return 1-based page numbers that contain figures from pass-1 results."""
    pages: set[int] = set()

    # Structured figures with page references
    if doc.figures:
        for fig in doc.figures:
            if fig.source:
                m = re.search(r"page[:\s]*(\d+)", str(fig.source), re.IGNORECASE)
                if m:
                    pages.add(int(m.group(1)))

    # Fallback: scan markdown for figure markers per page section
    if not pages and doc.markdown:
        page_sections = _split_markdown_by_page(doc.markdown)
        for page_num, section in page_sections:
            if _FIGURE_REF_RE.search(section):
                pages.add(page_num)

    return pages


def _detect_figure_pages_pymupdf(pdf_path: Path) -> set[int]:
    """Detect pages with significant images using pypdfium2."""
    doc = pdfium.PdfDocument(pdf_path)
    pages: set[int] = set()
    for idx in range(len(doc)):
        page = doc[idx]
        page_area = page.get_width() * page.get_height() or 1.0
        total_area = 0.0
        image_count = 0
        for obj in page.get_objects():
            if obj.type == pdfium.FPDF_PAGEOBJ_IMAGE:
                image_count += 1
                try:
                    left, bottom, right, top = obj.get_pos()
                    total_area += abs((right - left) * (top - bottom))
                except Exception:  # noqa: BLE001
                    total_area += page_area * 0.25
        # Only count pages where images occupy >5% of the page
        if image_count > 0 and total_area / page_area > 0.05:
            pages.add(idx + 1)
        page.close()
    doc.close()
    return pages


# ---------------------------------------------------------------------------
# Chunk building & parallel analysis
# ---------------------------------------------------------------------------


def _build_page_pdf(pdf_path: Path, page_numbers: list[int]) -> bytes:
    """Extract specific 1-based page numbers into a new PDF."""
    src = pdfium.PdfDocument(pdf_path)
    out = pdfium.PdfDocument.new()
    out.import_pages(src, [pn - 1 for pn in page_numbers])
    buf = io.BytesIO()
    out.save(buf)
    out.close()
    src.close()
    return buf.getvalue()


def _group_consecutive(pages: list[int], max_group: int = 4) -> list[list[int]]:
    """Group sorted page numbers into consecutive runs, max *max_group* each."""
    if not pages:
        return []
    groups: list[list[int]] = []
    current: list[int] = [pages[0]]
    for p in pages[1:]:
        if p == current[-1] + 1 and len(current) < max_group:
            current.append(p)
        else:
            groups.append(current)
            current = [p]
    groups.append(current)
    return groups


def _analyze_chunk(
    endpoint: str,
    credential: DefaultAzureCredential,
    analyzer_id: str,
    chunk_bytes: bytes,
    chunk_index: int,
    page_label: str,
    pass_label: str,
) -> tuple[int, AnalysisResult, str, float, dict | None]:
    client = ContentUnderstandingClient(endpoint=endpoint, credential=credential)

    t0 = time.perf_counter()
    print(f"  [{pass_label} chunk {chunk_index:>2}] Submitting {page_label} "
          f"({len(chunk_bytes):,} bytes) to {analyzer_id} ...")

    poller = client.begin_analyze_binary(
        analyzer_id=analyzer_id,
        binary_input=chunk_bytes,
    )
    op_id: str = poller.operation_id
    print(f"  [{pass_label} chunk {chunk_index:>2}] Operation {op_id}  (polling ...)")

    result: AnalysisResult = poller.result()
    elapsed = time.perf_counter() - t0
    print(f"  [{pass_label} chunk {chunk_index:>2}] Complete — {page_label} "
          f"in {_fmt_elapsed(elapsed)}")

    usage = _extract_usage(poller)
    return chunk_index, result, op_id, elapsed, usage


def _run_targeted_pass(
    endpoint: str,
    credential: DefaultAzureCredential,
    pdf_path: Path,
    page_list: list[int],
    analyzer_id: str,
    pass_label: str,
    max_workers: int,
    group_size: int,
) -> tuple[list[AnalysisResult], list[str], list[DocumentContent], float, list[tuple[str, float]], list[dict | None]]:
    """Submit page groups for a targeted pass and return aggregated results.

    Returns (results, op_ids, docs, wall_time, per_chunk_timings, usages).
    """
    results: list[AnalysisResult] = []
    op_ids: list[str] = []
    docs: list[DocumentContent] = []
    chunk_timings: list[tuple[str, float]] = []
    usages: list[dict | None] = []

    page_groups = _group_consecutive(page_list, group_size)
    print(f"\n{pass_label} — submitting {len(page_groups)} chunk(s) covering "
          f"{len(page_list)} pages to {analyzer_id} ...\n")

    # Build chunk PDFs
    chunk_data: list[tuple[bytes, str]] = []
    for grp in page_groups:
        chunk_pdf = _build_page_pdf(pdf_path, grp)
        label = f"p{grp[0]}" if len(grp) == 1 else f"p{grp[0]}–{grp[-1]}"
        chunk_data.append((chunk_pdf, label))

    chunk_results: dict[int, tuple[int, AnalysisResult, str, float, dict | None]] = {}
    t_start = time.perf_counter()

    workers = min(max_workers, len(chunk_data))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _analyze_chunk, endpoint, credential, analyzer_id,
                chunk_pdf, idx, label, pass_label,
            ): idx
            for idx, (chunk_pdf, label) in enumerate(chunk_data)
        }
        for future in as_completed(futures):
            idx, result, op_id, elapsed, usage = future.result()
            chunk_results[idx] = (idx, result, op_id, elapsed, usage)

    wall_time = time.perf_counter() - t_start

    # Order by chunk index
    for idx in range(len(chunk_data)):
        _, result, op_id, elapsed, usage = chunk_results[idx]
        results.append(result)
        op_ids.append(op_id)
        usages.append(usage)
        label = chunk_data[idx][1]
        chunk_timings.append((f"  {pass_label} chunk {idx} ({label})", elapsed))
        if result.contents:
            docs.append(cast(DocumentContent, result.contents[0]))

    seq_time = sum(chunk_results[i][3] for i in range(len(chunk_data)))
    print(f"\n{pass_label} complete in {_fmt_elapsed(wall_time)} wall-clock  "
          f"(sequential sum: {_fmt_elapsed(seq_time)}, "
          f"speedup: {seq_time / wall_time:.1f}x)")

    return results, op_ids, docs, wall_time, chunk_timings, usages


# ---------------------------------------------------------------------------
# Usage helpers
# ---------------------------------------------------------------------------


def _extract_usage(poller) -> dict | None:
    try:
        return poller.polling_method()._pipeline_response.http_response.json().get("usage")
    except Exception:  # noqa: BLE001
        return None


def _merge_usage(usages: list[dict | None]) -> dict | None:
    merged: dict = {}
    for u in usages:
        if not u:
            continue
        for key in ("documentPagesMinimal", "documentPagesBasic",
                    "documentPagesStandard",
                    "contextualizationTokens", "contextualizationToken"):
            if u.get(key) is not None:
                merged[key] = merged.get(key, 0) + u[key]
        for key in ("audioHours", "videoHours"):
            if u.get(key) is not None:
                merged[key] = merged.get(key, 0.0) + u[key]
        if u.get("tokens"):
            mt = merged.setdefault("tokens", {})
            for tk, tv in u["tokens"].items():
                mt[tk] = mt.get(tk, 0) + tv
    return merged or None


def _print_usage(usage: dict | None, label: str = "Usage details") -> None:
    if not usage:
        return
    _SEP = "\u2500" * 60
    print(f"\n{_SEP}")
    print(f"  {label}")
    print(_SEP)

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

    audio_h = usage.get("audioHours")
    video_h = usage.get("videoHours")
    if audio_h is not None:
        print(f"  Audio                : {audio_h:.3f} hours")
    if video_h is not None:
        print(f"  Video                : {video_h:.3f} hours")

    ctx_tokens = usage.get("contextualizationTokens") or usage.get("contextualizationToken") or 0
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


def _save_json(
    pass1_result: AnalysisResult,
    pass2_results: list[AnalysisResult],
    pass3_results: list[AnalysisResult],
    output_dir: Path,
) -> None:
    path = output_dir / "result.json"
    payload = {
        "pass1_text": pass1_result.as_dict(),
        "pass2_tables": [r.as_dict() for r in pass2_results],
        "pass3_figures": [r.as_dict() for r in pass3_results],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
    print(f"  [saved] {path}")


def _save_markdown(doc: DocumentContent, output_dir: Path) -> None:
    path = output_dir / "document.md"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc.markdown or "")
    print(f"  [saved] {path}")


def _save_tables(docs: list[DocumentContent], output_dir: Path) -> None:
    """Save tables from pass-2 docs (prebuilt-layout results)."""
    all_tables = [t for doc in docs for t in (doc.tables or [])]
    if not all_tables:
        print("  No structured tables found in pass 2; skipping tables.md")
        return
    path = output_dir / "tables.md"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Extracted Tables\n\n")
        for idx, table in enumerate(all_tables, start=1):
            caption_text = table.caption.content if table.caption else f"Table {idx}"
            fh.write(f"## {caption_text}\n\n")
            fh.write(f"*{table.row_count} rows \u00d7 {table.column_count} columns*\n\n")
            grid: dict[tuple[int, int], str] = {}
            for cell in table.cells or []:
                grid[(cell.row_index, cell.column_index)] = (
                    cell.content.replace("|", "\\|").replace("\n", " ")
                )
            cols = range(table.column_count)
            fh.write("| " + " | ".join(grid.get((0, c), "") for c in cols) + " |\n")
            fh.write("| " + " | ".join(["---"] * table.column_count) + " |\n")
            for row in range(1, table.row_count):
                fh.write("| " + " | ".join(grid.get((row, c), "") for c in cols) + " |\n")
            fh.write("\n")
    print(f"  [saved] {path}  ({len(all_tables)} table(s))")


def _safe_stem(figure_id: str) -> str:
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
    try:
        response = client.get_result_file(operation_id=operation_id, path=f"figures/{figure.id}")
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


def _save_figures(
    client: ContentUnderstandingClient,
    pass3_docs: list[DocumentContent],
    pass3_op_ids: list[str],
    output_dir: Path,
) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    all_figures = [(fig, oid) for doc, oid in zip(pass3_docs, pass3_op_ids)
                   for fig in (doc.figures or [])]
    if not all_figures:
        print("  No figures found in pass 3; skipping figures/")
        return
    print(f"  Processing {len(all_figures)} figure(s) from pass 3 ...")
    for figure, op_id in all_figures:
        t_fig = time.perf_counter()
        _save_figure_description(figure, figures_dir)
        _try_download_figure_image(client, op_id, figure, figures_dir)
        print(f"    [time]  {figure.id}: {_fmt_elapsed(time.perf_counter() - t_fig)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Three-pass PDF processing: text (prebuilt-read) + "
                    "tables (prebuilt-layout) + figures (prebuilt-documentSearch).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("pdf_path", help="Path to the input PDF file.")
    parser.add_argument(
        "--output-dir", "-o", default="output",
        help="Directory for output files (default: ./output).",
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=8,
        help="Max concurrent submissions per pass (default: 8).",
    )
    parser.add_argument(
        "--table-group-size", type=int, default=8,
        help="Max consecutive table-pages per pass-2 chunk (default: 8).",
    )
    parser.add_argument(
        "--figure-group-size", type=int, default=4,
        help="Max consecutive figure-pages per pass-3 chunk (default: 4).",
    )
    parser.add_argument(
        "--save-extras", action="store_true", default=False,
        help="Also save tables.md and figures/*.",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path).resolve()
    if not pdf_path.is_file():
        print(f"Error: file not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    t_global = time.perf_counter()

    # ---- Read PDF ----
    print(f"Reading {pdf_path.name} ...")
    with open(pdf_path, "rb") as fh:
        pdf_bytes = fh.read()
    print(f"  {len(pdf_bytes):,} bytes\n")

    load_dotenv()
    endpoint = os.environ.get("AZURE_CONTENT_UNDERSTANDING_ENDPOINT", "")
    if not endpoint:
        print("Error: AZURE_CONTENT_UNDERSTANDING_ENDPOINT is not set.", file=sys.stderr)
        sys.exit(1)
    credential = DefaultAzureCredential()
    client = ContentUnderstandingClient(endpoint=endpoint, credential=credential)

    all_usages: list[dict | None] = []
    timings: list[tuple[str, float]] = []

    # ==================================================================
    # PASS 1 — Text extraction (prebuilt-read, full document)
    # ==================================================================
    pass1_result, pass1_op_id, pass1_time, pass1_usage = _pass1_text(client, pdf_bytes)
    timings.append(("Pass 1 (text)", pass1_time))
    all_usages.append(pass1_usage)

    if not pass1_result.contents:
        print("Error: empty response from pass 1.", file=sys.stderr)
        sys.exit(1)

    pass1_doc = cast(DocumentContent, pass1_result.contents[0])
    total_pages = (pass1_doc.end_page_number or 0) - (pass1_doc.start_page_number or 1) + 1
    print(
        f"\n  Document: {pass1_doc.mime_type or 'unknown'}  "
        f"pages {pass1_doc.start_page_number}–{pass1_doc.end_page_number}"
    )

    # ---- Detect table and figure pages ----
    t = time.perf_counter()
    table_pages = _detect_table_pages_from_markdown(pass1_doc)
    if not table_pages:
        print("  No tables detected in pass-1 response; scanning PDF for tables ...")
        table_pages = _detect_table_pages_pymupdf(pdf_path)
    detect_table_time = time.perf_counter() - t
    timings.append(("Table detection", detect_table_time))

    t = time.perf_counter()
    figure_pages = _detect_figure_pages(pass1_doc)
    if not figure_pages:
        print("  No figures detected in pass-1 response; scanning PDF for images ...")
        figure_pages = _detect_figure_pages_pymupdf(pdf_path)
    detect_fig_time = time.perf_counter() - t
    timings.append(("Figure detection", detect_fig_time))

    table_page_list = sorted(table_pages)
    figure_page_list = sorted(figure_pages)

    # Remove overlap: pages in both sets are processed by pass 3 (richer analyzer)
    table_only_pages = sorted(table_pages - figure_pages)
    overlap_pages = sorted(table_pages & figure_pages)

    print(f"\n  Table pages:  {len(table_page_list):>4}  "
          f"({', '.join(map(str, table_page_list[:15]))}{'...' if len(table_page_list) > 15 else ''})")
    print(f"  Figure pages: {len(figure_page_list):>4}  "
          f"({', '.join(map(str, figure_page_list[:15]))}{'...' if len(figure_page_list) > 15 else ''})")
    if overlap_pages:
        print(f"  Overlap:      {len(overlap_pages):>4}  "
              f"(pages with both tables+figures → handled in pass 3)")
    print(f"  Text-only:    {total_pages - len(table_pages | figure_pages):>4}")

    # ==================================================================
    # PASS 2 — Table extraction (prebuilt-layout, table pages only)
    # ==================================================================
    pass2_results: list[AnalysisResult] = []
    pass2_op_ids: list[str] = []
    pass2_docs: list[DocumentContent] = []

    if table_only_pages:
        (pass2_results, pass2_op_ids, pass2_docs,
         pass2_wall, pass2_chunk_timings, pass2_usages) = _run_targeted_pass(
            endpoint, credential, pdf_path,
            table_only_pages,
            analyzer_id="prebuilt-layout",
            pass_label="Pass 2",
            max_workers=args.workers,
            group_size=args.table_group_size,
        )
        timings.append(("Pass 2 (tables, wall)", pass2_wall))
        timings.extend(pass2_chunk_timings)
        all_usages.extend(pass2_usages)
    else:
        print("\nPass 2 — skipped (no table-only pages detected).")

    # ==================================================================
    # PASS 3 — Figure analysis (prebuilt-documentSearch, figure pages)
    # ==================================================================
    pass3_results: list[AnalysisResult] = []
    pass3_op_ids: list[str] = []
    pass3_docs: list[DocumentContent] = []

    if figure_page_list:
        (pass3_results, pass3_op_ids, pass3_docs,
         pass3_wall, pass3_chunk_timings, pass3_usages) = _run_targeted_pass(
            endpoint, credential, pdf_path,
            figure_page_list,
            analyzer_id="prebuilt-documentSearch",
            pass_label="Pass 3",
            max_workers=args.workers,
            group_size=args.figure_group_size,
        )
        timings.append(("Pass 3 (figures, wall)", pass3_wall))
        timings.extend(pass3_chunk_timings)
        all_usages.extend(pass3_usages)
    else:
        print("\nPass 3 — skipped (no figure pages detected).")

    # ---- Save outputs ----
    print(f"\nWriting output to: {output_dir}\n")

    t = time.perf_counter()
    _save_json(pass1_result, pass2_results, pass3_results, output_dir)
    timings.append(("Save JSON", time.perf_counter() - t))

    t = time.perf_counter()
    _save_markdown(pass1_doc, output_dir)
    timings.append(("Save markdown", time.perf_counter() - t))

    if args.save_extras:
        # Tables come from pass 2 (table-only pages) + pass 3 (overlap pages)
        t = time.perf_counter()
        all_table_docs = pass2_docs + [d for d in pass3_docs if d.tables]
        _save_tables(all_table_docs, output_dir)
        timings.append(("Save tables", time.perf_counter() - t))

        if pass3_docs:
            t = time.perf_counter()
            _save_figures(client, pass3_docs, pass3_op_ids, output_dir)
            timings.append(("Save figures", time.perf_counter() - t))

    # ---- Summary ----
    total = time.perf_counter() - t_global
    _SEP = "\u2500" * 60
    print(f"\n{_SEP}")
    print("  Timing summary")
    print(_SEP)
    for label, elapsed in timings:
        print(f"  {label:<35s}  {elapsed:9.2f}s  ({_fmt_elapsed(elapsed)})")
    print(_SEP)
    print(f"  {'Total':<35s}  {total:9.2f}s  ({_fmt_elapsed(total)})")
    print(f"\n  Pages processed:")
    print(f"    Pass 1 (text):    {total_pages:>4} pages  (prebuilt-read)")
    print(f"    Pass 2 (tables):  {len(table_only_pages):>4} pages  (prebuilt-layout)")
    print(f"    Pass 3 (figures): {len(figure_page_list):>4} pages  (prebuilt-documentSearch)")
    print(f"    Total API pages:  {total_pages + len(table_only_pages) + len(figure_page_list):>4}")
    print(_SEP)

    _print_usage(_merge_usage(all_usages), "Usage details (all passes combined)")

    print("\nDone.")


if __name__ == "__main__":
    main()
