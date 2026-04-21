#!/usr/bin/env python3
"""
Two-pass PDF processing with Azure Content Understanding.

Pass 1 — Fast layout extraction (prebuilt-read) on the full document.
          Identifies which pages contain figures, charts, or diagrams.

Pass 2 — Targeted rich analysis (prebuilt-documentSearch) only on
          figure-bearing pages (submitted in parallel).  Provides AI
          descriptions, Chart.js data, and Mermaid diagrams.

The final output merges pass-1 markdown/tables with pass-2 figure data,
giving the best of both: fast text extraction and rich figure analysis
only where needed.

Usage
-----
    python process_document_twopass.py report.pdf
    python process_document_twopass.py report.pdf --workers 8
    python process_document_twopass.py report.pdf --save-extras

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

import pymupdf  # PyMuPDF
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
# Pass 1 — Layout extraction (prebuilt-read, full document)
# ---------------------------------------------------------------------------


def _pass1_layout(
    client: ContentUnderstandingClient,
    pdf_bytes: bytes,
) -> tuple[AnalysisResult, str, float, dict | None]:
    """Run prebuilt-read on the full document for fast layout extraction.

    Returns (result, operation_id, elapsed_secs, usage).
    """
    t0 = time.perf_counter()
    print("Pass 1 — submitting full document to prebuilt-read ...")

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
# Figure page detection from pass 1
# ---------------------------------------------------------------------------

# prebuilt-read markdown may embed figure references like  :figure:1  or
# <!-- FigureStart -->...<!-- FigureEnd --> depending on API version.
# The structured result also carries `figures` on DocumentContent.
_FIGURE_REF_RE = re.compile(
    r":figure:\d+|<!-- *Figure|!\[.*?\]\(figures/",
    re.IGNORECASE,
)


def _detect_figure_pages(doc: DocumentContent) -> set[int]:
    """Return 1-based page numbers that contain figures.

    Uses the structured figures list (with page numbers) when available,
    falling back to regex on markdown content.
    """
    pages: set[int] = set()

    # Structured figures with page references
    if doc.figures:
        for fig in doc.figures:
            if fig.source:
                # source typically contains page info like "page 5"
                m = re.search(r"page[:\s]*(\d+)", str(fig.source), re.IGNORECASE)
                if m:
                    pages.add(int(m.group(1)))

    # Fallback: scan markdown for figure markers per page section
    if not pages and doc.markdown:
        # If the markdown has page breaks, use them to identify figure pages.
        # Common format: "<!-- PageBreak -->" or "---" as a separator
        page_sections = re.split(
            r"<!-- *PageBreak *-->|<!-- *PageNumber=\"(\d+)\" *-->",
            doc.markdown,
        )
        for i, section in enumerate(page_sections):
            if section and _FIGURE_REF_RE.search(section):
                pages.add(i + 1)

    # If we still couldn't map to pages, use PyMuPDF image detection as fallback
    return pages


def _detect_figure_pages_pymupdf(pdf_path: Path) -> set[int]:
    """Detect pages with images using PyMuPDF as a last-resort fallback."""
    doc = pymupdf.open(pdf_path)
    pages: set[int] = set()
    for idx in range(len(doc)):
        page = doc[idx]
        images = page.get_images(full=True)
        if images:
            page_area = page.rect.width * page.rect.height or 1.0
            total_area = 0.0
            for img_info in images:
                try:
                    for r in page.get_image_rects(img_info[0]):
                        total_area += r.width * r.height
                except Exception:  # noqa: BLE001
                    total_area += page_area * 0.25
            # Only count pages where images occupy >5% of the page
            # (filters out tiny logos/decorations)
            if total_area / page_area > 0.05:
                pages.add(idx + 1)
    doc.close()
    return pages


# ---------------------------------------------------------------------------
# Pass 2 — Targeted rich analysis (prebuilt-documentSearch)
# ---------------------------------------------------------------------------


def _build_page_pdf(pdf_path: Path, page_numbers: list[int]) -> bytes:
    """Extract specific 1-based page numbers into a new PDF."""
    doc = pymupdf.open(pdf_path)
    out = pymupdf.open()
    for pn in page_numbers:
        out.insert_pdf(doc, from_page=pn - 1, to_page=pn - 1)
    buf = io.BytesIO()
    out.save(buf)
    out.close()
    doc.close()
    return buf.getvalue()


def _analyze_chunk(
    endpoint: str,
    credential: DefaultAzureCredential,
    chunk_bytes: bytes,
    chunk_index: int,
    page_label: str,
) -> tuple[int, AnalysisResult, str, float, dict | None]:
    client = ContentUnderstandingClient(endpoint=endpoint, credential=credential)

    t0 = time.perf_counter()
    print(f"  [pass2 chunk {chunk_index:>2}] Submitting {page_label} "
          f"({len(chunk_bytes):,} bytes) ...")

    poller = client.begin_analyze_binary(
        analyzer_id="prebuilt-documentSearch",
        binary_input=chunk_bytes,
    )
    op_id: str = poller.operation_id
    print(f"  [pass2 chunk {chunk_index:>2}] Operation {op_id}  (polling ...)")

    result: AnalysisResult = poller.result()
    elapsed = time.perf_counter() - t0
    print(f"  [pass2 chunk {chunk_index:>2}] Complete — {page_label} "
          f"in {_fmt_elapsed(elapsed)}")

    usage = _extract_usage(poller)
    return chunk_index, result, op_id, elapsed, usage


def _group_consecutive(pages: list[int], max_group: int = 4) -> list[list[int]]:
    """Group sorted page numbers into consecutive runs, max *max_group* pages each."""
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
    output_dir: Path,
) -> None:
    path = output_dir / "result.json"
    payload = {
        "pass1_layout": pass1_result.as_dict(),
        "pass2_figures": [r.as_dict() for r in pass2_results],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
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
            grid: dict[tuple[int, int], str] = {}
            for cell in table.cells or []:
                grid[(cell.row_index, cell.column_index)] = cell.content.replace("|", "\\|").replace("\n", " ")
            cols = range(table.column_count)
            fh.write("| " + " | ".join(grid.get((0, c), "") for c in cols) + " |\n")
            fh.write("| " + " | ".join(["---"] * table.column_count) + " |\n")
            for row in range(1, table.row_count):
                fh.write("| " + " | ".join(grid.get((row, c), "") for c in cols) + " |\n")
            fh.write("\n")
    print(f"  [saved] {path}  ({len(doc.tables)} table(s))")


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
    pass2_docs: list[DocumentContent],
    pass2_op_ids: list[str],
    output_dir: Path,
) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    all_figures = [(fig, oid) for doc, oid in zip(pass2_docs, pass2_op_ids)
                   for fig in (doc.figures or [])]
    if not all_figures:
        print("  No figures found in pass 2; skipping figures/")
        return
    print(f"  Processing {len(all_figures)} figure(s) from pass 2 ...")
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
        description="Two-pass PDF processing: fast layout (prebuilt-read) + "
                    "targeted figure analysis (prebuilt-documentSearch).",
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
        help="Max concurrent pass-2 submissions (default: 8).",
    )
    parser.add_argument(
        "--figure-group-size", type=int, default=4,
        help="Max consecutive figure-pages per pass-2 chunk (default: 4).",
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

    t_start = time.perf_counter()

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

    # ==== PASS 1 — Fast layout ====
    pass1_result, pass1_op_id, pass1_time, pass1_usage = _pass1_layout(client, pdf_bytes)
    timings.append(("Pass 1 (layout)", pass1_time))
    all_usages.append(pass1_usage)

    if not pass1_result.contents:
        print("Error: empty response from pass 1.", file=sys.stderr)
        sys.exit(1)

    pass1_doc = cast(DocumentContent, pass1_result.contents[0])
    print(
        f"\n  Document: {pass1_doc.mime_type or 'unknown'}  "
        f"pages {pass1_doc.start_page_number}–{pass1_doc.end_page_number}"
    )

    # ---- Detect figure pages ----
    t = time.perf_counter()
    figure_pages = _detect_figure_pages(pass1_doc)
    if not figure_pages:
        # Fall back to PyMuPDF image detection
        print("  No figures detected in pass-1 response; scanning with PyMuPDF ...")
        figure_pages = _detect_figure_pages_pymupdf(pdf_path)
    detect_time = time.perf_counter() - t
    timings.append(("Figure detection", detect_time))

    total_pages = (pass1_doc.end_page_number or 0) - (pass1_doc.start_page_number or 1) + 1
    figure_page_list = sorted(figure_pages)
    print(f"  {len(figure_page_list)} of {total_pages} pages contain figures"
          f"  ({', '.join(map(str, figure_page_list[:20]))}{'...' if len(figure_page_list) > 20 else ''})\n")

    # ==== PASS 2 — Targeted rich analysis (only figure pages) ====
    pass2_results: list[AnalysisResult] = []
    pass2_op_ids: list[str] = []
    pass2_docs: list[DocumentContent] = []

    if figure_page_list:
        page_groups = _group_consecutive(figure_page_list, args.figure_group_size)
        print(f"Pass 2 — submitting {len(page_groups)} chunk(s) covering "
              f"{len(figure_page_list)} figure pages to prebuilt-documentSearch ...\n")

        # Build chunk PDFs
        chunk_data: list[tuple[bytes, str]] = []
        for grp in page_groups:
            chunk_pdf = _build_page_pdf(pdf_path, grp)
            label = f"p{grp[0]}" if len(grp) == 1 else f"p{grp[0]}–{grp[-1]}"
            chunk_data.append((chunk_pdf, label))

        pass2_chunk_results: dict[int, tuple[int, AnalysisResult, str, float, dict | None]] = {}
        t_pass2 = time.perf_counter()

        workers = min(args.workers, len(chunk_data))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _analyze_chunk, endpoint, credential,
                    chunk_pdf, idx, label,
                ): idx
                for idx, (chunk_pdf, label) in enumerate(chunk_data)
            }
            for future in as_completed(futures):
                idx, result, op_id, elapsed, usage = future.result()
                pass2_chunk_results[idx] = (idx, result, op_id, elapsed, usage)

        pass2_time = time.perf_counter() - t_pass2
        timings.append(("Pass 2 (figures, wall)", pass2_time))

        # Order by chunk index
        for idx in range(len(chunk_data)):
            _, result, op_id, elapsed, usage = pass2_chunk_results[idx]
            pass2_results.append(result)
            pass2_op_ids.append(op_id)
            all_usages.append(usage)
            label = chunk_data[idx][1]
            timings.append((f"  pass2 chunk {idx} ({label})", elapsed))

            if result.contents:
                pass2_docs.append(cast(DocumentContent, result.contents[0]))

        seq_time = sum(pass2_chunk_results[i][3] for i in range(len(chunk_data)))
        print(f"\nPass 2 complete in {_fmt_elapsed(pass2_time)} wall-clock  "
              f"(sequential sum: {_fmt_elapsed(seq_time)}, "
              f"speedup: {seq_time / pass2_time:.1f}x)\n")
    else:
        print("Pass 2 — skipped (no figure pages detected).\n")

    # ---- Save outputs ----
    print(f"Writing output to: {output_dir}\n")

    t = time.perf_counter()
    _save_json(pass1_result, pass2_results, output_dir)
    timings.append(("Save JSON", time.perf_counter() - t))

    t = time.perf_counter()
    _save_markdown(pass1_doc, output_dir)
    timings.append(("Save markdown", time.perf_counter() - t))

    if args.save_extras:
        t = time.perf_counter()
        _save_tables(pass1_doc, output_dir)
        timings.append(("Save tables", time.perf_counter() - t))

        if pass2_docs:
            t = time.perf_counter()
            _save_figures(client, pass2_docs, pass2_op_ids, output_dir)
            timings.append(("Save figures", time.perf_counter() - t))

    # ---- Summary ----
    total = time.perf_counter() - t_start
    _SEP = "\u2500" * 60
    print(f"\n{_SEP}")
    print("  Timing summary")
    print(_SEP)
    for label, elapsed in timings:
        print(f"  {label:<30s}  {elapsed:9.2f}s  ({_fmt_elapsed(elapsed)})")
    print(_SEP)
    print(f"  {'Total':<30s}  {total:9.2f}s  ({_fmt_elapsed(total)})")
    print(_SEP)

    _print_usage(_merge_usage(all_usages), "Usage details (pass 1 + pass 2 combined)")

    print("\nDone.")


if __name__ == "__main__":
    main()
