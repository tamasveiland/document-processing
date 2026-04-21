#!/usr/bin/env python3
"""
Per-page / per-range PDF processing with Azure Content Understanding.

Submission strategy depends on how pages are specified:

  - Individual pages (e.g. ``5 25 42``) → one API call **per page**.
  - Page ranges   (e.g. ``15-18``)      → pages extracted into a single
    PDF and submitted as **one API call**.
  - Mixed         (e.g. ``5 15-18 42``) → three API calls: page 5,
    pages 15–18 together, page 42.

All submissions run in parallel.  Detailed per-unit metrics are printed
at the end.

Usage
-----
    # Individual pages — one API call each
    python process_document_pages.py report.pdf --pages 5 25 42

    # Range — single API call for pages 15-18
    python process_document_pages.py report.pdf --pages 15-18

    # Mixed — 3 API calls: p5, p15–18, p42
    python process_document_pages.py report.pdf --pages 5 15-18 42

    # All pages (one call per page)
    python process_document_pages.py report.pdf --all

    # Control parallelism
    python process_document_pages.py report.pdf --pages 1-20 --workers 4

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
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
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


@dataclass
class SubmissionUnit:
    """A group of pages to submit as a single API call."""
    pages: list[int]           # 1-based page numbers
    label: str = ""            # human-readable label (e.g. "p5" or "p15–18")
    pdf_bytes: bytes = b""     # extracted PDF
    # filled after analysis:
    result: AnalysisResult | None = None
    op_id: str = ""
    elapsed: float = 0.0
    usage: dict | None = None
    doc: DocumentContent | None = None

    def __post_init__(self) -> None:
        if not self.label:
            if len(self.pages) == 1:
                self.label = f"p{self.pages[0]}"
            else:
                self.label = f"p{self.pages[0]}–{self.pages[-1]}"


def _parse_page_spec(spec: str, max_page: int) -> list[int]:
    """Parse a page specification like '5', '1-10', or '3-' into page numbers."""
    spec = spec.strip()
    if "-" in spec:
        parts = spec.split("-", 1)
        start = int(parts[0]) if parts[0] else 1
        end = int(parts[1]) if parts[1] else max_page
        if start < 1 or end > max_page or start > end:
            raise ValueError(
                f"Invalid page range '{spec}': must be within 1–{max_page}"
            )
        return list(range(start, end + 1))
    else:
        p = int(spec)
        if p < 1 or p > max_page:
            raise ValueError(f"Page {p} out of range (1–{max_page})")
        return [p]


def _build_submission_units(
    page_specs: list[str] | None,
    all_pages: bool,
    max_page: int,
) -> list[SubmissionUnit]:
    """Build submission units from CLI page specs.

    - Individual page numbers become one unit each (one API call per page).
    - Ranges (e.g. '15-18') become a single unit (one API call for the range).
    - ``--all`` creates one unit per page.
    """
    if all_pages:
        return [SubmissionUnit(pages=[p]) for p in range(1, max_page + 1)]
    if not page_specs:
        raise SystemExit("Error: provide --pages or --all.")

    units: list[SubmissionUnit] = []
    for spec in page_specs:
        pages = _parse_page_spec(spec, max_page)
        if "-" in spec.strip():
            # Range spec → single submission unit
            units.append(SubmissionUnit(pages=pages))
        else:
            # Individual page → its own unit
            units.append(SubmissionUnit(pages=pages))
    return units


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------


def _get_page_count(pdf_path: Path) -> int:
    doc = pymupdf.open(pdf_path)
    count = len(doc)
    doc.close()
    return count


def _extract_pages(pdf_path: Path, page_numbers: list[int]) -> bytes:
    """Extract 1-based page numbers into a new in-memory PDF."""
    doc = pymupdf.open(pdf_path)
    out = pymupdf.open()
    for pn in page_numbers:
        out.insert_pdf(doc, from_page=pn - 1, to_page=pn - 1)
    buf = io.BytesIO()
    out.save(buf)
    out.close()
    doc.close()
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _analyze_unit(
    endpoint: str,
    credential: DefaultAzureCredential,
    analyzer_id: str,
    unit_index: int,
    unit_label: str,
    pdf_bytes: bytes,
) -> tuple[int, AnalysisResult, str, float, dict | None]:
    """Submit a PDF (single- or multi-page) and return results with metrics.

    Returns (unit_index, result, operation_id, elapsed_secs, usage).
    """
    client = ContentUnderstandingClient(endpoint=endpoint, credential=credential)

    t0 = time.perf_counter()
    print(f"  [{unit_label:>12}] Submitting ({len(pdf_bytes):,} bytes) ...")

    poller = client.begin_analyze_binary(
        analyzer_id=analyzer_id,
        binary_input=pdf_bytes,
    )
    op_id: str = poller.operation_id
    print(f"  [{unit_label:>12}] Operation {op_id}  (polling ...)")

    result: AnalysisResult = poller.result()
    elapsed = time.perf_counter() - t0
    print(f"  [{unit_label:>12}] Complete in {_fmt_elapsed(elapsed)}")

    usage = _extract_usage(poller)
    return unit_index, result, op_id, elapsed, usage


# ---------------------------------------------------------------------------
# Usage helpers
# ---------------------------------------------------------------------------


def _extract_usage(poller) -> dict | None:
    try:
        return (
            poller.polling_method()
            ._pipeline_response.http_response.json()
            .get("usage")
        )
    except Exception:  # noqa: BLE001
        return None


def _merge_usage(usages: list[dict | None]) -> dict | None:
    merged: dict = {}
    for u in usages:
        if not u:
            continue
        for key in (
            "documentPagesMinimal",
            "documentPagesBasic",
            "documentPagesStandard",
            "contextualizationTokens",
            "contextualizationToken",
        ):
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


def _total_tokens(usage: dict | None) -> int:
    if not usage:
        return 0
    ctx = usage.get("contextualizationTokens") or usage.get("contextualizationToken") or 0
    tok = sum((usage.get("tokens") or {}).values())
    return ctx + tok


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

    ctx_tokens = (
        usage.get("contextualizationTokens")
        or usage.get("contextualizationToken")
        or 0
    )
    tokens = usage.get("tokens") or {}
    if ctx_tokens or tokens:
        print("  Tokens:")
        if ctx_tokens:
            print(f"    {'Contextualization':<33s}: {ctx_tokens:>10,}")
        for key, count in sorted(tokens.items()):
            print(f"    {key:<33s}: {count:>10,}")
        total = sum(tokens.values()) + ctx_tokens
        print(f"    {'TOTAL':<33s}: {total:>10,}")
    print(_SEP)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _save_json(
    units: list[SubmissionUnit],
    output_dir: Path,
) -> None:
    path = output_dir / "result.json"
    payload = {}
    for unit in units:
        key = f"pages_{'_'.join(map(str, unit.pages))}"
        payload[key] = unit.result.as_dict() if unit.result else None
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
    print(f"  [saved] {path}")


def _save_markdown(
    units: list[SubmissionUnit],
    output_dir: Path,
) -> None:
    path = output_dir / "document.md"
    with open(path, "w", encoding="utf-8") as fh:
        for unit in units:
            if not unit.doc:
                continue
            page_str = ', '.join(map(str, unit.pages))
            fh.write(f"<!-- Pages: {page_str} -->\n\n")
            fh.write(unit.doc.markdown or "")
            fh.write("\n\n<!-- PageBreak -->\n\n")
    print(f"  [saved] {path}")


def _save_tables(
    units: list[SubmissionUnit],
    output_dir: Path,
) -> None:
    all_tables: list[tuple[str, int, object]] = []
    for unit in units:
        if not unit.doc:
            continue
        for idx, table in enumerate(unit.doc.tables or [], start=1):
            all_tables.append((unit.label, idx, table))
    if not all_tables:
        print("  No structured tables found; skipping tables.md")
        return
    path = output_dir / "tables.md"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Extracted Tables\n\n")
        for label, idx, table in all_tables:
            caption_text = (
                table.caption.content if table.caption else f"Table {idx} ({label})"
            )
            fh.write(f"## {caption_text}\n\n")
            fh.write(
                f"*{label} — {table.row_count} rows \u00d7 "
                f"{table.column_count} columns*\n\n"
            )
            grid: dict[tuple[int, int], str] = {}
            for cell in table.cells or []:
                grid[(cell.row_index, cell.column_index)] = (
                    cell.content.replace("|", "\\|").replace("\n", " ")
                )
            cols = range(table.column_count)
            fh.write(
                "| " + " | ".join(grid.get((0, c), "") for c in cols) + " |\n"
            )
            fh.write(
                "| " + " | ".join(["---"] * table.column_count) + " |\n"
            )
            for row in range(1, table.row_count):
                fh.write(
                    "| "
                    + " | ".join(grid.get((row, c), "") for c in cols)
                    + " |\n"
                )
            fh.write("\n")
    print(f"  [saved] {path}  ({len(all_tables)} table(s))")


def _safe_stem(figure_id: str) -> str:
    return figure_id.replace("/", "_").replace(":", "_").replace(".", "_")


def _save_figures(
    client: ContentUnderstandingClient,
    units: list[SubmissionUnit],
    output_dir: Path,
) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    all_figures: list[tuple[str, DocumentFigure, str]] = []
    for unit in units:
        if not unit.doc:
            continue
        for fig in unit.doc.figures or []:
            all_figures.append((unit.label, fig, unit.op_id))
    if not all_figures:
        print("  No figures found; skipping figures/")
        return
    print(f"  Processing {len(all_figures)} figure(s) ...")
    for label, figure, op_id in all_figures:
        stem = _safe_stem(figure.id)
        safe_label = label.replace("\u2013", "-")
        # Description
        desc_path = figures_dir / f"{safe_label}_figure_{stem}.md"
        lines: list[str] = [f"# Figure `{figure.id}` ({label})\n\n"]
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
        with open(desc_path, "w", encoding="utf-8") as fh:
            fh.writelines(lines)
        print(f"    [saved] {desc_path.name}")
        # Image
        try:
            response = client.get_result_file(
                operation_id=op_id, path=f"figures/{figure.id}"
            )
            image_bytes = b"".join(response)
            if image_bytes:
                img_path = figures_dir / f"{safe_label}_figure_{stem}.png"
                with open(img_path, "wb") as fh:
                    fh.write(image_bytes)
                print(f"    [saved] {img_path.name}  ({len(image_bytes):,} bytes)")
        except Exception as exc:  # noqa: BLE001
            print(f"    [info]  Image not available for {figure.id}: {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process specific PDF pages with Azure Content "
        "Understanding.  Individual pages get one API call each; "
        "page ranges are submitted as a single API call.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("pdf_path", help="Path to the input PDF file.")
    parser.add_argument(
        "--pages",
        "-p",
        nargs="+",
        metavar="SPEC",
        help="Page numbers or ranges (e.g. 5 25 42 → 3 calls; "
        "15-18 → 1 call; 5 15-18 42 → 3 calls).",
    )
    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        default=False,
        help="Process all pages (one API call per page).",
    )
    parser.add_argument(
        "--analyzer",
        default="prebuilt-documentSearch",
        help="Analyzer ID (default: prebuilt-documentSearch).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="output",
        help="Directory for output files (default: ./output).",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=8,
        help="Max concurrent API submissions (default: 8).",
    )
    parser.add_argument(
        "--save-extras",
        action="store_true",
        default=False,
        help="Also save tables.md and figures/*.",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path).resolve()
    if not pdf_path.is_file():
        print(f"Error: file not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    total_doc_pages = _get_page_count(pdf_path)

    try:
        units = _build_submission_units(args.pages, args.all, total_doc_pages)
    except (ValueError, SystemExit) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    all_pages = sorted({p for u in units for p in u.pages})

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    load_dotenv()
    endpoint = os.environ.get("AZURE_CONTENT_UNDERSTANDING_ENDPOINT", "")
    if not endpoint:
        print(
            "Error: AZURE_CONTENT_UNDERSTANDING_ENDPOINT is not set.",
            file=sys.stderr,
        )
        sys.exit(1)
    credential = DefaultAzureCredential()
    client = ContentUnderstandingClient(endpoint=endpoint, credential=credential)

    workers = min(args.workers, len(units))
    print(f"Document: {pdf_path.name}  ({total_doc_pages} pages total)")
    print(f"Pages:    {len(all_pages)} across {len(units)} API call(s)")
    for i, u in enumerate(units):
        kind = "range" if len(u.pages) > 1 else "single"
        print(f"  call {i + 1}: {u.label}  ({len(u.pages)} page(s), {kind})")
    print(f"Analyzer: {args.analyzer}")
    print(f"Workers:  {workers}\n")

    t_global = time.perf_counter()

    # ---- Extract PDFs for each submission unit ----
    for unit in units:
        unit.pdf_bytes = _extract_pages(pdf_path, unit.pages)

    # ---- Submit units in parallel ----
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _analyze_unit,
                endpoint,
                credential,
                args.analyzer,
                idx,
                unit.label,
                unit.pdf_bytes,
            ): idx
            for idx, unit in enumerate(units)
        }
        for future in as_completed(futures):
            idx, result, op_id, elapsed, usage = future.result()
            unit = units[idx]
            unit.result = result
            unit.op_id = op_id
            unit.elapsed = elapsed
            unit.usage = usage
            if result.contents:
                unit.doc = cast(DocumentContent, result.contents[0])

    wall_time = time.perf_counter() - t_global

    # ---- Save outputs ----
    print(f"\nWriting output to: {output_dir}\n")
    _save_json(units, output_dir)
    _save_markdown(units, output_dir)

    if args.save_extras:
        _save_tables(units, output_dir)
        _save_figures(client, units, output_dir)

    # ---- Per-unit metrics table ----
    _SEP = "\u2500" * 92
    print(f"\n{_SEP}")
    print("  Per-submission metrics")
    print(_SEP)
    print(
        f"  {'Unit':<14s}  {'Pages':>5s}  {'Size':>10s}  {'Time':>10s}  "
        f"{'Tables':>7s}  {'Figures':>8s}  {'Tokens':>10s}"
    )
    print(
        f"  {'─' * 14}  {'─' * 5}  {'─' * 10}  {'─' * 10}  "
        f"{'─' * 7}  {'─' * 8}  {'─' * 10}"
    )

    total_pages_count = 0
    total_size = 0
    total_tables = 0
    total_figures = 0
    total_tokens_all = 0

    for unit in units:
        n_pages = len(unit.pages)
        size = len(unit.pdf_bytes)
        n_tables = len(unit.doc.tables) if unit.doc and unit.doc.tables else 0
        n_figures = len(unit.doc.figures) if unit.doc and unit.doc.figures else 0
        tokens = _total_tokens(unit.usage)
        total_pages_count += n_pages
        total_size += size
        total_tables += n_tables
        total_figures += n_figures
        total_tokens_all += tokens
        print(
            f"  {unit.label:<14s}  {n_pages:>5d}  {size:>9,}B  "
            f"{unit.elapsed:>9.2f}s  "
            f"{n_tables:>7d}  {n_figures:>8d}  {tokens:>10,}"
        )

    print(
        f"  {'─' * 14}  {'─' * 5}  {'─' * 10}  {'─' * 10}  "
        f"{'─' * 7}  {'─' * 8}  {'─' * 10}"
    )

    seq_sum = sum(u.elapsed for u in units)
    avg_time = seq_sum / len(units) if units else 0
    fastest = min(units, key=lambda u: u.elapsed) if units else None
    slowest = max(units, key=lambda u: u.elapsed) if units else None

    print(
        f"  {'Total':<14s}  {total_pages_count:>5d}  {total_size:>9,}B  "
        f"{seq_sum:>9.2f}s  "
        f"{total_tables:>7d}  {total_figures:>8d}  {total_tokens_all:>10,}"
    )
    print(_SEP)

    # ---- Timing summary ----
    print(f"\n{_SEP}")
    print("  Timing summary")
    print(_SEP)
    print(f"  API calls            : {len(units)}")
    print(f"  Total pages          : {total_pages_count}")
    print(f"  Workers              : {workers}")
    print(f"  Wall-clock time      : {_fmt_elapsed(wall_time):>10s}")
    print(f"  Sequential sum       : {_fmt_elapsed(seq_sum):>10s}")
    if wall_time > 0:
        print(f"  Speedup              : {seq_sum / wall_time:.1f}x")
    print(f"  Avg per call         : {_fmt_elapsed(avg_time):>10s}")
    if fastest:
        print(f"  Fastest              : {fastest.label} — {_fmt_elapsed(fastest.elapsed)}")
    if slowest:
        print(f"  Slowest              : {slowest.label} — {_fmt_elapsed(slowest.elapsed)}")
    print(_SEP)

    # ---- Aggregated usage ----
    _print_usage(
        _merge_usage([u.usage for u in units]),
        "Aggregated usage",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
