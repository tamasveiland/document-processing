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
import asyncio
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import cast

import pypdfium2 as pdfium  # PDF operations (Apache-2.0)
import pypdfium2.raw as pdfium_c
from azure.ai.contentunderstanding.aio import ContentUnderstandingClient
from azure.ai.contentunderstanding.models import (
    AnalysisResult,
    DocumentChartFigure,
    DocumentContent,
    DocumentFigure,
    DocumentMermaidFigure,
)
from azure.identity.aio import DefaultAzureCredential
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


async def _pass1_layout(
    client: ContentUnderstandingClient,
    pdf_bytes: bytes,
) -> tuple[AnalysisResult, str, float, dict | None]:
    """Run prebuilt-read on the full document for fast layout extraction.

    Returns (result, operation_id, elapsed_secs, usage).
    """
    t0 = time.perf_counter()
    print("Pass 1 — submitting full document to prebuilt-layout ...")

    poller = await client.begin_analyze_binary(
        analyzer_id="prebuilt-layout",
        binary_input=pdf_bytes,
    )
    op_id: str = poller.operation_id
    print(f"  Operation {op_id}  (polling ...)")

    result: AnalysisResult = await poller.result()
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
        # NOTE: use non-capturing group (?:...) so re.split doesn't inject
        #       captured values into the result list and inflate indices.
        page_sections = re.split(
            r"<!-- *PageBreak *-->|<!-- *PageNumber=\"(?:\d+)\" *-->",
            doc.markdown,
        )
        for i, section in enumerate(page_sections):
            if section and _FIGURE_REF_RE.search(section):
                pages.add(i + 1)

    # If we still couldn't map to pages, use PyMuPDF image detection as fallback
    return pages


# def _detect_figure_pages_pymupdf(pdf_path: Path) -> set[int]:
#     """Detect pages with images using pypdfium2 as a last-resort fallback."""
#     doc = pdfium.PdfDocument(pdf_path)
#     pages: set[int] = set()
#     for idx in range(len(doc)):
#         page = doc[idx]
#         page_area = page.get_width() * page.get_height() or 1.0
#         total_area = 0.0
#         image_count = 0
#         for obj in page.get_objects():
#             if obj.type == pdfium_c.FPDF_PAGEOBJ_IMAGE:
#                 image_count += 1
#                 try:
#                     left, bottom, right, top = obj.get_pos()
#                     total_area += abs((right - left) * (top - bottom))
#                 except Exception:  # noqa: BLE001
#                     total_area += page_area * 0.25
#         # Only count pages where images occupy >5% of the page
#         # (filters out tiny logos/decorations)
#         if image_count > 0 and total_area / page_area > 0.05:
#             pages.add(idx + 1)
#         page.close()
#     doc.close()
#     return pages


# ---------------------------------------------------------------------------
# Pass 2 — Targeted rich analysis (prebuilt-documentSearch)
# ---------------------------------------------------------------------------


def _build_page_pdf(src: pdfium.PdfDocument, page_numbers: list[int]) -> bytes:
    """Extract specific 1-based page numbers from *src* into a new PDF."""
    out = pdfium.PdfDocument.new()
    out.import_pages(src, [pn - 1 for pn in page_numbers])
    buf = io.BytesIO()
    out.save(buf)
    out.close()
    return buf.getvalue()


async def _analyze_chunk(
    client: ContentUnderstandingClient,
    chunk_bytes: bytes,
    chunk_index: int,
    page_label: str,
    semaphore: asyncio.Semaphore,
    timeout: float = 300,
) -> tuple[int, AnalysisResult, str, float, dict | None]:
    async with semaphore:
        t0 = time.perf_counter()
        print(f"  [pass2 chunk {chunk_index:>2}] Submitting {page_label} "
              f"({len(chunk_bytes):,} bytes) ...")

        poller = await client.begin_analyze_binary(
            analyzer_id="prebuilt-documentSearch",
            binary_input=chunk_bytes,
        )
        op_id: str = poller.operation_id
        print(f"  [pass2 chunk {chunk_index:>2}] Operation {op_id}  (polling ...)")

        result: AnalysisResult = await asyncio.wait_for(
            poller.result(), timeout=timeout,
        )
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


def _batch_pages(pages: list[int], batch_size: int) -> list[list[int]]:
    """Split sorted page numbers into fixed-size batches (pages need not be consecutive)."""
    return [pages[i:i + batch_size] for i in range(0, len(pages), batch_size)]


def _page_label(grp: list[int]) -> str:
    """Human-readable label for a group of page numbers."""
    if len(grp) == 1:
        return f"p{grp[0]}"
    if grp[-1] == grp[0] + len(grp) - 1:  # consecutive
        return f"p{grp[0]}\u2013{grp[-1]}"
    return f"p{','.join(map(str, grp))}"
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


async def _try_download_figure_image(
    client: ContentUnderstandingClient,
    operation_id: str,
    figure: DocumentFigure,
    figures_dir: Path,
    semaphore: asyncio.Semaphore,
) -> str | None:
    """Download one figure image asynchronously. Returns a status message."""
    async with semaphore:
        try:
            response = await client.get_result_file(operation_id=operation_id, path=f"figures/{figure.id}")
            chunks: list[bytes] = []
            async for chunk in response:
                chunks.append(chunk)
            image_bytes = b"".join(chunks)
            if not image_bytes:
                return None
            stem = _safe_stem(figure.id)
            out_path = figures_dir / f"figure_{stem}.png"
            with open(out_path, "wb") as fh:
                fh.write(image_bytes)
            return f"    [saved] {out_path.name}  ({len(image_bytes):,} bytes)"
        except Exception as exc:  # noqa: BLE001
            return f"    [info]  Image not available for figure {figure.id}: {exc}"


async def _save_figures(
    client: ContentUnderstandingClient,
    pass2_docs: list[DocumentContent],
    pass2_op_ids: list[str],
    output_dir: Path,
    max_workers: int = 16,
) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    all_figures = [(fig, oid) for doc, oid in zip(pass2_docs, pass2_op_ids)
                   for fig in (doc.figures or [])]
    if not all_figures:
        print("  No figures found in pass 2; skipping figures/")
        return
    print(f"  Processing {len(all_figures)} figure(s) from pass 2 "
          f"(up to {max_workers} concurrent downloads) ...")

    # Save descriptions synchronously (fast, local I/O)
    for figure, _op_id in all_figures:
        _save_figure_description(figure, figures_dir)

    # Download images concurrently
    semaphore = asyncio.Semaphore(max_workers)
    tasks = [
        _try_download_figure_image(client, op_id, figure, figures_dir, semaphore)
        for figure, op_id in all_figures
    ]
    dl_results = await asyncio.gather(*tasks)
    for msg in dl_results:
        if msg:
            print(msg)


# ---------------------------------------------------------------------------
# Main async workflow
# ---------------------------------------------------------------------------


async def _async_main(args: argparse.Namespace) -> None:
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
    async with ContentUnderstandingClient(
        endpoint=endpoint, credential=credential,
    ) as client:

        all_usages: list[dict | None] = []
        timings: list[tuple[str, float]] = []

        # ==== PASS 1 — Fast layout ====
        pass1_result, pass1_op_id, pass1_time, pass1_usage = await _pass1_layout(client, pdf_bytes)
        timings.append(("Pass 1 (layout)", pass1_time))
        all_usages.append(pass1_usage)

        if not pass1_result.contents:
            print("Error: empty response from pass 1.", file=sys.stderr)
            sys.exit(1)

        pass1_doc = cast(DocumentContent, pass1_result.contents[0])
        print(
            f"\n  Document: {pass1_doc.mime_type or 'unknown'}  "
            f"pages {pass1_doc.start_page_number}\u2013{pass1_doc.end_page_number}"
        )

        # ---- Detect figure pages ----
        t = time.perf_counter()
        figure_pages = _detect_figure_pages(pass1_doc)
        if not figure_pages:
            # Fall back to pypdfium2 image detection
            print("  No figures detected in pass-1 response; scanning PDF for images ...")
            # figure_pages = _detect_figure_pages_pymupdf(pdf_path)
        detect_time = time.perf_counter() - t
        timings.append(("Figure detection", detect_time))

        total_pages = (pass1_doc.end_page_number or 0) - (pass1_doc.start_page_number or 1) + 1
        figure_page_list = sorted(p for p in figure_pages if 1 <= p <= total_pages)
        print(f"  {len(figure_page_list)} of {total_pages} pages contain figures"
              f"  ({', '.join(map(str, figure_page_list[:20]))}{'...' if len(figure_page_list) > 20 else ''})\n")

        # ---- Start saving pass-1 outputs while pass-2 runs ----
        async def _save_pass1_outputs() -> list[tuple[str, float]]:
            """Save pass-1 outputs (runs concurrently with pass 2)."""
            t_timings: list[tuple[str, float]] = []
            t0 = time.perf_counter()
            _save_markdown(pass1_doc, output_dir)
            t_timings.append(("Save markdown", time.perf_counter() - t0))
            if args.save_extras:
                t0 = time.perf_counter()
                _save_tables(pass1_doc, output_dir)
                t_timings.append(("Save tables", time.perf_counter() - t0))
            return t_timings

        save_pass1_task = asyncio.create_task(_save_pass1_outputs())

        # ==== PASS 2 — Targeted rich analysis (only figure pages) ====
        pass2_results: list[AnalysisResult] = []
        pass2_op_ids: list[str] = []
        pass2_docs: list[DocumentContent] = []
        chunk_labels: list[str] = []

        if figure_page_list:
            page_groups = _batch_pages(figure_page_list, args.figure_group_size)
            print(f"Pass 2 \u2014 submitting {len(page_groups)} chunk(s) covering "
                  f"{len(figure_page_list)} figure pages to prebuilt-documentSearch ...\n")

            workers = min(args.workers, len(page_groups))
            semaphore = asyncio.Semaphore(workers)

            # Pre-build all chunk PDFs (sync) before async submission,
            # so all pypdfium2 C-library calls finish before the event
            # loop processes HTTP tasks.
            t_pass2 = time.perf_counter()
            src = pdfium.PdfDocument(pdf_path)
            chunk_data: list[tuple[bytes, str]] = []
            for grp in page_groups:
                chunk_pdf = _build_page_pdf(src, grp)
                label = _page_label(grp)
                chunk_data.append((chunk_pdf, label))
            src.close()

            # Now submit all chunks asynchronously
            chunk_timeout = args.chunk_timeout
            tasks: list[asyncio.Task] = []
            for idx, (chunk_bytes, label) in enumerate(chunk_data):
                chunk_labels.append(label)
                task = asyncio.create_task(
                    _analyze_chunk(client, chunk_bytes, idx, label, semaphore,
                                   timeout=chunk_timeout)
                )
                tasks.append(task)
                await asyncio.sleep(0)  # yield so earlier tasks can start I/O

            chunk_raw_results = await asyncio.gather(*tasks, return_exceptions=True)
            pass2_time = time.perf_counter() - t_pass2
            timings.append(("Pass 2 (figures, wall)", pass2_time))

            # Separate successes from failures
            successes: list[tuple[int, AnalysisResult, str, float, dict | None]] = []
            for r in chunk_raw_results:
                if isinstance(r, BaseException):
                    idx_hint = chunk_raw_results.index(r)
                    label_hint = chunk_labels[idx_hint] if idx_hint < len(chunk_labels) else "?"
                    print(f"  [pass2 chunk {idx_hint:>2}] FAILED ({label_hint}): {r}")
                else:
                    successes.append(r)

            # Order by chunk index
            for chunk_idx, result, op_id, elapsed, usage in sorted(successes, key=lambda x: x[0]):
                pass2_results.append(result)
                pass2_op_ids.append(op_id)
                all_usages.append(usage)
                timings.append((f"  pass2 chunk {chunk_idx} ({chunk_labels[chunk_idx]})", elapsed))
                if result.contents:
                    pass2_docs.append(cast(DocumentContent, result.contents[0]))

            seq_time = sum(r[3] for r in successes)
            failed_count = len(chunk_raw_results) - len(successes)
            summary = (f"\nPass 2 complete in {_fmt_elapsed(pass2_time)} wall-clock  "
                       f"(sequential sum: {_fmt_elapsed(seq_time)}, "
                       f"speedup: {seq_time / pass2_time:.1f}x)")
            if failed_count:
                summary += f"  [{failed_count} chunk(s) timed out]"
            print(summary + "\n")
        else:
            print("Pass 2 \u2014 skipped (no figure pages detected).\n")

        # ---- Await pass-1 save completion ----
        pass1_save_timings = await save_pass1_task
        timings.extend(pass1_save_timings)

        # ---- Save remaining outputs (depend on pass-2 results) ----
        print(f"Writing output to: {output_dir}\n")

        t = time.perf_counter()
        _save_json(pass1_result, pass2_results, output_dir)
        timings.append(("Save JSON", time.perf_counter() - t))

        if args.save_extras and pass2_docs:
            t = time.perf_counter()
            await _save_figures(
                client, pass2_docs, pass2_op_ids, output_dir,
                max_workers=args.max_figure_workers,
            )
            timings.append(("Save figures", time.perf_counter() - t))

    await credential.close()

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
        "--max-figure-workers", type=int, default=16,
        help="Max concurrent figure image downloads (default: 16).",
    )
    parser.add_argument(
        "--chunk-timeout", type=float, default=300,
        help="Per-chunk timeout in seconds (default: 300). "
             "Chunks that exceed this are skipped.",
    )
    parser.add_argument(
        "--save-extras", action="store_true", default=False,
        help="Also save tables.md and figures/*.",
    )
    args = parser.parse_args()

    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
