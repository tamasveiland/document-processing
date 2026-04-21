#!/usr/bin/env python3
"""
Content-aware parallel PDF processing with Azure Content Understanding.

Instead of fixed-size page chunks, this script pre-scans each page with
PyMuPDF to estimate visual complexity (image count, image area ratio) and
groups pages into balanced chunks so no single chunk becomes a straggler.

Usage
-----
    python process_document_smart.py report.pdf
    python process_document_smart.py report.pdf --max-weight 6.0 --workers 8
    python process_document_smart.py report.pdf --save-extras

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
from dataclasses import dataclass
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
# Page complexity scoring
# ---------------------------------------------------------------------------

# Weight constants — tuned so that a text-only page ≈ 1.0 and a page with
# a large chart ≈ 3–5.  These can be overridden via CLI --image-weight.
_BASE_WEIGHT = 1.0      # every page gets at least this
_PER_IMAGE_WEIGHT = 1.5  # per image on the page
_AREA_WEIGHT = 3.0       # multiplied by image-area / page-area fraction


@dataclass
class PageInfo:
    """Pre-scan metadata for a single page."""
    page_index: int       # 0-based
    image_count: int
    image_area_ratio: float  # 0.0 – 1.0
    weight: float


def _scan_pages(pdf_path: Path) -> list[PageInfo]:
    """Score every page in *pdf_path* by visual complexity."""
    doc = pymupdf.open(pdf_path)
    pages: list[PageInfo] = []

    for idx in range(len(doc)):
        page = doc[idx]
        page_area = page.rect.width * page.rect.height or 1.0

        images = page.get_images(full=True)
        image_count = len(images)

        # Sum bounding-box areas of all images on the page.
        total_image_area = 0.0
        for img_info in images:
            xref = img_info[0]
            try:
                rects = page.get_image_rects(xref)
                for r in rects:
                    total_image_area += r.width * r.height
            except Exception:  # noqa: BLE001
                # Fallback: assume a medium-sized image
                total_image_area += page_area * 0.25

        area_ratio = min(total_image_area / page_area, 1.0)
        weight = _BASE_WEIGHT + image_count * _PER_IMAGE_WEIGHT + area_ratio * _AREA_WEIGHT

        pages.append(PageInfo(
            page_index=idx,
            image_count=image_count,
            image_area_ratio=area_ratio,
            weight=weight,
        ))

    doc.close()
    return pages


# ---------------------------------------------------------------------------
# Balanced chunking
# ---------------------------------------------------------------------------


def _build_balanced_chunks(
    pages: list[PageInfo],
    max_weight: float,
) -> list[list[int]]:
    """Group consecutive pages into chunks with total weight ≤ *max_weight*.

    Pages are kept in order (no reordering) so the merged markdown matches
    the original document.  A single page whose weight exceeds max_weight
    gets its own chunk.
    """
    chunks: list[list[int]] = []
    current_chunk: list[int] = []
    current_weight = 0.0

    for page in pages:
        # If adding this page would exceed the budget, close the current chunk.
        if current_chunk and current_weight + page.weight > max_weight:
            chunks.append(current_chunk)
            current_chunk = []
            current_weight = 0.0

        current_chunk.append(page.page_index)
        current_weight += page.weight

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _create_chunk_pdfs(
    pdf_path: Path,
    chunk_page_indices: list[list[int]],
) -> list[tuple[bytes, int, int]]:
    """Build per-chunk PDFs and return (bytes, first_page_1based, last_page_1based)."""
    doc = pymupdf.open(pdf_path)
    result: list[tuple[bytes, int, int]] = []

    for page_indices in chunk_page_indices:
        chunk_doc = pymupdf.open()
        for idx in page_indices:
            chunk_doc.insert_pdf(doc, from_page=idx, to_page=idx)
        buf = io.BytesIO()
        chunk_doc.save(buf)
        chunk_doc.close()
        result.append((buf.getvalue(), page_indices[0] + 1, page_indices[-1] + 1))

    doc.close()
    return result


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _analyze_chunk(
    endpoint: str,
    credential: DefaultAzureCredential,
    chunk_bytes: bytes,
    chunk_index: int,
    start_page: int,
    end_page: int,
) -> tuple[int, AnalysisResult, str, float, dict | None]:
    client = ContentUnderstandingClient(endpoint=endpoint, credential=credential)

    t0 = time.perf_counter()
    n_pages = end_page - start_page + 1
    print(f"  [chunk {chunk_index:>2}] Submitting pages {start_page}–{end_page} "
          f"({n_pages} pg, {len(chunk_bytes):,} bytes) ...")

    poller = client.begin_analyze_binary(
        analyzer_id="prebuilt-documentSearch",
        binary_input=chunk_bytes,
    )
    op_id: str = poller.operation_id
    print(f"  [chunk {chunk_index:>2}] Operation {op_id}  (polling ...)")

    result: AnalysisResult = poller.result()
    elapsed = time.perf_counter() - t0
    print(f"  [chunk {chunk_index:>2}] Complete — pages {start_page}–{end_page} "
          f"in {_fmt_elapsed(elapsed)}")

    usage = _extract_usage(poller)
    return chunk_index, result, op_id, elapsed, usage


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


def _print_usage(usage: dict | None) -> None:
    if not usage:
        return

    _SEP = "\u2500" * 60
    print(f"\n{_SEP}")
    print("  Usage details (aggregated)")
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


def _merge_markdown(docs: list[DocumentContent]) -> str:
    parts = [(doc.markdown or "").strip() for doc in docs]
    return "\n\n---\n\n".join(p for p in parts if p)


def _save_json(results: list[AnalysisResult], output_dir: Path) -> None:
    path = output_dir / "result.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([r.as_dict() for r in results], fh, indent=2, ensure_ascii=False, default=str)
    print(f"  [saved] {path}")


def _save_markdown(merged_md: str, output_dir: Path) -> None:
    path = output_dir / "document.md"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(merged_md)
    print(f"  [saved] {path}")


def _save_tables(docs: list[DocumentContent], output_dir: Path) -> None:
    all_tables = [(i, t) for i, doc in enumerate(docs) for t in (doc.tables or [])]
    if not all_tables:
        print("  No structured tables found; skipping tables.md")
        return

    path = output_dir / "tables.md"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Extracted Tables\n\n")
        for idx, (_di, table) in enumerate(all_tables, start=1):
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
    docs: list[DocumentContent],
    operation_ids: list[str],
    output_dir: Path,
) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    all_figures = [(fig, oid) for doc, oid in zip(docs, operation_ids) for fig in (doc.figures or [])]
    if not all_figures:
        print("  No figures found; skipping figures/")
        return
    print(f"  Processing {len(all_figures)} figure(s) ...")
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
        description="Content-aware parallel PDF processing with Azure "
                    "Content Understanding (prebuilt-documentSearch).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("pdf_path", help="Path to the input PDF file.")
    parser.add_argument(
        "--output-dir", "-o", default="output",
        help="Directory for output files (default: ./output).",
    )
    parser.add_argument(
        "--max-weight", "-m", type=float, default=6.0,
        help="Max complexity weight per chunk (default: 6.0). "
             "Lower = more chunks (more parallelism), higher = fewer chunks.",
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=8,
        help="Max concurrent submissions (default: 8).",
    )
    parser.add_argument(
        "--save-extras", action="store_true", default=False,
        help="Also save tables.md and figures/*.",
    )
    parser.add_argument(
        "--show-scores", action="store_true", default=False,
        help="Print per-page complexity scores before splitting.",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path).resolve()
    if not pdf_path.is_file():
        print(f"Error: file not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()

    # ---- scan pages ----
    t = time.perf_counter()
    print(f"Scanning {pdf_path.name} for page complexity ...")
    pages = _scan_pages(pdf_path)
    scan_time = time.perf_counter() - t

    total_weight = sum(p.weight for p in pages)
    image_pages = sum(1 for p in pages if p.image_count > 0)
    total_images = sum(p.image_count for p in pages)
    print(f"  {len(pages)} pages scanned in {_fmt_elapsed(scan_time)}")
    print(f"  {image_pages} pages with images ({total_images} total images)")
    print(f"  Total weight: {total_weight:.1f}  (avg {total_weight / len(pages):.2f}/page)\n")

    if args.show_scores:
        _SEP_THIN = "\u2500" * 60
        print(_SEP_THIN)
        print(f"  {'Page':>6s}  {'Images':>6s}  {'Img%':>6s}  {'Weight':>7s}")
        print(_SEP_THIN)
        for p in pages:
            flag = " ***" if p.weight > args.max_weight else ""
            print(f"  {p.page_index + 1:>6d}  {p.image_count:>6d}  "
                  f"{p.image_area_ratio:>5.0%}  {p.weight:>7.2f}{flag}")
        print(_SEP_THIN)
        print()

    # ---- balanced splitting ----
    t = time.perf_counter()
    chunk_page_groups = _build_balanced_chunks(pages, args.max_weight)
    chunks = _create_chunk_pdfs(pdf_path, chunk_page_groups)
    split_time = time.perf_counter() - t

    # Print chunk composition
    print(f"Split into {len(chunks)} chunk(s) (max-weight={args.max_weight})  "
          f"in {_fmt_elapsed(split_time)}:")
    for i, (grp, (_, sp, ep)) in enumerate(zip(chunk_page_groups, chunks)):
        grp_weight = sum(pages[idx].weight for idx in grp)
        grp_images = sum(pages[idx].image_count for idx in grp)
        print(f"  chunk {i:>2}: pages {sp:>3}–{ep:<3}  "
              f"({len(grp):>2} pg, wt {grp_weight:>5.1f}, {grp_images} img)")
    print()

    # ---- parallel analysis ----
    load_dotenv()
    endpoint = os.environ.get("AZURE_CONTENT_UNDERSTANDING_ENDPOINT", "")
    if not endpoint:
        print("Error: AZURE_CONTENT_UNDERSTANDING_ENDPOINT is not set.", file=sys.stderr)
        sys.exit(1)
    credential = DefaultAzureCredential()

    workers = min(args.workers, len(chunks))
    print(f"Submitting {len(chunks)} chunk(s) with {workers} worker(s) ...\n")

    chunk_results: dict[int, tuple[AnalysisResult, str, float, dict | None]] = {}
    t_analysis = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _analyze_chunk, endpoint, credential,
                chunk_bytes, idx, start_page, end_page,
            ): idx
            for idx, (chunk_bytes, start_page, end_page) in enumerate(chunks)
        }
        for future in as_completed(futures):
            idx, result, op_id, elapsed, usage = future.result()
            chunk_results[idx] = (result, op_id, elapsed, usage)

    analysis_time = time.perf_counter() - t_analysis
    print(f"\nAll chunks complete in {_fmt_elapsed(analysis_time)} wall-clock.\n")

    # ---- order & merge ----
    ordered_results: list[AnalysisResult] = []
    ordered_op_ids: list[str] = []
    chunk_usages: list[dict | None] = []
    for idx in range(len(chunks)):
        res, op_id, _, usage = chunk_results[idx]
        ordered_results.append(res)
        ordered_op_ids.append(op_id)
        chunk_usages.append(usage)

    docs: list[DocumentContent] = []
    for res in ordered_results:
        if res.contents:
            docs.append(cast(DocumentContent, res.contents[0]))

    if not docs:
        print("Error: no content returned from any chunk.", file=sys.stderr)
        sys.exit(1)

    first, last = docs[0], docs[-1]
    print(
        f"Document: pages {first.start_page_number}–{last.end_page_number}  "
        f"({len(docs)} chunk(s) merged)\n"
        f"Writing output to: {output_dir}\n"
    )

    # ---- save outputs ----
    timings: list[tuple[str, float]] = [
        ("Page scan", scan_time),
        ("PDF split", split_time),
        ("Analysis (wall)", analysis_time),
    ]

    for idx in range(len(chunks)):
        _, _, elapsed, _ = chunk_results[idx]
        sp, ep = chunks[idx][1], chunks[idx][2]
        timings.append((f"  chunk {idx} (p{sp}–{ep})", elapsed))

    t = time.perf_counter()
    _save_json(ordered_results, output_dir)
    timings.append(("Save JSON", time.perf_counter() - t))

    t = time.perf_counter()
    _save_markdown(_merge_markdown(docs), output_dir)
    timings.append(("Save markdown", time.perf_counter() - t))

    if args.save_extras:
        t = time.perf_counter()
        _save_tables(docs, output_dir)
        timings.append(("Save tables", time.perf_counter() - t))

        t = time.perf_counter()
        client = ContentUnderstandingClient(endpoint=endpoint, credential=credential)
        _save_figures(client, docs, ordered_op_ids, output_dir)
        timings.append(("Save figures", time.perf_counter() - t))

    total = time.perf_counter() - t_start
    _SEP = "\u2500" * 60
    print(f"\n{_SEP}")
    print("  Timing summary")
    print(_SEP)
    for label, elapsed in timings:
        print(f"  {label:<28s}  {elapsed:9.2f}s  ({_fmt_elapsed(elapsed)})")
    print(_SEP)
    print(f"  {'Total':<28s}  {total:9.2f}s  ({_fmt_elapsed(total)})")
    print(_SEP)

    seq_time = sum(chunk_results[i][2] for i in range(len(chunks)))
    if analysis_time > 0:
        print(f"\n  Sequential sum: {_fmt_elapsed(seq_time)}  |  "
              f"Wall-clock: {_fmt_elapsed(analysis_time)}  |  "
              f"Speedup: {seq_time / analysis_time:.1f}x")

    _print_usage(_merge_usage(chunk_usages))

    print("\nDone.")


if __name__ == "__main__":
    main()
