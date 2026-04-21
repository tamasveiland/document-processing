#!/usr/bin/env python3
"""
Content-aware parallel PDF processing with Azure Document Intelligence (async).

Instead of fixed-size page chunks, this script pre-scans each page with
PyMuPDF to estimate visual complexity (image count, image area ratio) and
groups pages into balanced chunks so no single chunk becomes a straggler.
All API calls, polling, and figure downloads use the async DI client.

Usage
-----
    python src/process_document_di_smart.py path/to/document.pdf
    python src/process_document_di_smart.py report.pdf --max-weight 6.0 --workers 8
    python src/process_document_di_smart.py report.pdf --save-extras --show-scores

Outputs (under <output-dir>)
----------------------------
    result.json          Full raw API responses (one per chunk)
    document.md          Merged document markdown content
    tables.md            All tables rendered as markdown (when --save-extras)
    figures/             Figure descriptions + images  (when --save-extras)

Auth
----
    Uses DefaultAzureCredential (az login / managed identity — no keys in code).
    Endpoint is read from AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pymupdf  # PyMuPDF — PDF splitting & page analysis
from azure.ai.documentintelligence.aio import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import (
    AnalyzeOutputOption,
    AnalyzeResult,
    DocumentContentFormat,
    DocumentAnalysisFeature,
)
from azure.identity.aio import DefaultAzureCredential
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
# Page complexity scoring
# ---------------------------------------------------------------------------

_BASE_WEIGHT = 1.0        # every page gets at least this
_PER_IMAGE_WEIGHT = 1.5   # per image on the page
_AREA_WEIGHT = 3.0        # multiplied by image-area / page-area fraction


@dataclass
class PageInfo:
    """Pre-scan metadata for a single page."""
    page_index: int        # 0-based
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

        total_image_area = 0.0
        for img_info in images:
            xref = img_info[0]
            try:
                rects = page.get_image_rects(xref)
                for r in rects:
                    total_image_area += r.width * r.height
            except Exception:  # noqa: BLE001
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
# Result-ID extraction
# ---------------------------------------------------------------------------


def _extract_result_id(poller) -> str | None:
    """Extract the result ID from the poller's operation-location URL."""
    try:
        op_url = poller.polling_method()._initial_response.http_response.headers.get(
            "Operation-Location", ""
        )
        marker = "analyzeResults/"
        idx = op_url.find(marker)
        if idx == -1:
            return None
        rest = op_url[idx + len(marker):]
        return rest.split("?")[0].split("/")[0]
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Async chunk analysis
# ---------------------------------------------------------------------------


async def _analyze_chunk(
    endpoint: str,
    credential: DefaultAzureCredential,
    chunk_bytes: bytes,
    chunk_index: int,
    start_page: int,
    end_page: int,
    model_id: str,
    semaphore: asyncio.Semaphore,
) -> tuple[int, AnalyzeResult, str | None, float]:
    """Analyze one chunk. Returns (chunk_index, result, result_id, elapsed)."""
    async with semaphore:
        async with DocumentIntelligenceClient(
            endpoint=endpoint, credential=credential,
        ) as client:
            t0 = time.perf_counter()
            n_pages = end_page - start_page + 1
            print(f"  [chunk {chunk_index:>2}] Submitting pages {start_page}–{end_page} "
                  f"({n_pages} pg, {len(chunk_bytes):,} bytes) ...")

            poller = await client.begin_analyze_document(
                model_id,
                chunk_bytes,
                output_content_format=DocumentContentFormat.MARKDOWN,
                features=[DocumentAnalysisFeature.FORMULAS],
                output=[AnalyzeOutputOption.FIGURES],
            )
            print(f"  [chunk {chunk_index:>2}] Polling ...")

            result: AnalyzeResult = await poller.result()
            elapsed = time.perf_counter() - t0
            print(f"  [chunk {chunk_index:>2}] Complete — pages {start_page}–{end_page} "
                  f"in {_fmt_elapsed(elapsed)}")

            result_id = _extract_result_id(poller)
            return chunk_index, result, result_id, elapsed


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------


def _merge_markdown(results: list[AnalyzeResult]) -> str:
    """Merge markdown from ordered chunk results."""
    parts = [(r.content or "").strip() for r in results]
    return "\n\n---\n\n".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _save_json(results: list[AnalyzeResult], output_dir: Path) -> None:
    path = output_dir / "result.json"
    payload = [r.as_dict() for r in results]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
    print(f"  [saved] {path}")


def _save_markdown(merged_md: str, output_dir: Path) -> None:
    path = output_dir / "document.md"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(merged_md)
    print(f"  [saved] {path}")


def _save_tables(results: list[AnalyzeResult], output_dir: Path) -> None:
    all_tables = [(res_idx, t)
                  for res_idx, res in enumerate(results)
                  for t in (res.tables or [])]
    if not all_tables:
        print("  No structured tables found; skipping tables.md")
        return

    path = output_dir / "tables.md"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Extracted Tables\n\n")
        for idx, (_res_idx, table) in enumerate(all_tables, start=1):
            fh.write(f"## Table {idx}\n\n")
            fh.write(f"*{table.row_count} rows × {table.column_count} columns*\n\n")

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

    print(f"  [saved] {path}  ({len(all_tables)} table(s))")


# ---------------------------------------------------------------------------
# Async figure helpers
# ---------------------------------------------------------------------------


def _safe_stem(figure_id: str) -> str:
    return figure_id.replace("/", "_").replace(":", "_").replace(".", "_")


def _save_figure_description(figure, figures_dir: Path) -> str:
    """Save a markdown description file for a figure. Returns the filename."""
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
    return path.name


async def _download_figure_image(
    endpoint: str,
    credential: DefaultAzureCredential,
    model_id: str,
    result_id: str,
    figure,
    figures_dir: Path,
    semaphore: asyncio.Semaphore,
) -> str | None:
    """Download one figure image asynchronously. Returns a status message."""
    async with semaphore:
        try:
            async with DocumentIntelligenceClient(
                endpoint=endpoint, credential=credential,
            ) as client:
                response = await client.get_analyze_result_figure(
                    model_id=model_id,
                    result_id=result_id,
                    figure_id=figure.id,
                )
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


async def _save_figures_async(
    endpoint: str,
    credential: DefaultAzureCredential,
    results: list[AnalyzeResult],
    model_id: str,
    result_ids: list[str | None],
    output_dir: Path,
    max_workers: int = 8,
) -> None:
    """Save all figures across all chunks concurrently."""
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    all_figures: list[tuple[object, str | None]] = []
    for res, rid in zip(results, result_ids):
        for fig in res.figures or []:
            all_figures.append((fig, rid))

    if not all_figures:
        print("  No figures found; skipping figures/")
        return

    print(f"  Processing {len(all_figures)} figure(s) with up to {max_workers} concurrent downloads ...")
    semaphore = asyncio.Semaphore(max_workers)

    # Save descriptions synchronously (fast, local I/O)
    for figure, _rid in all_figures:
        name = _save_figure_description(figure, figures_dir)
        print(f"    [saved] {name}")

    # Download images concurrently
    tasks = []
    for figure, rid in all_figures:
        if rid:
            tasks.append(_download_figure_image(
                endpoint, credential, model_id, rid, figure, figures_dir, semaphore,
            ))

    if tasks:
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

    load_dotenv()
    endpoint = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "")
    if not endpoint:
        print("Error: AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT is not set.", file=sys.stderr)
        sys.exit(1)

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
        _SEP_THIN = "─" * 60
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

    print(f"Split into {len(chunks)} chunk(s) (max-weight={args.max_weight})  "
          f"in {_fmt_elapsed(split_time)}:")
    for i, (grp, (_, sp, ep)) in enumerate(zip(chunk_page_groups, chunks)):
        grp_weight = sum(pages[idx].weight for idx in grp)
        grp_images = sum(pages[idx].image_count for idx in grp)
        print(f"  chunk {i:>2}: pages {sp:>3}–{ep:<3}  "
              f"({len(grp):>2} pg, wt {grp_weight:>5.1f}, {grp_images} img)")
    print()

    # ---- parallel async analysis ----
    credential = DefaultAzureCredential()
    workers = min(args.workers, len(chunks))
    semaphore = asyncio.Semaphore(workers)
    print(f"Submitting {len(chunks)} chunk(s) with up to {workers} concurrent requests ...\n")

    t_analysis = time.perf_counter()
    tasks = [
        _analyze_chunk(
            endpoint, credential, chunk_bytes, idx,
            start_page, end_page, args.model_id, semaphore,
        )
        for idx, (chunk_bytes, start_page, end_page) in enumerate(chunks)
    ]
    chunk_raw_results = await asyncio.gather(*tasks)
    analysis_time = time.perf_counter() - t_analysis
    print(f"\nAll chunks complete in {_fmt_elapsed(analysis_time)} wall-clock.\n")

    # ---- order by chunk index ----
    chunk_raw_results_sorted = sorted(chunk_raw_results, key=lambda x: x[0])
    ordered_results: list[AnalyzeResult] = []
    ordered_result_ids: list[str | None] = []
    chunk_times: list[tuple[int, float]] = []
    for chunk_idx, result, result_id, elapsed in chunk_raw_results_sorted:
        ordered_results.append(result)
        ordered_result_ids.append(result_id)
        chunk_times.append((chunk_idx, elapsed))

    if not ordered_results:
        print("Error: no results returned from any chunk.", file=sys.stderr)
        sys.exit(1)

    # ---- summary ----
    total_page_count = sum(len(r.pages) for r in ordered_results if r.pages)
    total_table_count = sum(len(r.tables) for r in ordered_results if r.tables)
    total_figure_count = sum(len(r.figures) for r in ordered_results if r.figures)
    print(
        f"Document: {total_page_count} page(s), {total_table_count} table(s), "
        f"{total_figure_count} figure(s)\n"
        f"Writing output to: {output_dir}\n"
    )

    # ---- save outputs ----
    timings: list[tuple[str, float]] = [
        ("Page scan", scan_time),
        ("PDF split", split_time),
        ("Analysis (wall)", analysis_time),
    ]
    for chunk_idx, elapsed in chunk_times:
        sp, ep = chunks[chunk_idx][1], chunks[chunk_idx][2]
        timings.append((f"  chunk {chunk_idx} (p{sp}–{ep})", elapsed))

    t = time.perf_counter()
    _save_json(ordered_results, output_dir)
    timings.append(("Save JSON", time.perf_counter() - t))

    t = time.perf_counter()
    merged_md = _merge_markdown(ordered_results)
    _save_markdown(merged_md, output_dir)
    timings.append(("Save markdown", time.perf_counter() - t))

    if args.save_extras:
        t = time.perf_counter()
        _save_tables(ordered_results, output_dir)
        timings.append(("Save tables", time.perf_counter() - t))

        t = time.perf_counter()
        await _save_figures_async(
            endpoint, credential, ordered_results, args.model_id,
            ordered_result_ids, output_dir, max_workers=args.max_figure_workers,
        )
        timings.append(("Save figures", time.perf_counter() - t))

    await credential.close()

    total = time.perf_counter() - t_start
    _SEP = "─" * 60
    print(f"\n{_SEP}")
    print("  Timing summary")
    print(_SEP)
    for label, elapsed in timings:
        print(f"  {label:<28s}  {elapsed:9.2f}s  ({_fmt_elapsed(elapsed)})")
    print(_SEP)
    print(f"  {'Total':<28s}  {total:9.2f}s  ({_fmt_elapsed(total)})")
    print(_SEP)

    # Speedup estimate
    seq_time = sum(el for _, el in chunk_times)
    if analysis_time > 0:
        print(f"\n  Sequential sum: {_fmt_elapsed(seq_time)}  |  "
              f"Wall-clock: {_fmt_elapsed(analysis_time)}  |  "
              f"Speedup: {seq_time / analysis_time:.1f}x")

    print("\nDone.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Content-aware parallel PDF processing with Azure "
                    "Document Intelligence (async I/O).",
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
        help="Document Intelligence model ID (default: prebuilt-layout).",
    )
    parser.add_argument(
        "--max-weight",
        type=float,
        default=6.0,
        help="Max complexity weight per chunk (default: 6.0). "
             "Lower = more chunks (more parallelism), higher = fewer chunks.",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=8,
        help="Max concurrent chunk submissions (default: 8).",
    )
    parser.add_argument(
        "--max-figure-workers",
        type=int,
        default=8,
        help="Max concurrent figure image downloads (default: 8).",
    )
    parser.add_argument(
        "--save-extras",
        action="store_true",
        default=False,
        help="Also save tables (tables.md) and figures (figures/*).",
    )
    parser.add_argument(
        "--show-scores",
        action="store_true",
        default=False,
        help="Print per-page complexity scores before splitting.",
    )
    args = parser.parse_args()

    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
