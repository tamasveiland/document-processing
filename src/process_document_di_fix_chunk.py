#!/usr/bin/env python3
"""
Process a large PDF in parallel fixed-size chunks with Azure Document Intelligence.

The PDF is split into N-page chunks (default 30), each submitted as a separate
``begin_analyze_document`` call.  All API calls, polling, and figure downloads
use the async Document Intelligence client so I/O overlaps efficiently.

Usage
-----
    python src/process_document_di_fix_chunk.py path/to/document.pdf
    python src/process_document_di_fix_chunk.py report.pdf --chunk-size 20 --workers 4
    python src/process_document_di_fix_chunk.py report.pdf --save-extras --max-figure-workers 16

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
from pathlib import Path

import pypdfium2 as pdfium  # PDF splitting (Apache-2.0)
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
# PDF splitting
# ---------------------------------------------------------------------------


def _get_total_pages(pdf_path: Path) -> int:
    """Return the page count of *pdf_path* without splitting."""
    src = pdfium.PdfDocument(pdf_path)
    n = len(src)
    src.close()
    return n


def _split_one_chunk(src: pdfium.PdfDocument, start: int, end: int) -> bytes:
    """Extract pages [start, end) from *src* and return them as PDF bytes."""
    chunk_doc = pdfium.PdfDocument.new()
    chunk_doc.import_pages(src, list(range(start, end)))
    buf = io.BytesIO()
    chunk_doc.save(buf)
    chunk_doc.close()
    return buf.getvalue()


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
    client: DocumentIntelligenceClient,
    chunk_bytes: bytes,
    chunk_index: int,
    start_page: int,
    end_page: int,
    model_id: str,
    semaphore: asyncio.Semaphore,
) -> tuple[int, AnalyzeResult, str | None, float]:
    """Analyze one chunk. Returns (chunk_index, result, result_id, elapsed)."""
    async with semaphore:
        t0 = time.perf_counter()
        print(f"  [chunk {chunk_index}] Submitting pages {start_page}–{end_page} "
              f"({len(chunk_bytes):,} bytes) ...")

        poller = await client.begin_analyze_document(
            model_id,
            chunk_bytes,
            output_content_format=DocumentContentFormat.MARKDOWN,
            features=[DocumentAnalysisFeature.FORMULAS],
            output=[AnalyzeOutputOption.FIGURES],
        )
        print(f"  [chunk {chunk_index}] Polling ...")

        result: AnalyzeResult = await poller.result()
        elapsed = time.perf_counter() - t0
        print(f"  [chunk {chunk_index}] Complete — pages {start_page}–{end_page} "
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
    client: DocumentIntelligenceClient,
    model_id: str,
    result_id: str,
    figure,
    figures_dir: Path,
    semaphore: asyncio.Semaphore,
) -> str | None:
    """Download one figure image asynchronously. Returns a status message."""
    async with semaphore:
        try:
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
    client: DocumentIntelligenceClient,
    results: list[AnalyzeResult],
    model_id: str,
    result_ids: list[str | None],
    output_dir: Path,
    max_workers: int = 8,
) -> None:
    """Save all figures across all chunks concurrently."""
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Collect all figures with their corresponding result_id
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
                client, model_id, rid, figure, figures_dir, semaphore,
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

    # ---- pipelined split + analysis ----
    total_pages = _get_total_pages(pdf_path)
    num_chunks = -(-total_pages // args.chunk_size)  # ceil division
    print(f"Splitting {pdf_path.name} into {args.chunk_size}-page chunks ...")
    print(f"  {total_pages} pages → {num_chunks} chunk(s)\n")

    credential = DefaultAzureCredential()
    async with DocumentIntelligenceClient(
        endpoint=endpoint, credential=credential,
    ) as client:
        workers = min(args.workers, num_chunks)
        semaphore = asyncio.Semaphore(workers)
        print(f"Submitting {num_chunks} chunk(s) with up to {workers} concurrent "
              f"requests (pipelined split+submit) ...\n")

        t_pipeline = time.perf_counter()
        split_time_accum = 0.0
        tasks: list[asyncio.Task] = []
        chunk_pages: list[tuple[int, int]] = []  # (start_page, end_page) per idx

        src = pdfium.PdfDocument(pdf_path)
        for idx, page_start in enumerate(range(0, total_pages, args.chunk_size)):
            page_end = min(page_start + args.chunk_size, total_pages)
            t_s = time.perf_counter()
            chunk_bytes = _split_one_chunk(src, page_start, page_end)
            split_time_accum += time.perf_counter() - t_s

            start_page, end_page = page_start + 1, page_end
            chunk_pages.append((start_page, end_page))

            task = asyncio.create_task(_analyze_chunk(
                client, chunk_bytes, idx,
                start_page, end_page, args.model_id, semaphore,
            ))
            tasks.append(task)
            await asyncio.sleep(0)  # yield so earlier tasks can start I/O
        src.close()

        chunk_raw_results = await asyncio.gather(*tasks)
        pipeline_time = time.perf_counter() - t_pipeline
        print(f"\nAll chunks complete in {_fmt_elapsed(pipeline_time)} wall-clock "
              f"(split: {_fmt_elapsed(split_time_accum)} cumulative).\n")

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

        timings: list[tuple[str, float]] = [
            ("PDF split (cum.)", split_time_accum),
            ("Pipeline (wall)", pipeline_time),
        ]
        for chunk_idx, elapsed in chunk_times:
            sp, ep = chunk_pages[chunk_idx]
            timings.append((f"  chunk {chunk_idx} (p{sp}–{ep})", elapsed))

        # ---- save outputs ----
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
                client, ordered_results, args.model_id,
                ordered_result_ids, output_dir, max_workers=args.max_figure_workers,
            )
            timings.append(("Save figures", time.perf_counter() - t))

    await credential.close()

    total = time.perf_counter() - t_start
    _SEP = "─" * 55
    print(f"\n{_SEP}")
    print("  Timing summary")
    print(_SEP)
    for label, elapsed in timings:
        print(f"  {label:<25s}  {elapsed:9.2f}s  ({_fmt_elapsed(elapsed)})")
    print(_SEP)
    print(f"  {'Total':<25s}  {total:9.2f}s  ({_fmt_elapsed(total)})")
    print(_SEP)

    # Speedup estimate
    seq_time = sum(el for _, el in chunk_times)
    if pipeline_time > 0:
        print(f"\n  Sequential sum: {_fmt_elapsed(seq_time)}  |  "
              f"Wall-clock: {_fmt_elapsed(pipeline_time)}  |  "
              f"Speedup: {seq_time / pipeline_time:.1f}x")

    print("\nDone.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process a large PDF in parallel fixed-size chunks with "
                    "Azure Document Intelligence (async I/O).",
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
        "--chunk-size", "-c",
        type=int,
        default=30,
        help="Number of pages per chunk (default: 30).",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=6,
        help="Max concurrent chunk submissions (default: 6).",
    )
    parser.add_argument(
        "--max-figure-workers",
        type=int,
        default=32,
        help="Max concurrent figure image downloads (default: 32).",
    )
    parser.add_argument(
        "--save-extras",
        action="store_true",
        default=False,
        help="Also save tables (tables.md) and figures (figures/*).",
    )
    args = parser.parse_args()

    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
