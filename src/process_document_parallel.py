#!/usr/bin/env python3
"""
Process a large PDF in parallel by splitting into page-range chunks.

Each chunk is submitted as a separate Azure Content Understanding analysis,
polled concurrently, and the results are merged into a single output set
identical to what process_document.py produces.

Usage
-----
    python process_document_parallel.py report.pdf
    python process_document_parallel.py report.pdf --chunk-size 20 --workers 4
    python process_document_parallel.py report.pdf --save-extras

Auth
----
    Uses DefaultAzureCredential.  Endpoint is read from
    AZURE_CONTENT_UNDERSTANDING_ENDPOINT.
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
from typing import cast

import pypdfium2 as pdfium  # PDF splitting (Apache-2.0)
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


def _build_chunk_pdf(src: pdfium.PdfDocument, start_0: int, end_0: int) -> bytes:
    """Extract pages [start_0, end_0) (0-based) from *src* into a new PDF."""
    chunk_doc = pdfium.PdfDocument.new()
    chunk_doc.import_pages(src, list(range(start_0, end_0)))
    buf = io.BytesIO()
    chunk_doc.save(buf)
    chunk_doc.close()
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Single-chunk analysis (runs inside a thread)
# ---------------------------------------------------------------------------


async def _analyze_chunk(
    client: ContentUnderstandingClient,
    chunk_bytes: bytes,
    chunk_index: int,
    start_page: int,
    end_page: int,
    semaphore: asyncio.Semaphore,
    timeout: float = 300,
) -> tuple[int, AnalysisResult, str, float, dict | None]:
    """Analyze one chunk and return (chunk_index, result, operation_id, elapsed, usage)."""
    async with semaphore:
        t0 = time.perf_counter()
        print(f"  [chunk {chunk_index}] Submitting pages {start_page}–{end_page} "
              f"({len(chunk_bytes):,} bytes) ...")

        poller = await client.begin_analyze_binary(
            analyzer_id="prebuilt-documentSearch",
            binary_input=chunk_bytes,
        )
        op_id: str = poller.operation_id
        print(f"  [chunk {chunk_index}] Operation {op_id}  (polling ...)")

        result: AnalysisResult = await asyncio.wait_for(
            poller.result(), timeout=timeout,
        )
        elapsed = time.perf_counter() - t0
        print(f"  [chunk {chunk_index}] Complete — pages {start_page}–{end_page} "
              f"in {_fmt_elapsed(elapsed)}")

        usage = _extract_usage(poller)

        return chunk_index, result, op_id, elapsed, usage


def _extract_usage(poller) -> dict | None:
    """Pull usage details from the poller's final polling response."""
    try:
        response_json = (
            poller.polling_method()._pipeline_response.http_response.json()
        )
        return response_json.get("usage")
    except Exception:  # noqa: BLE001
        return None


def _merge_usage(usages: list[dict | None]) -> dict | None:
    """Aggregate usage dicts from multiple chunks into a single summary."""
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
            merged_tokens = merged.setdefault("tokens", {})
            for tk, tv in u["tokens"].items():
                merged_tokens[tk] = merged_tokens.get(tk, 0) + tv
    return merged or None


def _print_usage(usage: dict | None) -> None:
    """Print a usage summary block."""
    if not usage:
        return

    _SEP = "\u2500" * 55
    print(f"\n{_SEP}")
    print("  Usage details (aggregated across chunks)")
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
# Merging
# ---------------------------------------------------------------------------


def _merge_markdown(docs: list[DocumentContent]) -> str:
    parts: list[str] = []
    for doc in docs:
        md = (doc.markdown or "").strip()
        if md:
            parts.append(md)
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Output helpers  (reused from process_document.py)
# ---------------------------------------------------------------------------


def _save_json(results: list[AnalysisResult], output_dir: Path) -> None:
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


def _save_tables(docs: list[DocumentContent], output_dir: Path) -> None:
    all_tables = [(doc_idx, t) for doc_idx, doc in enumerate(docs)
                  for t in (doc.tables or [])]
    if not all_tables:
        print("  No structured tables found; skipping tables.md")
        return

    path = output_dir / "tables.md"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Extracted Tables\n\n")
        for idx, (_doc_idx, table) in enumerate(all_tables, start=1):
            caption_text = table.caption.content if table.caption else f"Table {idx}"
            fh.write(f"## {caption_text}\n\n")
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
            response = await client.get_result_file(
                operation_id=operation_id,
                path=f"figures/{figure.id}",
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


async def _save_figures(
    client: ContentUnderstandingClient,
    docs: list[DocumentContent],
    operation_ids: list[str],
    output_dir: Path,
    max_workers: int = 16,
) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    all_figures: list[tuple[DocumentFigure, str]] = []
    for doc, op_id in zip(docs, operation_ids):
        for fig in doc.figures or []:
            all_figures.append((fig, op_id))

    if not all_figures:
        print("  No figures found; skipping figures/")
        return

    print(f"  Processing {len(all_figures)} figure(s) "
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

    # ---- compute chunk ranges ----
    t = time.perf_counter()
    print(f"Splitting {pdf_path.name} into {args.chunk_size}-page chunks ...")
    src = pdfium.PdfDocument(pdf_path)
    total_pages = len(src)
    chunk_ranges: list[tuple[int, int]] = []  # (start_0, end_0) 0-based exclusive
    for start in range(0, total_pages, args.chunk_size):
        end = min(start + args.chunk_size, total_pages)
        chunk_ranges.append((start, end))
    split_time = time.perf_counter() - t
    print(f"  {total_pages} pages → {len(chunk_ranges)} chunk(s) in {_fmt_elapsed(split_time)}\n")

    # ---- async analysis ----
    load_dotenv()
    endpoint = os.environ.get("AZURE_CONTENT_UNDERSTANDING_ENDPOINT", "")
    if not endpoint:
        print("Error: AZURE_CONTENT_UNDERSTANDING_ENDPOINT is not set.", file=sys.stderr)
        sys.exit(1)

    credential = DefaultAzureCredential()
    async with ContentUnderstandingClient(
        endpoint=endpoint, credential=credential,
    ) as client:

        workers = min(args.workers, len(chunk_ranges))
        print(f"Submitting {len(chunk_ranges)} chunk(s) with {workers} worker(s) ...\n")

        semaphore = asyncio.Semaphore(workers)
        t_analysis = time.perf_counter()

        # Pre-build all chunk PDFs (sync) before async submission,
        # so all pypdfium2 C-library calls finish before the event
        # loop processes HTTP tasks.
        chunk_bytes_list: list[bytes] = []
        for start_0, end_0 in chunk_ranges:
            chunk_bytes_list.append(_build_chunk_pdf(src, start_0, end_0))
        src.close()

        # Now submit all chunks asynchronously
        chunk_timeout = args.chunk_timeout
        tasks: list[asyncio.Task] = []
        for idx, (start_0, end_0) in enumerate(chunk_ranges):
            task = asyncio.create_task(
                _analyze_chunk(client, chunk_bytes_list[idx], idx,
                               start_0 + 1, end_0, semaphore,
                               timeout=chunk_timeout)
            )
            tasks.append(task)
            await asyncio.sleep(0)  # yield so earlier tasks can start I/O

        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        analysis_time = time.perf_counter() - t_analysis

        # Separate successes from failures
        successes: list[tuple[int, AnalysisResult, str, float, dict | None]] = []
        for i, r in enumerate(raw_results):
            if isinstance(r, BaseException):
                s, e = chunk_ranges[i][0] + 1, chunk_ranges[i][1]
                print(f"  [chunk {i}] FAILED (p{s}–{e}): {r}")
            else:
                successes.append(r)

        failed_count = len(raw_results) - len(successes)
        summary = f"\nAll chunks complete in {_fmt_elapsed(analysis_time)} wall-clock."
        if failed_count:
            summary += f"  [{failed_count} chunk(s) timed out]"
        print(summary + "\n")

        # ---- order results by chunk index ----
        ordered: list[tuple[AnalysisResult, str, float, dict | None] | None] = [None] * len(chunk_ranges)
        for chunk_idx, result, op_id, elapsed, usage in successes:
            ordered[chunk_idx] = (result, op_id, elapsed, usage)

        ordered_results: list[AnalysisResult] = []
        ordered_op_ids: list[str] = []
        chunk_usages: list[dict | None] = []
        for entry in ordered:
            if entry is not None:
                r, oid, _, u = entry
                ordered_results.append(r)
                ordered_op_ids.append(oid)
                chunk_usages.append(u)

        # ---- extract DocumentContent per chunk ----
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
            ("PDF split", split_time),
            ("Analysis (wall)", analysis_time),
        ]

        # Per-chunk times for the summary
        for idx in range(len(chunk_ranges)):
            entry = ordered[idx]
            start_p, end_p = chunk_ranges[idx][0] + 1, chunk_ranges[idx][1]
            if entry is not None:
                timings.append((f"  chunk {idx} (p{start_p}–{end_p})", entry[2]))
            else:
                timings.append((f"  chunk {idx} (p{start_p}–{end_p}) TIMEOUT", 0.0))

        t = time.perf_counter()
        _save_json(ordered_results, output_dir)
        timings.append(("Save JSON", time.perf_counter() - t))

        t = time.perf_counter()
        merged_md = _merge_markdown(docs)
        _save_markdown(merged_md, output_dir)
        timings.append(("Save markdown", time.perf_counter() - t))

        if args.save_extras:
            t = time.perf_counter()
            _save_tables(docs, output_dir)
            timings.append(("Save tables", time.perf_counter() - t))

            t = time.perf_counter()
            await _save_figures(client, docs, ordered_op_ids, output_dir,
                                max_workers=args.max_figure_workers)
            timings.append(("Save figures", time.perf_counter() - t))

    await credential.close()

    total = time.perf_counter() - t_start
    _SEP = "\u2500" * 55
    print(f"\n{_SEP}")
    print("  Timing summary")
    print(_SEP)
    for label, elapsed in timings:
        print(f"  {label:<25s}  {elapsed:9.2f}s  ({_fmt_elapsed(elapsed)})")
    print(_SEP)
    print(f"  {'Total':<25s}  {total:9.2f}s  ({_fmt_elapsed(total)})")
    print(_SEP)

    # Speedup vs. sequential (sum of individual chunk times)
    seq_time = sum(entry[2] for entry in ordered if entry is not None)
    if analysis_time > 0:
        print(f"\n  Sequential sum: {_fmt_elapsed(seq_time)}  |  "
              f"Wall-clock: {_fmt_elapsed(analysis_time)}  |  "
              f"Speedup: {seq_time / analysis_time:.1f}x")

    _print_usage(_merge_usage(chunk_usages))

    print("\nDone.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process a large PDF in parallel chunks with Azure "
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
        "--chunk-size", "-c", type=int, default=30,
        help="Number of pages per chunk (default: 30).",
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=4,
        help="Max concurrent submissions (default: 4).",
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
        help="Also save tables (tables.md) and figures (figures/*). "
             "Disabled by default; only result.json and document.md are saved.",
    )
    args = parser.parse_args()

    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
