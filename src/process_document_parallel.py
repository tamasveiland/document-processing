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
import io
import json
import os
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
# PDF splitting
# ---------------------------------------------------------------------------


def _split_pdf(pdf_path: Path, chunk_size: int) -> list[tuple[bytes, int, int]]:
    """Split *pdf_path* into chunks of *chunk_size* pages.

    Returns a list of (pdf_bytes, start_page_1based, end_page_1based).
    """
    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)
    chunks: list[tuple[bytes, int, int]] = []

    for start in range(0, total_pages, chunk_size):
        end = min(start + chunk_size, total_pages) - 1  # 0-based inclusive
        chunk_doc = pymupdf.open()  # new empty PDF
        chunk_doc.insert_pdf(doc, from_page=start, to_page=end)
        buf = io.BytesIO()
        chunk_doc.save(buf)
        chunk_doc.close()
        chunks.append((buf.getvalue(), start + 1, end + 1))  # 1-based for display

    doc.close()
    return chunks


# ---------------------------------------------------------------------------
# Single-chunk analysis (runs inside a thread)
# ---------------------------------------------------------------------------


def _analyze_chunk(
    endpoint: str,
    credential: DefaultAzureCredential,
    chunk_bytes: bytes,
    chunk_index: int,
    start_page: int,
    end_page: int,
) -> tuple[int, AnalysisResult, str, float, dict | None]:
    """Analyze one chunk and return (chunk_index, result, operation_id, elapsed, usage)."""
    # Each thread gets its own client to avoid any shared-state issues.
    client = ContentUnderstandingClient(endpoint=endpoint, credential=credential)

    t0 = time.perf_counter()
    print(f"  [chunk {chunk_index}] Submitting pages {start_page}–{end_page} "
          f"({len(chunk_bytes):,} bytes) ...")

    poller = client.begin_analyze_binary(
        analyzer_id="prebuilt-documentSearch",
        binary_input=chunk_bytes,
    )
    op_id: str = poller.operation_id
    print(f"  [chunk {chunk_index}] Operation {op_id}  (polling ...)")

    result: AnalysisResult = poller.result()
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


def _try_download_figure_image(
    client: ContentUnderstandingClient,
    operation_id: str,
    figure: DocumentFigure,
    figures_dir: Path,
) -> None:
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


def _save_figures(
    client: ContentUnderstandingClient,
    docs: list[DocumentContent],
    operation_ids: list[str],
    output_dir: Path,
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
        description="Process a large PDF in parallel chunks with Azure "
                    "Content Understanding (prebuilt-documentSearch).",
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
        "--chunk-size", "-c",
        type=int,
        default=30,
        help="Number of pages per chunk (default: 30).",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="Max concurrent submissions (default: 4).",
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

    # ---- split ----
    t = time.perf_counter()
    print(f"Splitting {pdf_path.name} into {args.chunk_size}-page chunks ...")
    chunks = _split_pdf(pdf_path, args.chunk_size)
    split_time = time.perf_counter() - t
    total_pages = chunks[-1][2]  # end page of last chunk (1-based)
    print(f"  {total_pages} pages → {len(chunks)} chunk(s) in {_fmt_elapsed(split_time)}\n")

    # ---- parallel analysis ----
    load_dotenv()
    endpoint = os.environ.get("AZURE_CONTENT_UNDERSTANDING_ENDPOINT", "")
    if not endpoint:
        print("Error: AZURE_CONTENT_UNDERSTANDING_ENDPOINT is not set.", file=sys.stderr)
        sys.exit(1)
    credential = DefaultAzureCredential()

    workers = min(args.workers, len(chunks))
    print(f"Submitting {len(chunks)} chunk(s) with {workers} worker(s) ...\n")

    # Collect results keyed by chunk_index for ordered merging.
    chunk_results: dict[int, tuple[AnalysisResult, str, float, dict | None]] = {}
    t_analysis = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _analyze_chunk,
                endpoint, credential,
                chunk_bytes, idx, start_page, end_page,
            ): idx
            for idx, (chunk_bytes, start_page, end_page) in enumerate(chunks)
        }
        for future in as_completed(futures):
            idx, result, op_id, elapsed, usage = future.result()
            chunk_results[idx] = (result, op_id, elapsed, usage)

    analysis_time = time.perf_counter() - t_analysis
    print(f"\nAll chunks complete in {_fmt_elapsed(analysis_time)} wall-clock.\n")

    # ---- order results by chunk index ----
    ordered_results: list[AnalysisResult] = []
    ordered_op_ids: list[str] = []
    chunk_usages: list[dict | None] = []
    for idx in range(len(chunks)):
        res, op_id, _, usage = chunk_results[idx]
        ordered_results.append(res)
        ordered_op_ids.append(op_id)
        chunk_usages.append(usage)

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
    for idx in range(len(chunks)):
        _, _, elapsed, _ = chunk_results[idx]
        start_p, end_p = chunks[idx][1], chunks[idx][2]
        timings.append((f"  chunk {idx} (p{start_p}–{end_p})", elapsed))

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
        client = ContentUnderstandingClient(endpoint=endpoint, credential=credential)
        _save_figures(client, docs, ordered_op_ids, output_dir)
        timings.append(("Save figures", time.perf_counter() - t))

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
    seq_time = sum(chunk_results[i][2] for i in range(len(chunks)))
    if analysis_time > 0:
        print(f"\n  Sequential sum: {_fmt_elapsed(seq_time)}  |  "
              f"Wall-clock: {_fmt_elapsed(analysis_time)}  |  "
              f"Speedup: {seq_time / analysis_time:.1f}x")

    _print_usage(_merge_usage(chunk_usages))

    print("\nDone.")


if __name__ == "__main__":
    main()
