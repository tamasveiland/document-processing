"""
Parallel figure processing for Azure Document Intelligence results.

Downloads figure images and saves descriptions concurrently using a thread pool.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult


def _fmt_elapsed(secs: float) -> str:
    if secs >= 60:
        m, s = divmod(secs, 60)
        return f"{int(m)}m {s:.0f}s"
    return f"{secs:.2f}s"


def _safe_stem(figure_id: str) -> str:
    """Convert a figure ID to a safe filename stem."""
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


def _try_download_figure_image(
    client: DocumentIntelligenceClient,
    model_id: str,
    result_id: str,
    figure,
    figures_dir: Path,
) -> str | None:
    """Download the cropped figure image. Returns a status message or None."""
    try:
        response = client.get_analyze_result_figure(
            model_id=model_id,
            result_id=result_id,
            figure_id=figure.id,
        )
        image_bytes = b"".join(response)
        if not image_bytes:
            return None
        stem = _safe_stem(figure.id)
        out_path = figures_dir / f"figure_{stem}.png"
        with open(out_path, "wb") as fh:
            fh.write(image_bytes)
        return f"{out_path.name}  ({len(image_bytes):,} bytes)"
    except Exception as exc:  # noqa: BLE001
        return f"[info]  Image not available for figure {figure.id}: {exc}"


def _process_single_figure(
    client: DocumentIntelligenceClient,
    model_id: str,
    result_id: str | None,
    figure,
    figures_dir: Path,
) -> tuple[str, float, list[str]]:
    """Process one figure (description + image download). Returns (figure_id, elapsed, messages)."""
    t0 = time.perf_counter()
    messages: list[str] = []

    desc_name = _save_figure_description(figure, figures_dir)
    messages.append(f"    [saved] {desc_name}")

    if result_id:
        img_msg = _try_download_figure_image(client, model_id, result_id, figure, figures_dir)
        if img_msg:
            messages.append(f"    [saved] {img_msg}" if not img_msg.startswith("[") else f"    {img_msg}")

    elapsed = time.perf_counter() - t0
    messages.append(f"    [time]  {figure.id}: {_fmt_elapsed(elapsed)}")
    return figure.id, elapsed, messages


def save_figures_parallel(
    client: DocumentIntelligenceClient,
    result: AnalyzeResult,
    model_id: str,
    result_id: str | None,
    output_dir: Path,
    max_workers: int = 8,
) -> None:
    """Save all figures in parallel using a thread pool."""
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    if not result.figures:
        print("  No figures found; skipping figures/")
        return

    if not result_id:
        print("  [warn] Could not determine result_id; skipping figure image downloads.")

    print(f"  Processing {len(result.figures)} figure(s) with {max_workers} workers ...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_single_figure, client, model_id, result_id, figure, figures_dir
            ): figure.id
            for figure in result.figures
        }

        for future in as_completed(futures):
            figure_id, elapsed, messages = future.result()
            for msg in messages:
                print(msg)
