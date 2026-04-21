# Document Processing — `src/`

Python scripts that submit a PDF to **Azure Content Understanding** and save structured output including full markdown, tables, and figure descriptions. Multiple processing strategies are provided for different document sizes and workloads.

## Prerequisites

| Requirement | Details |
|---|---|
| Python | 3.9 or later |
| Azure subscription | With a deployed Content Understanding resource |
| Azure CLI | `az login` — used for passwordless auth |
| Model defaults | GPT-4.1-mini and text-embedding-3-large mapped in your Content Understanding resource (one-time setup, see below) |

## Setup

### 1 — Clone and navigate

```bash
cd src
```

### 2 — Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### 4 — Configure environment

Copy the template and fill in your values:

```bash
copy .env.sample .env   # Windows
cp .env.sample .env     # macOS / Linux
```

The only **required** variable for this script is:

```
AZURE_CONTENT_UNDERSTANDING_ENDPOINT=https://<your-resource>.cognitiveservices.azure.com/
```

All other variables are populated automatically when the infrastructure is provisioned with `azd up` from the repository root.

### 5 — Authenticate

```bash
az login
```

> No API keys are stored. Authentication uses `DefaultAzureCredential`, which picks up your `az login` session locally and a managed identity in Azure.

### 6 — One-time model defaults setup

`prebuilt-documentSearch` requires GPT-4.1-mini and text-embedding-3-large to be set as defaults on your Content Understanding resource. You can do this from the [Content Understanding settings page](https://contentunderstanding.ai.azure.com/settings) or by running the included setup script:

First, list the deployments in your AI Foundry resource to find the exact names:

```bash
pip install azure-mgmt-cognitiveservices   # one-time, if not already installed
python setup_model_defaults.py --list-deployments
```

Then configure the defaults using the names shown:

```bash
python setup_model_defaults.py --completion-model <deployment-name> --embedding-model <deployment-name>
```

Example:

```bash
python setup_model_defaults.py --completion-model gpt-5.2 --embedding-model text-embedding-3-large
```

This only needs to be done **once per Content Understanding resource**. The `--completion-model` value must exactly match the deployment name in Foundry (not the model name). The `--embedding-model` argument defaults to `text-embedding-3-large` if omitted.

## Usage

### Single-file processing

```
python process_document.py <pdf-path> [--output-dir <dir>] [--save-extras]
```

### Parallel processing (large documents)

For large PDFs, `process_document_parallel.py` splits the document into page-range chunks and submits them concurrently, significantly reducing wall-clock time:

```
python process_document_parallel.py <pdf-path> [--output-dir <dir>] [--chunk-size N] [--workers N] [--save-extras]
```

### Content-aware splitting (balanced workloads)

`process_document_smart.py` pre-scans each page with PyMuPDF to measure visual complexity (image count + area). Pages are grouped into weight-balanced chunks so that image-heavy pages don't become stragglers:

```
python process_document_smart.py <pdf-path> [--output-dir <dir>] [--max-weight N] [--workers N] [--save-extras] [--show-scores]
```

### Two-pass processing (minimize GPU token spend)

`process_document_twopass.py` runs a fast layout-only pass (`prebuilt-read`) on the full document, detects which pages contain figures, then submits only those pages to `prebuilt-documentSearch` for AI-enriched analysis. This avoids burning GPT tokens on text-only pages:

```
python process_document_twopass.py <pdf-path> [--output-dir <dir>] [--workers N] [--figure-group-size N] [--save-extras]
```

### Three-pass processing (separate text, tables, and figures)

`process_document_threepass.py` uses a different analyzer for each content type:

1. **Pass 1** — `prebuilt-read` on the full document (fast text extraction)
2. **Pass 2** — `prebuilt-layout` on table-bearing pages only (detailed cell-level tables)
3. **Pass 3** — `prebuilt-documentSearch` on figure-bearing pages only (AI descriptions, Chart.js, Mermaid)

Pages containing both tables and figures are routed to pass 3 (the richer analyzer). This gives the best quality for each content type while keeping costs low:

```
python process_document_threepass.py <pdf-path> [--output-dir <dir>] [--workers N] [--table-group-size N] [--figure-group-size N] [--save-extras]
```

### Arguments

#### `process_document.py`

| Argument | Required | Default | Description |
|---|---|---|---|
| `pdf_path` | Yes | — | Path to the input PDF file |
| `--output-dir`, `-o` | No | `./output` | Directory where output files are written |
| `--save-extras` | No | disabled | Also save `tables.md` and `figures/*` in addition to `result.json` and `document.md` |

#### `process_document_parallel.py` (additional arguments)

| Argument | Required | Default | Description |
|---|---|---|---|
| `--chunk-size`, `-c` | No | `30` | Number of pages per chunk |
| `--workers`, `-w` | No | `4` | Maximum concurrent API submissions |

#### `process_document_smart.py` (additional arguments)

| Argument | Required | Default | Description |
|---|---|---|---|
| `--max-weight`, `-m` | No | `6.0` | Maximum total weight per chunk (higher = fewer, larger chunks) |
| `--workers`, `-w` | No | `8` | Maximum concurrent API submissions |
| `--show-scores` | No | disabled | Print per-page complexity scores before processing |

#### `process_document_twopass.py` (additional arguments)

| Argument | Required | Default | Description |
|---|---|---|---|
| `--workers`, `-w` | No | `8` | Maximum concurrent pass-2 API submissions |
| `--figure-group-size` | No | `4` | Max consecutive figure-pages per pass-2 chunk |

#### `process_document_threepass.py` (additional arguments)

| Argument | Required | Default | Description |
|---|---|---|---|
| `--workers`, `-w` | No | `8` | Maximum concurrent submissions per pass |
| `--table-group-size` | No | `8` | Max consecutive table-pages per pass-2 chunk |
| `--figure-group-size` | No | `4` | Max consecutive figure-pages per pass-3 chunk |

### Examples

```bash
# Basic — output written to ./output/
python process_document.py report.pdf

# Custom output directory
python process_document.py report.pdf --output-dir results/report

# Also save tables and figures
python process_document.py report.pdf --save-extras

# Custom output directory and extras
python process_document.py report.pdf --output-dir results/report --save-extras

# Parallel — split a 151-page PDF into 30-page chunks, 4 workers
python process_document_parallel.py report.pdf

# Parallel — custom chunk size and workers
python process_document_parallel.py report.pdf --chunk-size 20 --workers 6

# Parallel with extras
python process_document_parallel.py report.pdf --save-extras --output-dir results/

# Smart — content-aware splitting with balanced chunks
python process_document_smart.py report.pdf

# Smart — custom weight limit and show per-page scores
python process_document_smart.py report.pdf --max-weight 8.0 --show-scores

# Two-pass — fast layout + targeted figure analysis
python process_document_twopass.py report.pdf

# Two-pass — custom worker count and save extras
python process_document_twopass.py report.pdf --workers 12 --save-extras

# Three-pass — text, tables, figures each with the best analyzer
python process_document_threepass.py report.pdf

# Three-pass — with extras and custom group sizes
python process_document_threepass.py report.pdf --save-extras --table-group-size 10 --figure-group-size 3
```

## Output

By default only two files are written:

```
output/
├── result.json          # Full raw API response
└── document.md          # Complete document markdown (text + table + figure refs)
```

With `--save-extras` the full set is produced:

```
output/
├── result.json          # Full raw API response
├── document.md          # Complete document markdown (text + table + figure refs)
├── tables.md            # All tables rendered as markdown grids
└── figures/
    ├── figure_<id>.md   # Per-figure: AI description, caption, Chart.js / Mermaid data
    └── figure_<id>.png  # Cropped figure image (when available from the service)
```

### `result.json`
Complete JSON response from the Content Understanding API, including usage metrics, confidence scores, and bounding box coordinates for every element.

### `document.md`
Full document content rendered as markdown. Tables are represented as markdown tables; figures appear as placeholders referencing their IDs.

### `tables.md`
All detected tables extracted from the document and rendered as individual markdown tables, each with caption and row/column count.

### `figures/figure_<id>.md`
Per-figure file containing:
- **Caption** (if present in the document)
- **AI-generated description** — a textual explanation of the figure's content
- **Chart.js data** (for detected charts, e.g. bar, line, pie)
- **Mermaid syntax** (for detected diagrams, e.g. flowcharts, network diagrams)
- **Source coordinates** in the original PDF

### `figures/figure_<id>.png`
When the service returns image bytes for a figure, the cropped image is saved here. If the service does not expose image bytes for a particular figure type, only the `.md` file is produced.

## Analyzer

Uses [`prebuilt-documentSearch`](https://learn.microsoft.com/azure/ai-services/content-understanding/concepts/prebuilt-analyzers) — the recommended analyzer for RAG and document ingestion scenarios. It provides:

- Layout-preserving markdown extraction
- Table detection with cell-level content
- Figure detection with AI descriptions
- Chart analysis in Chart.js format
- Diagram recognition in Mermaid syntax

## Environment variables reference

See [`.env.sample`](.env.sample) for the full list. The table below covers variables actively used by this script.

| Variable | Required | Description |
|---|---|---|
| `AZURE_CONTENT_UNDERSTANDING_ENDPOINT` | **Yes** | Endpoint URL of the Content Understanding resource |
| `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT` | No | Document Intelligence endpoint (for other pipelines) |
| `AZURE_OPENAI_ENDPOINT` | No | Azure OpenAI endpoint |
| `APPLICATIONINSIGHTS_CONNECTION_STRING` | No | Application Insights telemetry |
