# Doc Parser Benchmark

This project benchmarks multiple document parsers by running `main.py` over one or more input files and writing comparable outputs to a result directory.

## Prerequisites

- `uv` installed: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)
- Python 3.11, 3.12, or 3.13 available to `uv`

## Setup with uv

From the project directory:

```bash
uv sync
```

This resolves dependencies from `pyproject.toml`, creates `.venv`, and writes `uv.lock`.

## Run

Show CLI options:

```bash
uv run python main.py --help
```

Run a benchmark on one file:

```bash
uv run python main.py /path/to/document.pdf
```

Run against multiple files or a folder:

```bash
uv run python main.py /path/to/doc1.pdf /path/to/slides.pptx ./sample_docs --output-dir benchmark_output
```

Run all files in `/data` and generate parser rankings:

```bash
uv run python main.py /data --output-dir benchmark_output
```

Limit parsers (comma-separated adapter names from `main.py`):

```bash
uv run python main.py ./sample_docs --parsers docling,pymupdf4llm,native_docx
```

## Heuristic score

Results are ranked by a **heuristic_score** so you can compare parsers per document. The score is computed as follows:

- **+1.0** if the parse succeeded, **0.0** if it failed or was skipped.
- **+3.0 × text_similarity_to_ground_truth** when a `.groundtruth.txt` file exists for that input (0–1 similarity).
- **+2.0 × heading_recall** when a `.headings.json` ground truth exists (0–1 recall).
- **+ up to 1.0** from heading count: `min(heading_count, 20) / 20`.
- **+ up to 0.5** from table-like count: `min(table_like_count, 10) / 20`.
- **− up to 1.0** for runtime: `min(elapsed_seconds, 600) / 600` (slower runs get a lower score).

The final value is rounded to 4 decimal places. Higher scores mean better success, better match to ground truth (when present), more structure (headings/tables), and faster runs. Summary tables and `summary.json` are ordered by this score (descending).

In addition to per-document rankings, the pipeline now writes:

- `parser_leaderboard_overall.json`: best parser ranking across all attempted runs.
- `parser_leaderboard_by_filetype.json`: best parser ranking per detected file type (`pdf`, `docx`, `pptx`, etc.).
- `summary.md`: includes both overall and per-filetype leaderboard sections.

## Environment Variables

Copy and edit the template before using cloud adapters:

```bash
cp .env.example .env
```

Variables used by cloud adapters:

- `LLAMA_CLOUD_API_KEY` or `LLAMA_PARSE_API_KEY` for `llamaparse`
- `DOCUMENTINTELLIGENCE_ENDPOINT`, `DOCUMENTINTELLIGENCE_API_KEY` (plus optional version/model vars) for `azure_document_intelligence`
- `TEXTRACT_S3_BUCKET`, `TEXTRACT_S3_PREFIX` for PDF runs with `amazon_textract`

Notes:

- `main.py` reads environment variables directly from your shell session.
- If you use a `.env` file, load it in your shell before running (for example: `set -a; source .env; set +a` in bash).

## External Service/Binary Notes

- `marker` adapter prefers the `marker_single` CLI when available. If it is missing, the Python API fallback only covers PDFs.
- `amazon_textract` requires valid AWS credentials in your environment/profile and, for PDFs, an S3 bucket.
- `azure_document_intelligence` requires a valid endpoint and API key.
- `llamaparse` requires a valid API key.

## Troubleshooting

- If an adapter is unavailable, check `benchmark_output/availability.json` for the missing dependency or env var reason.
- If parsers are slow on large files, raise or lower `--timeout-seconds` as needed.
- Use `--parsers` to run only the adapters you are currently validating.
