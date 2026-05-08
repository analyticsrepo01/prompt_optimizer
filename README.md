# Vertex AI Prompt Optimizer

Author: [SaurabhM](https://github.com/SaurabhM)

Notebooks and utilities for the [Vertex AI Prompt Optimizer](https://cloud.google.com/vertex-ai/generative-ai/docs/prompt-optimizer/overview) — a tool that automatically refines and improves LLM prompts using two approaches:

- **Zero-Shot Optimizer**: Refines a prompt based on best-practice guidelines, no dataset required.
- **Data-Driven Optimizer (VAPO)**: Uses your evaluation dataset to iteratively find the highest-scoring system instruction.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| GCP project with Vertex AI API enabled | [Enable here](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com) |
| GCS bucket | Used for VAPO config and results |
| Service account with roles: `aiplatform.user`, `storage.objectAdmin`, `artifactregistry.reader` | Compute Engine default SA is used |

### Install dependencies

```bash
pip install "google-cloud-aiplatform>=1.108.0" "pydantic" "etils" \
            "protobuf==4.25.3" "importlib_resources" "google-auth>=2.35.0" \
            "gradio"
```

> **Note:** `protobuf==4.25.3` is pinned for compatibility with `google-cloud-aiplatform>=1.108.0`. `google-auth>=2.35.0` is required to avoid `AttributeError: module 'google.auth.environment_vars' has no attribute 'GCE_METADATA_TIMEOUT'`.

---

## Notebooks

### 1. `vertex_ai_prompt_optimizer.ipynb`

**Full walkthrough** — covers both optimization approaches end-to-end with an optional Gradio results viewer.

- **Part 1 – Zero-Shot Optimizer**: Calls `client.prompts.optimize(prompt=...)` to get an improved prompt with applied guideline explanations. Response fields are accessed via `response.parsed_response.suggested_prompt`, `.applicable_guidelines`, `.original_prompt`, `.optimization_type`.
- **Part 2 – Data-Driven Optimizer (VAPO)**: Builds an `OptimizationConfig` pydantic model, serializes it to GCS, then launches a VAPO custom job via `client.prompts.launch_optimization_job(method=vt.PromptOptimizerMethod.VAPO, config=vt.OptimizeJobConfig(...))`.
- **Optional**: Launches a Gradio app (`VAPOResultsViewer`) to visually browse all optimization runs, templates, and per-example eval metrics.

**Dataset**: `gs://github-repo/prompts/prompt_optimizer/rag_qa_dataset.jsonl` (RAG QA)

**Target model**: `gemini-2.5-flash` (VAPO); zero-shot optimizer uses `global` location client.

---

### 2. `vertex_ai_prompt_optimizer_long_prompt.ipynb`

**Long system instructions** — optimizes prompts that include substantial context, static examples, or documentation using the Data-Driven Optimizer only.

- Uses `optimization_mode: "instruction"` on a long-form date-range extraction prompt.
- Demonstrates the `placeholder_to_content` config field to inject static few-shot `{{examples}}` into the system instruction before optimization, while `{{question}}` and `{{target}}` remain dynamic per-row placeholders.
- Uploads the config JSON to GCS with `etils.epath` and launches the job via `client.prompts.launch_optimization_job(method=vt.PromptOptimizerMethod.VAPO, config=vt.OptimizeJobConfig(...))`.

**Dataset**: `gs://github-repo/prompts/prompt_optimizer/qa_long_prompt_dataset.jsonl` (100 rows)

**Target model**: `gemini-2.5-flash`

---

### 3. `vertex_ai_prompt_optimizer_ui.ipynb`

**Step-by-step UI** — a form-style notebook that walks through VAPO configuration and job launch without writing JSON manually.

- **Step 1**: Define a system instruction with an `{{examples}}` static placeholder and a prompt template with `{{question}}` / `{{target}}` dynamic placeholders.
- **Step 2**: Set project, bucket, location, and service account.
- **Step 3**: Choose optimization mode, target model (`gemini-3.1-pro-preview`), eval metrics, and provide `PLACEHOLDER_TO_VALUE` examples for static injection.
- **Step 4**: Optional advanced settings (steps, demo set size, QPS).
- **Step 5**: Launches the VAPO job using `client.prompts.launch_optimization_job(method=vt.PromptOptimizerMethod.VAPO, config=vt.OptimizeJobConfig(...))`.
- **Step 6**: Reads results from GCS and displays the best-scoring optimized prompt.

**Dataset**: `gs://github-repo/prompts/prompt_optimizer/qa_long_prompt_dataset.jsonl`

**Target model**: `gemini-3.1-pro-preview`

---

### Other notebooks

| Notebook | Description |
|---|---|
| `get_started_with_vertex_ai_prompt_optimizer_custom_metric.ipynb` | Deploy a Cloud Run function as a custom eval scorer and plug it into VAPO via `custom_metric_name` / `custom_metric_cloud_function_name`. |
| `get_started_with_vertex_ai_prompt_optimizer_multimodality.ipynb` | Optimize prompts for image+text tasks; sets `has_multimodal_inputs: true`. |
| `get_started_with_vertex_ai_prompt_optimizer_tool_usage.ipynb` | Optimize system instructions for function-calling scenarios; uses `tool_call_valid`, `tool_name_match`, `tool_parameter_key_match` metrics. |

---

### `vapo_lib.py`

Shared utility library used by the UI notebook. Includes GCS helpers, dataset validation, config generation, and job submission wrappers.

---

## Project Configuration

All notebooks are pre-configured with:

```
PROJECT_ID  = my-project-0004-346516
BUCKET_URI  = gs://my-project-0004-346516-prompt-optimizer
LOCATION    = us-central1
```

The VAPO job runs as the Compute Engine default service account (`255766800726-compute@developer.gserviceaccount.com`).

---

## Key API Patterns

### Zero-Shot Optimizer

```python
import vertexai

client = vertexai.Client(project=PROJECT_ID, location="global")
response = client.prompts.optimize(prompt=prompt)

suggested  = response.parsed_response.suggested_prompt
guidelines = response.parsed_response.applicable_guidelines   # list of guideline objects
original   = response.parsed_response.original_prompt
opt_type   = response.parsed_response.optimization_type
```

> **Note:** `client.prompt_optimizer.optimize_prompt` is deprecated — use `client.prompts.optimize`.

### Data-Driven Optimizer (VAPO)

```python
import vertexai._genai.types as vt
from etils import epath
import json

# Write config to GCS
with epath.Path(config_path).open("w") as f:
    json.dump(vapo_config_dict, f)

# Launch job
run_config = vt.OptimizeJobConfig(
    config_path=config_path,
    wait_for_completion=True,
    service_account=SERVICE_ACCOUNT,
)
result = client.prompts.launch_optimization_job(
    method=vt.PromptOptimizerMethod.VAPO,
    config=run_config,
)
```

### GCS data loading

Use `google-cloud-storage` directly — `pd.read_json("gs://...")` via `fsspec`/`gcsfs` is broken in this environment:

```python
import io
from google.cloud import storage as gcs

bucket_name, blob_name = gcs_path[5:].split("/", 1)
data = gcs.Client().bucket(bucket_name).blob(blob_name).download_as_bytes()
df = pd.read_json(io.BytesIO(data), lines=True)
```
