# Script Specifications

## Structure

```
work/jobs/{job_name}/
│
├── README.md
├── history.yaml
│
├── checkpoints/
│   ├── checkpoint_v1.pt
│   └── checkpoint_v2.pt
│
├── finetuning/
│   │
│   ├── data/
│   │   ├── v1/
│   │   │   ├── finetunes/
│   │   │   │   └── {job_name}_v1_finetune.yaml
│   │   │   └── eval/
│   │   │       └── {job_name}_v1_eval.yaml
│   │   ├── v2/
│   │   │   ├── finetunes/
│   │   │   │   └── {job_name}_v2_finetune.yaml
│   │   │   └── eval/
│   │   │       └── {job_name}_v2_eval.yaml
│   │   └── standard_eval_set/
│   │       └── standard_eval.yaml
│   │
│   └── results/
│       ├── v1/
│       │   ├── tuning_summary.yaml
│       │   ├── distribution.png
│       │   └── predictions.yaml
│       └── v2/
│           ├── tuning_summary.yaml
│           ├── distribution.png
│           └── predictions.yaml
│
└── inference_runs/
    │
    └── {run_id}/
        ├── data/
        │   └── {job_name}_checkpoint_v{m}_run_{run_id}_inference.yaml
        └── results/
            ├── inference_report.yaml
            ├── distribution.png
            └── predictions.yaml
```

# File Format Specifications

This section defines the structure of all YAML data files used and generated within the project. The file extension `.yaml` is used for all YAML files to maintain consistency.

### 1. Input Data

- **Principle:** One YAML file represents a complete dataset (a collection of data points).
- **Format:** The file contains a list of text-value pairs.

#### Fine-tuning & Evaluation Data (`finetuning/data/...`)
- **Fine-tuning Data:** The file in `finetuning/data/v{n}/finetunes/` is named `{job_name}_v{n}_finetune.yaml`.
- **Evaluation Data:** The file in `finetuning/data/v{n}/eval/` is named `{job_name}_v{n}_eval.yaml`. This is copied from a master file (e.g., `standard_eval.yaml`) in the `standard_eval_set` directory.

#### Inference Data (`inference_runs/...`)
- **Inference Data:** The file in `inference_runs/{run_id}/data/` is named `{job_name}_checkpoint_v{m}_run_{run_id}_inference.yaml`.

```yaml
# Example Input Data Format
- text: "Apple's quarterly earnings exceeded expectations by 5%."
  value: 5.0
- text: "The new iPhone model saw a 10% increase in pre-orders."
  value: 10.0
```

### 2. Job History (`history.yaml`)

- **Filename:** `history.yaml`
- **Format:** A list of events, recording each tuning and inference run.

```yaml
- event_type: "tuning"
  timestamp: "2025-07-13T10:00:00Z"
  version: 1
  input_data_dir: "path/to/finetuning/data/for_v1"
  base_model: "default"
  results:
    mse: 0.021
    mae: 0.11
    r2_score: 0.89
  checkpoint_path: "work/jobs/{job_name}/checkpoints/checkpoint_v1.pt"

- event_type: "inference"
  timestamp: "2025-07-15T09:00:00Z"
  run_id: "q3_earnings_forecast"
  using_version: 2
  input_data_dir: "path/to/inference_data"
  results_path: "work/jobs/{job_name}/inference_runs/q3_earnings_forecast/results"
```

### 3. Tuning Summary (`tuning_summary.yaml`)

- **Filename:** `tuning_summary.yaml`
- **Format:** A structured report of the tuning process results.

```yaml
overview:
  job_name: {job_name}
  version_created: {version}
  timestamp: "{timestamp}" # ISO 8601 format
  base_model_used: "{path_to_base_model_or_previous_version}"

data_sources:
  finetuning_data: "{path_to_finetuning_data_dir}"
  evaluation_data: "{path_to_evaluation_data_dir}"

performance_metrics:
  mse: {mse_value}
  mae: {mae_value}
  r2_score: {r2_score_value}

prediction_error_analysis:
  mean: {error_mean}
  std_dev: {error_std}
  min: {error_min}
  max: {error_max}
  quartiles:
    q1: {q1}
    median: {median}
    q3: {q3}

process_timing:
  total_tuning_seconds: {duration_in_seconds}

output_files:
  checkpoint: "{path_to_checkpoint_pt}"
  predictions_yaml: "{path_to_predictions_yaml}"
  error_histogram: "{path_to_distribution_png}"
```

### 4. Inference Report (`inference_report.yaml`)

- **Filename:** `inference_report.yaml`
- **Format:** A structured report of the inference run.

```yaml
overview:
  job_name: {job_name}
  run_id: "{run_id}"
  timestamp: "{timestamp}" # ISO 8601 format
  model_version_used: {version}

data_source:
  inference_data: "{path_to_inference_data_dir}"

prediction_statistics:
  num_samples: {num_samples}
  mean: {prediction_mean}
  std_dev: {prediction_std}
  min: {prediction_min}
  max: {prediction_max}

process_timing:
  total_inference_seconds: {duration_in_seconds}

output_files:
  predictions_yaml: "{path_to_predictions_yaml}"
  prediction_histogram: "{path_to_distribution_png}"
```

### 5. Prediction Results (`predictions.yaml`)

- **Filename:** `predictions.yaml`
- **Format:** A list of individual prediction results, including distribution statistics.

```yaml
predictions:
  - text: "The company's revenue grew by 5%."
    # Actual value (only exists during evaluation)
    actual_value: 5.0
    # Statistical summary of the prediction distribution
    prediction_summary:
      # The mean of the prediction distribution (the representative predicted value)
      mean: 4.8
      # The standard deviation of the prediction (indicates uncertainty)
      std_dev: 0.5
      min: 3.9
      max: 5.2
      num_samples: 100
    # The error between the representative prediction and the actual value
    error: -0.2 # (prediction_summary.mean - actual_value)

  # In an inference run, 'actual_value' and 'error' keys will be omitted.
```

---

## `tune.py`

**Purpose:** Manages the fine-tuning process, creating new model versions and evaluating their performance.

**Arguments:**
- `--job-name` (string, required): The name of the job to operate on.
- `--data-file` (path, required): Path to the single YAML file containing the data for this fine-tuning session.
- `--eval-set-file` (path, required for new jobs): Path to the master evaluation YAML file.
- `--base-model` (path, optional): Path to a pre-trained base model to start a new job from.

**Behavior:**
1.  Checks if the job directory `work/jobs/{job_name}` exists. If not, it creates the entire directory structure (initializes the job).
2.  Determines the new version number (e.g., `v1` for a new job, or `v(n+1)` for an existing one).
3.  **Data Handling:**
    - Copies the content of `--data-file` to `finetuning/data/v<n>/finetunes/{job_name}_v{n}_finetune.yaml`.
    - On the first run, copies `--eval-set-file` to `finetuning/data/standard_eval_set/standard_eval.yaml`.
    - Copies the master `standard_eval_set` to `finetuning/data/v<n>/eval/{job_name}_v{n}_eval.yaml`.
    - Copies the content of `--data-dir` to `finetuning/data/v<n>/finetunes/`.
    - On the first run, copies `--eval-set-dir` to `finetuning/data/standard_eval_set/`.
    - Copies the master `standard_eval_set` to `finetuning/data/v<n>/eval/`.
4.  **Tuning Process:**
    - Loads the latest checkpoint (`checkpoint_v<n-1>.pt`). If it's the first run, it loads the `--base-model` or the default model.
    - Fine-tunes the model using the data in `finetuning/data/v<n>/finetunes/`.
    - Saves the newly trained model as `checkpoints/checkpoint_v<n>.pt`.
5.  **Evaluation & Reporting:**
    - Evaluates the new model's performance using the data in `finetuning/data/v<n>/eval/`.
    - Generates `tuning_summary.txt`, `distribution.png`, and `predictions.csv` in `finetuning/results/v<n>/`.
    - Appends a summary of the run to `history.yaml`.

## `infer.py`

**Purpose:** Runs inference using a specified model checkpoint.

**Arguments:**
- `--job-name` (string, required): The name of the job.
- `--checkpoint-version` (integer, required): The version number of the checkpoint to use for inference.
- `--data-file` (path, required): Path to the single YAML file containing the data for inference.
- `--run-id` (string, required): A unique identifier for this inference run (e.g., a timestamp like `20250713_003000`). as the directory name.

**Behavior:**
1.  Creates the run directory: `inference_runs/{run_id}/`.
2.  **Data Handling:**
    - Copies the content of `--data-file` to `inference_runs/{run_id}/data/`.
3.  **Inference Process:**
    - Loads the specified model from `checkpoints/checkpoint_v<version>.pt`.
    - Runs inference on the data located in `inference_runs/{run_id}/data/`.
4.  **Reporting:**
    - Generates `inference_report.txt`, `distribution.png` (if applicable), and `predictions.csv` in `inference_runs/{run_id}/results/`.
    - Appends a summary of the run to `history.yaml`.

---

# User Interface

The ultimate goal for this project's user interface is to expose the workflow functionality through a **Model Context Protocol (MCP) Server**. This will allow AI assistants like Cascade to interact with the system programmatically via natural language commands.

The development will proceed in two phases:
1.  **Phase 1: Core Logic Scripts:** First, implement the core functionalities as standalone Python scripts (`tune.py`, `infer.py`). This ensures the underlying logic is robust and testable.
2.  **Phase 2: MCP Server Implementation:** Once the core scripts are complete, they will be refactored and wrapped into an MCP server. This server will expose the core functions as a set of tools (API endpoints).

## MCP Server Tool Specification

The MCP server, tentatively named `regresslm-mcp`, will provide the following tools:

### `list_jobs()`
- **Description:** Lists all available job names.
- **Arguments:** None
- **Returns:** A list of strings (job names).

### `get_job_details(job_name: str)`
- **Description:** Retrieves details for a specific job, including its description and available checkpoint versions.
- **Arguments:**
    - `job_name` (string): The name of the job to inspect.
- **Returns:** A dictionary containing job details.

### `init_job(job_name: str, eval_set_dir: str, base_model: str = None)`
- **Description:** Initializes a new job directory structure.
- **Arguments:**
    - `job_name` (string): The name for the new job.
    - `eval_set_dir` (string): Path to the standard evaluation dataset.
    - `base_model` (string, optional): Path to a pre-trained base model.
- **Returns:** A success or failure message.

### `tune_job(job_name: str, data_dir: str)`
- **Description:** Runs a new tuning session for an existing job, creating the next version.
- **Arguments:**
    - `job_name` (string): The name of the job to tune.
    - `data_dir` (string): Path to the new data for fine-tuning.
- **Returns:** A dictionary summarizing the tuning results (e.g., new version number, performance metrics).

### `infer(job_name: str, version: int, data_dir: str, run_id: str)`
- **Description:** Runs inference using a specified model version.
- **Arguments:**
    - `job_name` (string): The name of the job.
    - `version` (int): The model version to use.
    - `data_dir` (string): Path to the data for inference.
    - `run_id` (string): A unique identifier for this inference run.
- **Returns:** A dictionary containing the path to the results.