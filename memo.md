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
│   │   │   └── eval/
│   │   ├── v2/
│   │   │   ├── finetunes/
│   │   │   └── eval/
│   │   └── standard_eval_set/
│   │
│   └── results/
│       ├── v1/
│       │   ├── tuning_summary.txt
│       │   ├── distribution.png
│       │   └── predictions.csv
│       └── v2/
│           ├── tuning_summary.txt
│           ├── distribution.png
│           └── predictions.csv
│
└── inference_runs/
    │
    └── {run_id}/
        ├── data/
        └── results/
            ├── inference_report.txt
            ├── distribution.png
            └── predictions.csv
```

## `tune.py`

**Purpose:** Manages the fine-tuning process, creating new model versions and evaluating their performance.

**Arguments:**
- `--job-name` (string, required): The name of the job to operate on (e.g., `apple_share_price_predictor`).
- `--data-dir` (path, required): Path to the directory containing the data for this fine-tuning session.
- `--eval-set-dir` (path, optional): Path to the standard evaluation dataset. This is **required for the very first run** of a new job to establish the master evaluation set. It is ignored on subsequent runs.
- `--base-model` (path, optional): Path to a pre-trained base model (`.pt` file) to start a new job from. If not provided, the job starts from the default base model.

**Behavior:**
1.  Checks if the job directory `work/jobs/{job_name}` exists. If not, it creates the entire directory structure (initializes the job).
2.  Determines the new version number (e.g., `v1` for a new job, or `v(n+1)` for an existing one).
3.  **Data Handling:**
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

**Purpose:** Executes a prediction task using a specific, previously trained model checkpoint.

**Arguments:**
- `--job-name` (string, required): The name of the job whose model you want to use.
- `--version` (int, required): The version number of the checkpoint to use for inference (e.g., `2` for `checkpoint_v2.pt`).
- `--data-dir` (path, required): Path to the directory containing the data you want to run predictions on.
- `--run-id` (string, required): A unique name for this inference run (e.g., `2025_q3_earnings_forecast`). This will be used as the directory name.

**Behavior:**
1.  Creates the run directory: `inference_runs/{run_id}/`.
2.  **Data Handling:**
    - Copies the content of `--data-dir` to `inference_runs/{run_id}/data/`.
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