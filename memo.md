# Project Specifications

## Scope of Responsibility

### User Responsibilities (Strategy & Preparation)

- **Dataset Preparation**: The user is responsible for creating and preparing dataset files. This involves:
  - Sourcing raw data.
  - Deciding on the data strategy (e.g., which data points to include for a specific tuning or evaluation session).
  - Formatting the data into a single YAML file that conforms to the specifications outlined in the "Input Data Format" section below.
- **Execution Command**: The user initiates the automated workflow by running the appropriate script (`tune.py` or `infer.py`) with the correct arguments (e.g., job name, path to the prepared dataset file).

### Software Responsibilities (Automation of Routine Tasks)

Once initiated by the user, the software is responsible for the entire automated workflow:

- **Directory Management**: Automatically creates and manages the complete, versioned directory structure as defined in the `Structure` section.
- **Data Handling**: Copies the user-provided dataset files into the managed job directory with the standardized naming convention.
- **Model & Checkpoint Lifecycle**: Loads the appropriate model, runs the fine-tuning or inference process, and saves the resulting checkpoints.
- **Result Generation**: (Future) Generates all report files (`tuning_summary.yaml`, `predictions.yaml`, etc.) as specified.

## Job Structure

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
│   │   └── standard_eval_set/         # Master evaluation set for the entire job
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

### 1. Input Data Format

- **Principle:** One YAML file represents a complete dataset (a collection of data points).
- **Format:** The data format is distinct for training/evaluation versus inference.

#### Fine-tuning & Evaluation Data Format
For fine-tuning and evaluation, each entry in the list **must** contain both `text` and `value` keys.

```yaml
# Fine-tuning/Evaluation Data Example
- text: "Apple's quarterly earnings exceeded expectations by 5%."
  value: 5.0
- text: "The new iPhone model saw a 10% increase in pre-orders."
  value: 10.0
```

#### Inference Data Format
For inference, each entry in the list **must** contain only the `text` key. The `value` key must be omitted.

```yaml
# Inference Data Example
- text: "Market sentiment is positive following the latest tech conference."
- text: "New regulations are expected to impact the automotive sector."
```

- **File Naming and Usage:**
  - **Fine-tuning Data:** A user-provided dataset file (using the Fine-tuning/Evaluation format) passed to `tune.py` via the `--data-file` argument.
  - **Evaluation Data:**
    - **Principle:** To ensure a fair and consistent comparison of model performance across different versions (v1, v2, ...), the evaluation dataset is fixed for the entire lifecycle of a job.
    - **Master Copy (`standard_eval.yaml`):** When a job is first created, the file provided via `--eval-set-file` is copied to `finetuning/data/standard_eval_set/standard_eval.yaml`. This file acts as the single, immutable "source of truth" for evaluation for the entire job.
    - **Per-Version Snapshot (`.../v{n}/eval/`):** The existence of a separate `eval` directory for each version, containing a copy of the master evaluation set, is a deliberate design choice to guarantee **long-term reproducibility and auditability**.
      - **Purpose (Why this exists):** It creates a self-contained, immutable historical record. Each version folder (e.g., `v2/`) acts as a "time capsule," containing the *exact* training data and evaluation data used for that specific run. This prevents future modifications to the master `standard_eval.yaml` from retroactively corrupting the historical record of past versions.
      - **Behavior (What happens):** During each tuning run, the script automatically copies the master `standard_eval.yaml` into the corresponding version's `eval` directory. This ensures that anyone can audit or perfectly reproduce a past version's results by only looking inside that version's directory, without ambiguity.
  - **Inference Data:** A user-provided dataset file (using the Inference format) passed to a future `infer.py` script.

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

#### Interpretation
This file serves as the primary "report card" for a tuning run. It provides a high-level, quantitative overview of the model's performance.
- **`performance_metrics`**: These are standard academic metrics to measure prediction accuracy. Lower `mse` (Mean Squared Error) and `mae` (Mean Absolute Error) are better. `r2_score` closer to 1.0 indicates a better fit.
- **`prediction_error_analysis`**: This section reveals the model's tendencies. The `mean` of the error shows if the model has a systematic bias (e.g., a positive mean means it tends to over-predict). The `std_dev` (Standard Deviation) shows how consistent the errors are; a smaller value is better.

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

#### Interpretation
This file allows for a detailed, micro-level analysis of the model's predictions. It's used to investigate exactly which data points the model performed well or poorly on.
- **`actual_value`**: The ground-truth value for comparison.
- **`prediction_summary`**: This block shows the statistics of the multiple prediction samples generated for a single input.
  - **`mean`**: This is the representative predicted value for the data point.
  - **`std_dev`**: This indicates the model's uncertainty or "confidence" for this specific prediction. A high standard deviation means the model's predictions for this item were scattered and inconsistent.
- **`error`**: This field (`mean - actual_value`) makes it easy to sort and find the data points with the largest prediction errors.

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

### 6. Distribution Plot (`distribution.png`)

- **Filename:** `distribution.png`
- **Type:** PNG Image
- **Purpose:** To provide a quick visual understanding of the model's predictions. The meaning of the plot differs depending on the context (tuning vs. inference).

#### In a Tuning Context (Error Distribution)

- **Content:** A histogram visualizing the distribution of **prediction errors** (`actual_value - predicted_value`).

##### Interpretation
This histogram provides a quick, visual summary of the model's prediction accuracy, complementing the statistics in `tuning_summary.yaml`.
- **X-axis (Prediction Error)**: Shows how far off a prediction was from the actual value. `0` represents a perfect prediction.
- **Y-axis (Frequency)**: Shows how many predictions fell into a particular error range.

**What to Look For:**
- **Ideal Shape**: A tall, narrow spike centered at `0`. This indicates that most predictions were highly accurate with very little variance.
- **Bias**: If the entire histogram is shifted to the left or right of `0`, it signifies a systematic bias (the model consistently under-predicts or over-predicts).
- **Variance**: A wide, flat histogram means the predictions are inconsistent and spread out.
- **Outliers**: Bars far from the center represent specific predictions that were very wrong. These can be investigated further using `predictions.yaml`.

##### Plot Specifications:
- **Title:** "Prediction Error Distribution"
- **X-axis Label:** "Prediction Error"
- **Y-axis Label:** "Frequency"

#### In an Inference Context (Prediction Distribution)

- **Content:** A histogram visualizing the distribution of the **final predicted values** themselves.

##### Interpretation
This histogram shows the overall trend of the model's predictions for a given inference dataset. It helps in understanding the nature of the outputs without ground truth.
- **X-axis (Predicted Value)**: The range of values the model predicted.
- **Y-axis (Frequency)**: How many predictions fell into a particular value range.

**What to Look For:**
- **Central Tendency**: Where the predictions are centered. This shows the most common output value.
- **Spread (Variance)**: A narrow distribution means the model's predictions are very consistent across different inputs. A wide distribution suggests the outputs vary significantly.
- **Modality**: A single peak (unimodal) suggests a consistent prediction pattern. Multiple peaks (multimodal) might indicate the model is identifying different subgroups within the inference data.

##### Plot Specifications:
- **Title:** "Prediction Value Distribution"
- **X-axis Label:** "Predicted Value"
- **Y-axis Label:** "Frequency"

---

## `tune.py`

**Purpose:** Manages the fine-tuning process, creating new model versions and evaluating their performance.

**Arguments:**
- `--job-name` (string, required): The name of the job to operate on.
- `--data-file` (path, required): Path to the single YAML file containing the data for this fine-tuning session.
- `--eval-set-file` (path): Path to the master evaluation YAML file. **This argument is required and used ONLY when creating a new job (i.e., when the `--new` flag is present).**
- `--base-model` (path, optional): Path to a pre-trained base model to start a new job from.
- `--new` (flag, optional): **Explicitly declares the intention to create a new job.** If this flag is present, the script will create a new job directory. If a job with the same name already exists, the script will exit with an error to prevent accidental overwrites.

**Behavior:**
1.  **If `--new` flag is present (New Job Creation):**
    - The script operates in "new job" mode.
    - It checks if `work/jobs/{job_name}` already exists. If it does, the script exits with an error to prevent accidental collision.
    - If it does not exist, it creates the entire directory structure for `v1`.
    - `--eval-set-file` is required in this mode.
2.  **If `--new` flag is NOT present (Existing Job Continuation):**
    - The script operates in "continue job" mode.
    - It checks if `work/jobs/{job_name}` exists. If it does NOT, the script exits with an error, prompting the user to use `--new` if they intended to create a new job.
    - If it exists, it determines the next version number (e.g., `v(n+1)`) and proceeds with the fine-tuning process.
    - `--eval-set-file` is ignored in this mode.
3.  **Data Handling:**
    - Copies the content of `--data-file` to `finetuning/data/v<n>/finetunes/{job_name}_v{n}_finetune.yaml`.
    - On the first run, copies `--eval-set-file` to `finetuning/data/standard_eval_set/standard_eval.yaml`.
    - Copies the master `standard_eval_set` to `finetuning/data/v<n>/eval/{job_name}_v{n}_eval.yaml`.
    - Copies the content of `--data-dir` to `finetuning/data/v<n>/finetunes/`.
    - On the first run, copies `--eval-set-dir` to `finetuning/data/standard_eval_set/`.
4.  **Tuning Process:**
    - Loads the latest checkpoint (`checkpoint_v<n-1>.pt`). If it's the first run, it loads the `--base-model` or the default model.
    - Fine-tunes the model using the data in `finetuning/data/v<n>/finetunes/`.
    - Saves the newly trained model as `checkpoints/checkpoint_v<n>.pt`.
5.  **Evaluation & Reporting:**
    - Evaluates the new model's performance using the data in `finetuning/data/v<n>/eval/`.
    - Generates `tuning_summary.yaml`, `distribution.png`, and `predictions.yaml` in `finetuning/results/v<n>/`.
    - Appends a summary of the run to `history.yaml`.

---

## `infer.py`

**Purpose:** Runs inference using a specified model checkpoint.

**Arguments:**
- `--job-name` (string, required): The name of the job.
- `--checkpoint-version` (integer, required): The version number of the checkpoint to use for inference.
- `--data-file` (path, required): Path to the single YAML file containing the data for inference.
- `--run-id` (string, required): A unique identifier for this inference run (e.g., `20250713_003000`) to be used as the directory name.

**Behavior:**
1.  Creates the run directory: `inference_runs/{run_id}/`.
2.  **Data Handling:**
    - Copies the content of `--data-file` to `inference_runs/{run_id}/data/`.
3.  **Inference Process:**
    - Loads the specified model from `checkpoints/checkpoint_v<version>.pt`.
    - Runs inference on the data located in `inference_runs/{run_id}/data/`.
4.  **Reporting:**
    - Generates `inference_report.yaml`, `distribution.png` (if applicable), and `predictions.yaml` in `inference_runs/{run_id}/results/`.
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

---

## Glossary of Terms

This section explains the key metrics and concepts used in the evaluation reports.

### Performance Metrics

- **MSE (Mean Squared Error)**
  - **What it is:** A standard way to measure the average squared difference between the estimated values and the actual value.
  - **Interpretation:** It heavily penalizes larger errors. A lower value is better. A value of 0 means the model is perfect.

- **MAE (Mean Absolute Error)**
  - **What it is:** The average of the absolute differences between the predictions and the actual values.
  - **Interpretation:** It's easier to interpret than MSE as it's in the same units as the original data. A lower value is better.

- **R² Score (Coefficient of Determination)**
  - **What it is:** A statistical measure of how well the regression predictions approximate the real data points.
  - **Interpretation:** A score of 1.0 indicates that the model perfectly predicts the data. A score of 0 indicates the model is no better than just predicting the mean of the actual values.

### Error Concepts

- **Prediction Error**
  - **What it is:** The simple difference between a single prediction and its corresponding actual value (`Prediction - Actual`).
  - **Interpretation:** It shows the direction and magnitude of a single miss. A positive error is an over-prediction; a negative error is an under-prediction.

- **Bias**
  - **What it is:** The tendency of a model to consistently predict higher or lower than the actual value.
  - **Interpretation:** In the reports, the `mean` of the `prediction_error_analysis` indicates bias. A value far from zero suggests a systematic bias.

- **Variance**
  - **What it is:** A measure of how much a model's predictions would change if it were trained on different data.
  - **Interpretation:** High variance is indicated by a wide, flat histogram in `distribution.png` or a large `std_dev` in the `prediction_error_analysis`.

### Prediction Distribution Concepts

- **Prediction Distribution**
  - **What it is:** For a single input, the model can generate multiple different predictions (samples). The collection of these samples forms a distribution.
  - **Interpretation:** This distribution reveals the model's certainty for a given prediction.

- **Mean (of prediction distribution)**
  - **What it is:** The average of the multiple prediction samples taken for a single input.
  - **Interpretation:** This is used as the single, most representative prediction for that input.

- **Standard Deviation (std_dev)**
  - **What it is:** A measure of the spread or dispersion of the prediction samples for a single input.
  - **Interpretation:** This indicates the model's "confidence" or consistency. A low `std_dev` means the model consistently produced similar predictions (high confidence). A high `std_dev` means the predictions were scattered (low confidence).