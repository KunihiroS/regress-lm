#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Fine-tunes a regression model based on user-provided data and parameters."""

import argparse
import os
import sys
from pathlib import Path
import re
import shutil
import yaml
from datetime import datetime, timezone
from typing import Optional
import time
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Add project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from regress_lm.rlm import RegressLM
from regress_lm.core import Example


def load_data_from_yaml(path: Path) -> list[Example]:
    """Loads data from a YAML file and returns it as a list of core.Example objects."""
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return [Example(x=item['text'], y=item['value']) for item in data]


def evaluate_model(model: RegressLM, eval_data: list[Example], num_prediction_samples: int) -> Optional[dict]:
    """
    Evaluates the model on the given data, generating multiple prediction samples for each
    data point, and returns a dictionary of results.
    """
    print(f"\n--- Evaluating model with {num_prediction_samples} samples per data point ---")

    if not eval_data:
        print("No evaluation data provided. Skipping evaluation.")
        return None

    # Use the high-level sample method to get multiple predictions per example
    prediction_samples = model.sample(eval_data, num_samples=num_prediction_samples)

    # Calculate the mean of the samples for each prediction to get a representative value
    mean_predictions = np.array([np.mean(s) for s in prediction_samples])
    actuals = np.array([e.y for e in eval_data])

    # Note: Loss calculation is not performed here as it's less representative
    # with sampling. The key metrics are based on prediction accuracy.
    print("Evaluation complete.")

    return {
        "mean_predictions": mean_predictions,
        "actuals": actuals,
        "prediction_samples": prediction_samples
    }


def plot_error_distribution(errors: np.ndarray, output_path: Path):
    """Generates a histogram of prediction errors and saves it as a PNG image."""
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, alpha=0.75, color='blue', edgecolor='black')
    plt.title('Prediction Error Distribution')
    plt.xlabel('Prediction Error (Predicted - Actual)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved prediction error histogram to {output_path}")


def generate_predictions_yaml(
    output_path: Path,
    eval_data: list[Example],
    prediction_samples: list[np.ndarray]
):
    """
    Generates a detailed predictions file in YAML format.

    Args:
        output_path: Path to save the YAML file.
        eval_data: The list of evaluation examples (containing text and actual values).
        prediction_samples: A list where each element is a numpy array of prediction
                            samples for the corresponding example in eval_data.
    """
    predictions_list = []
    for example, samples in zip(eval_data, prediction_samples):
        mean_pred = np.mean(samples)
        std_dev_pred = np.std(samples)

        prediction_entry = {
            'text': example.x,
            'actual_value': float(example.y),
            'prediction_summary': {
                'mean': float(mean_pred),
                'std_dev': float(std_dev_pred),
                'min': float(np.min(samples)),
                'max': float(np.max(samples)),
                'num_samples': len(samples)
            },
            'error': float(mean_pred - example.y)
        }
        predictions_list.append(prediction_entry)

    with open(output_path, 'w') as f:
        yaml.dump({'predictions': predictions_list}, f, default_flow_style=False, sort_keys=False, indent=2)
    print(f"Saved detailed predictions to {output_path}")


def generate_tuning_summary(
    summary_path: Path,
    job_name: str,
    version: int,
    base_model_path: Optional[Path],
    finetune_data_path: Path,
    eval_data_path: Path,
    checkpoint_path: Path,
    tuning_duration_seconds: float,
    eval_results: dict
) -> dict:
    """Generates a detailed tuning summary report and saves it as a YAML file."""
    # Use the mean of the prediction samples for overall performance metrics.
    predictions = np.array(eval_results["mean_predictions"])
    actuals = np.array(eval_results["actuals"])

    # Calculate performance metrics
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)

    # Calculate prediction error analysis
    errors = predictions - actuals
    error_mean = np.mean(errors)
    error_std = np.std(errors)
    error_min = np.min(errors)
    error_max = np.max(errors)
    q1, median, q3 = np.percentile(errors, [25, 50, 75])

    summary_data = {
        'overview': {
            'job_name': job_name,
            'version_created': version,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'base_model_used': str(base_model_path) if base_model_path else 'default',
        },
        'data_sources': {
            'finetuning_data': str(finetune_data_path),
            'evaluation_data': str(eval_data_path),
        },
        'performance_metrics': {
            'mse': float(mse),
            'mae': float(mae),
            'r2_score': float(r2),
        },
        'prediction_error_analysis': {
            'mean': float(error_mean),
            'std_dev': float(error_std),
            'min': float(error_min),
            'max': float(error_max),
            'quartiles': {
                'q1': float(q1),
                'median': float(median),
                'q3': float(q3),
            },
        },
        'process_timing': {
            'total_tuning_seconds': tuning_duration_seconds,
        },
        'output_files': {
            'checkpoint': str(checkpoint_path),
            'predictions_yaml': str(summary_path.parent / 'predictions.yaml'),
            'error_histogram': str(summary_path.parent / 'distribution.png'),
        },
    }

    with open(summary_path, 'w') as f:
        yaml.dump(summary_data, f, default_flow_style=False, sort_keys=False)
    print(f"Saved tuning summary to {summary_path}")

    # Return key metrics for history.yaml
    return {
        'mse': float(mse),
        'mae': float(mae),
        'r2_score': float(r2),
    }


def update_history(history_file: Path, version: int, finetune_data_path: Path, base_model_path: Optional[Path], checkpoint_path: Path, results: dict):
    """Updates the history file with the latest tuning event details."""
    history_data = []
    if history_file.exists():
        with open(history_file, 'r') as f:
            # Handle empty file case by checking content
            content = f.read()
            if content:
                history_data = yaml.safe_load(content)

    new_event = {
        'event_type': 'tuning',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'version': version,
        'input_data_dir': str(finetune_data_path.parent),
        'base_model': str(base_model_path) if base_model_path else 'default',
        'results': results,
        'checkpoint_path': str(checkpoint_path)
    }
    history_data.append(new_event)

    with open(history_file, 'w') as f:
        yaml.dump(history_data, f, default_flow_style=False, sort_keys=False)


def main():
    """Main function to handle argument parsing and job initialization."""
    parser = argparse.ArgumentParser(description="Fine-tune a regression model.")
    parser.add_argument(
        "--job-name",
        type=str,
        required=True,
        help="The name of the job to operate on."
    )
    parser.add_argument(
        "--data-file",
        type=str,
        required=True,
        help="Path to the single YAML file containing the data for this fine-tuning session."
    )
    parser.add_argument(
        "--eval-set-file",
        type=str,
        help="Path to the master evaluation YAML file. Required for new jobs."
    )
    parser.add_argument(
        "--base-model",
        type=str,
        help="Path to a pre-trained base model to start a new job from."
    )
    parser.add_argument(
        "--new",
        action="store_true",
        help="Explicitly create a new job."
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=10,
        help="Number of epochs for fine-tuning."
    )

    args = parser.parse_args()

    job_path = Path("work/jobs") / args.job_name

    if args.new:
        # --- NEW JOB CREATION ---
        if job_path.exists():
            print(f"Error: Job '{args.job_name}' already exists. Use a different name or run without --new to continue.", file=sys.stderr)
            sys.exit(1)
        if not args.eval_set_file:
            print("Error: --eval-set-file is required when creating a new job with --new.", file=sys.stderr)
            sys.exit(1)

        is_new_job = True
        version = 1
        print(f"Starting new job '{args.job_name}' as v{version}.")

        # Create directory structure according to memo.md
        (job_path / 'checkpoints').mkdir(parents=True, exist_ok=True)
        finetuning_data_path = job_path / 'finetuning' / 'data'
        standard_eval_path = finetuning_data_path / 'standard_eval_set'
        standard_eval_path.mkdir(parents=True, exist_ok=True)
        (job_path / 'finetuning' / 'results').mkdir(parents=True, exist_ok=True)
        (job_path / 'inference_runs').mkdir(parents=True, exist_ok=True)

        # Create version-specific directories
        version_data_path = finetuning_data_path / f'v{version}'
        (version_data_path / 'finetunes').mkdir(parents=True, exist_ok=True)
        (version_data_path / 'eval').mkdir(parents=True, exist_ok=True)
        (job_path / 'finetuning' / 'results' / f'v{version}').mkdir(parents=True, exist_ok=True)

        # Create README.md for the job
        readme_content = f"# Job: {args.job_name}\n\nThis directory contains all artifacts for the '{args.job_name}' job."
        (job_path / "README.md").write_text(readme_content)

        # Copy master evaluation set
        master_eval_file = standard_eval_path / "standard_eval.yaml"
        shutil.copyfile(args.eval_set_file, master_eval_file)

    else:
        # --- CONTINUE EXISTING JOB ---
        is_new_job = False
        if not job_path.exists():
            print(f"Error: Job '{args.job_name}' not found. To create a new job, use the --new flag.", file=sys.stderr)
            sys.exit(1)

        # Determine the latest version by looking in the correct data directory
        data_path = job_path / 'finetuning' / 'data'
        version_dirs = sorted(
            [d for d in data_path.iterdir() if d.is_dir() and d.name.startswith('v')],
            key=lambda d: int(d.name[1:])
        )

        if not version_dirs:
            print(f"Error: No versions found for job '{args.job_name}'. This may indicate a corrupted job directory.", file=sys.stderr)
            sys.exit(1)
        
        latest_version = int(version_dirs[-1].name[1:])
        version = latest_version + 1
        print(f"Continuing job '{args.job_name}'. Creating new version v{version}.")

        # Create version-specific directories for existing job
        version_data_path = job_path / 'finetuning' / 'data' / f'v{version}'
        (version_data_path / 'finetunes').mkdir(parents=True, exist_ok=True)
        (version_data_path / 'eval').mkdir(parents=True, exist_ok=True)
        (job_path / 'finetuning' / 'results' / f'v{version}').mkdir(parents=True, exist_ok=True)

    # --- Common Logic for both new and existing jobs ---
    version_data_path = job_path / 'finetuning' / 'data' / f'v{version}'
    master_eval_file = job_path / 'finetuning' / 'data' / 'standard_eval_set' / 'standard_eval.yaml'

    finetune_dest = version_data_path / 'finetunes' / f"{args.job_name}_v{version}_finetune.yaml"
    eval_dest = version_data_path / 'eval' / f"{args.job_name}_v{version}_eval.yaml"

    shutil.copyfile(args.data_file, finetune_dest)
    shutil.copyfile(master_eval_file, eval_dest)
    print(f"Prepared data for v{version} in {version_data_path}")

    # --- Model Training ---
    print("\n--- Starting Model Training ---")
    model = RegressLM.from_default()

    checkpoint_to_load = None
    if is_new_job:
        if args.base_model:
            checkpoint_to_load = Path(args.base_model)
    else:  # Continuing an existing job
        # latest_version was calculated when determining the new version number
        prev_checkpoint_path = job_path / 'checkpoints' / f'checkpoint_v{latest_version}.pt'
        if prev_checkpoint_path.exists():
            checkpoint_to_load = prev_checkpoint_path
        else:
            print(f"Warning: Checkpoint for v{latest_version} not found. Using fresh default model for v{version}.")

    if checkpoint_to_load:
        print(f"Loading checkpoint from: {checkpoint_to_load}")
        model.load_checkpoint(checkpoint_to_load)
    else:
        print("No checkpoint specified or found, using a fresh default model.")

    train_data = load_data_from_yaml(finetune_dest)
    print(f"Loading training data from {finetune_dest}")

    print("Starting fine-tuning...")
    start_time = time.time()
    model.fine_tune(train_data, max_epochs=args.max_epochs, batch_size=4)
    end_time = time.time()
    tuning_duration = end_time - start_time
    print(f"Fine-tuning complete in {tuning_duration:.2f} seconds.")

    new_checkpoint_path = job_path / 'checkpoints' / f'checkpoint_v{version}.pt'
    model.save_checkpoint(new_checkpoint_path)
    print(f"Saved new checkpoint for v{version} to {new_checkpoint_path}")

    # --- Evaluation ---
    eval_data = load_data_from_yaml(eval_dest)
    # Generate multiple samples per prediction for detailed analysis
    eval_results = evaluate_model(model, eval_data, num_prediction_samples=100)

    if eval_results is not None:
        results_path = job_path / 'finetuning' / 'results' / f'v{version}'
        results_path.mkdir(exist_ok=True)

        # Generate all specified output artifacts
        predictions_yaml_path = results_path / 'predictions.yaml'
        distribution_png_path = results_path / 'distribution.png'
        summary_file_path = results_path / 'tuning_summary.yaml'

        # Create detailed predictions file
        generate_predictions_yaml(
            output_path=predictions_yaml_path,
            eval_data=eval_data,
            prediction_samples=eval_results["prediction_samples"]
        )

        # Create prediction error histogram
        errors = eval_results["mean_predictions"] - eval_results["actuals"]
        plot_error_distribution(errors, distribution_png_path)

        # Generate the detailed tuning summary
        key_metrics = generate_tuning_summary(
            summary_path=summary_file_path,
            job_name=args.job_name,
            version=version,
            base_model_path=checkpoint_to_load,
            finetune_data_path=finetune_dest,
            eval_data_path=eval_dest,
            checkpoint_path=new_checkpoint_path,
            tuning_duration_seconds=tuning_duration,
            eval_results=eval_results
        )

        # Update history with key metrics from the summary
        history_file = job_path / 'history.yaml'
        update_history(
            history_file=history_file,
            version=version,
            finetune_data_path=finetune_dest,
            base_model_path=checkpoint_to_load,
            checkpoint_path=new_checkpoint_path,
            results=key_metrics
        )


if __name__ == "__main__":
    main()
