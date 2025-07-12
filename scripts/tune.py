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

# Add project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from regress_lm.rlm import RegressLM


def load_data_from_yaml(file_path):
    """Loads data from a YAML file and returns it as a list of (text, value) tuples."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    # Convert list of dicts to list of tuples
    return [(item['text'], item['value']) for item in data]


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

    args = parser.parse_args()

    job_path = Path("work/jobs") / args.job_name
    is_new_job = not job_path.exists()

    version = 1
    
    if is_new_job:
        print(f"Initializing new job: {args.job_name}")
        if not args.eval_set_file:
            print("Error: --eval-set-file is required for a new job.", file=sys.stderr)
            sys.exit(1)
        
        # Create the full directory structure
        paths_to_create = [
            job_path / "checkpoints",
            job_path / f"finetuning/data/v{version}/finetunes",
            job_path / f"finetuning/data/v{version}/eval",
            job_path / "finetuning/data/standard_eval_set",
            job_path / f"finetuning/results/v{version}",
            job_path / "inference_runs"
        ]

        for path in paths_to_create:
            path.mkdir(parents=True, exist_ok=True)
        
        (job_path / "history.yaml").touch()
        (job_path / "README.md").touch()
        print(f"Created new job structure at {job_path}")
        
        # Copy master evaluation set
        master_eval_file = job_path / "finetuning/data/standard_eval_set/standard_eval.yaml"
        shutil.copyfile(args.eval_set_file, master_eval_file)
        print(f"Copied master evaluation set to {master_eval_file}")

    else:
        print(f"Found existing job: {args.job_name}")
        finetuning_data_root = job_path / "finetuning/data"
        
        existing_versions = [
            int(re.search(r'v(\d+)', d.name).group(1))
            for d in finetuning_data_root.iterdir()
            if d.is_dir() and re.match(r'v\d+', d.name)
        ]
        
        version = max(existing_versions) + 1 if existing_versions else 1
        print(f"Starting new version: {version}")

        # Create directories for the new version
        (job_path / f"finetuning/data/v{version}/finetunes").mkdir(parents=True, exist_ok=True)
        (job_path / f"finetuning/data/v{version}/eval").mkdir(parents=True, exist_ok=True)
        (job_path / f"finetuning/results/v{version}").mkdir(parents=True, exist_ok=True)
        print(f"Created directories for version {version}")

    # --- Common Logic for both new and existing jobs ---
    finetune_data_path = job_path / f"finetuning/data/v{version}"
    master_eval_file = job_path / "finetuning/data/standard_eval_set/standard_eval.yaml"
    
    # Data Preparation
    finetune_dest = finetune_data_path / f"finetunes/{args.job_name}_v{version}_finetune.yaml"
    eval_dest = finetune_data_path / f"eval/{args.job_name}_v{version}_eval.yaml"
    shutil.copyfile(args.data_file, finetune_dest)
    shutil.copyfile(master_eval_file, eval_dest)
    print(f"Prepared data for v{version} in {finetune_data_path}")

    # Model Training
    print("\n--- Starting Model Training ---")
    model = RegressLM()

    checkpoint_to_load = None
    if is_new_job:
        if args.base_model:
            checkpoint_to_load = args.base_model
            print(f"Initializing job from base model: {checkpoint_to_load}")
        else:
            print("Initializing job from default pre-trained model.")
    else:
        prev_version = version - 1
        if prev_version > 0:
            checkpoint_to_load = job_path / f"checkpoints/checkpoint_v{prev_version}.pt"
            if checkpoint_to_load.exists():
                print(f"Loading checkpoint from previous version: {checkpoint_to_load}")
            else:
                print(f"Warning: Checkpoint for v{prev_version} not found. Training from default model.", file=sys.stderr)
                checkpoint_to_load = None
        else:
             if args.base_model:
                checkpoint_to_load = args.base_model
                print(f"Initializing job from base model: {checkpoint_to_load}")
             else:
                print("Starting v1 of existing job from default pre-trained model.")

    if checkpoint_to_load:
        model.load_checkpoint(checkpoint_to_load)

    train_data = load_data_from_yaml(finetune_dest)
    print(f"Loading training data from {finetune_dest}")
    
    print("Starting fine-tuning...")
    model.fine_tune(train_data, max_epochs=10)
    print("Fine-tuning complete.")

    new_checkpoint_path = job_path / f"checkpoints/checkpoint_v{version}.pt"
    model.save_checkpoint(new_checkpoint_path)
    print(f"Saved new checkpoint for v{version} to {new_checkpoint_path}")


if __name__ == "__main__":
    main()
