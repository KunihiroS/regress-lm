import yaml
import os
from pathlib import Path

def convert_and_merge_data(source_dir: str, output_file: str):
    """
    Merges multiple YAML files from a source directory into a single YAML file
    with the 'text' and 'value' keys.

    Args:
        source_dir: Directory containing the source YAML files.
        output_file: Path to the output YAML file.
    """
    merged_data = []
    source_path = Path(source_dir)
    
    if not source_path.is_dir():
        print(f"Error: Source directory '{source_dir}' not found.")
        return

    for filepath in sorted(source_path.glob('*.yml')):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            # Handle both single-entry dict and list of dicts formats
            if isinstance(data, dict) and 'news' in data and 'change_rate' in data:
                merged_data.append({
                    'text': data['news'],
                    'value': data['change_rate']
                })
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and 'news' in item and 'change_rate' in item:
                        merged_data.append({
                            'text': item['news'],
                            'value': item['change_rate']
                        })

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(merged_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"Successfully converted and merged {len(merged_data)} items into '{output_file}'")

if __name__ == '__main__':
    # For test-job-07, we'll use the data from test/cleaned_data/train
    # This script will need to be adapted if other data sources are used in the future.
    TRAIN_SOURCE_DIR = 'test/cleaned_data/train'
    EVAL_SOURCE_DIR = 'test/cleaned_data/test'
    OUTPUT_DIR = 'work'
    
    JOB_NAME = 'test-job-07'

    train_output_file = os.path.join(OUTPUT_DIR, f"{JOB_NAME}_train_data.yaml")
    eval_output_file = os.path.join(OUTPUT_DIR, f"{JOB_NAME}_eval_data.yaml")

    print(f"Preparing data for job: {JOB_NAME}")
    
    print("\nConverting training data...")
    convert_and_merge_data(TRAIN_SOURCE_DIR, train_output_file)
    
    print("\nConverting evaluation data...")
    convert_and_merge_data(EVAL_SOURCE_DIR, eval_output_file)
    
    print("\nData preparation complete.")
