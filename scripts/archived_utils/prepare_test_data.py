import yaml
import argparse
from pathlib import Path

def main():
    """
    Aggregates individual YAML data files from a source directory into a single
    dataset file, conforming to the project's list-of-dicts format.
    """
    parser = argparse.ArgumentParser(description="Prepare test dataset for regress-lm.")
    parser.add_argument("--source-dir", type=str, required=True, help="Directory containing the source .yml files.")
    parser.add_argument("--output-file", type=str, required=True, help="Path for the aggregated output YAML file.")
    args = parser.parse_args()

    source_path = Path(args.source_dir)
    output_path = Path(args.output_file)

    if not source_path.is_dir():
        print(f"Error: Source directory not found at '{source_path}'")
        return

    all_data = []
    print(f"Reading files from {source_path}...")
    for yaml_file in sorted(source_path.glob("*.yml")):
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
            # Convert to the required {'text': ..., 'value': ...} format
            # The source format is {'news': ..., 'change_rate': ...}
            if 'news' in data and 'change_rate' in data:
                formatted_data = {
                    'text': data['news'],
                    'value': data['change_rate']
                }
                all_data.append(formatted_data)
            else:
                print(f"Warning: Skipping file with unexpected format: {yaml_file}")

    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the aggregated data to the output file
    with open(output_path, 'w') as f:
        yaml.dump(all_data, f, allow_unicode=True, sort_keys=False)

    print(f"Successfully created aggregated dataset at '{output_path}' with {len(all_data)} records.")

if __name__ == "__main__":
    main()
