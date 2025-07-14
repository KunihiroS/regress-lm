import yaml
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Create clean, non-overlapping datasets for a job.')
    parser.add_argument('--master-dataset', type=str, required=True, help='Path to the master dataset YAML file.')
    parser.add_argument('--eval-dataset', type=str, required=True, help='Path to the evaluation dataset YAML file.')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save the output files.')
    parser.add_argument('--v1-size', type=int, default=50, help='Number of items for the v1 fine-tuning set.')
    args = parser.parse_args()

    print(f'Creating dataset directory: {args.output_dir}')
    os.makedirs(args.output_dir, exist_ok=True)

    print(f'Loading master data from: {args.master_dataset}')
    with open(args.master_dataset, 'r', encoding='utf-8') as f:
        master_data = yaml.safe_load(f)
    
    print(f'Loading evaluation data from: {args.eval_dataset}')
    with open(args.eval_dataset, 'r', encoding='utf-8') as f:
        eval_data = yaml.safe_load(f)

    # Create a set of evaluation texts for efficient lookup
    eval_texts = {item['text'] for item in eval_data}
    print(f'Found {len(eval_texts)} unique texts in evaluation data.')

    # Filter master data to create a clean training pool (no overlap with eval)
    train_pool = [item for item in master_data if item['text'] not in eval_texts]
    print(f'Created a clean training pool of {len(train_pool)} items.')

    # Split the training pool into v1 and v2
    v1_data = train_pool[:args.v1_size]
    v2_data = train_pool[args.v1_size:]
    print(f'Splitting into v1 ({len(v1_data)} items) and v2 ({len(v2_data)} items).')

    # --- Write output files ---
    def write_yaml(path, data):
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, indent=2, allow_unicode=True)
        print(f'Successfully wrote {len(data)} items to {path}')

    # Write v1, v2, and a copy of the eval set to the output directory
    write_yaml(os.path.join(args.output_dir, 'v1_finetune.yaml'), v1_data)
    write_yaml(os.path.join(args.output_dir, 'v2_finetune.yaml'), v2_data)
    write_yaml(os.path.join(args.output_dir, 'eval.yaml'), eval_data)

    print('\nData preparation complete. All files are valid and stored in the job directory.')

if __name__ == '__main__':
    main()
