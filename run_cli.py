import argparse
import logging
import os
import datetime
import torch
import time

from pathlib import Path
from regress_lm import core
from regress_lm import rlm

def handle_train(args):
    """Handles the 'train' subcommand."""
    logging.info("Executing train command...")
    logging.info(f"Arguments: {args}")

    # --- 計測開始 ---
    start_time = time.time()
    model_instantiation_time = None
    fine_tuning_time = None

    # 1. Extract data_label from the data directory path
    data_dir_path = Path(args.data_dir)
    if not data_dir_path.is_dir():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    data_label = data_dir_path.name
    logging.info(f"Data label extracted: {data_label}")

    # 2. Generate a timestamp for the current run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.info(f"Generated timestamp: {timestamp}")

    # 3. Create model and output save directories
    model_run_dir = Path(args.model_save_dir) / data_label / timestamp
    output_run_dir = Path(args.output_dir) / data_label / timestamp
    
    logging.info(f"Creating model save directory: {model_run_dir}")
    model_run_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Creating output directory: {output_run_dir}")
    output_run_dir.mkdir(parents=True, exist_ok=True)

    # 4. Validate checkpoint if resuming
    if args.resume_from_checkpoint:
        logging.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        checkpoint_path = Path(args.resume_from_checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {args.resume_from_checkpoint}")
        
        # Extract original data_label from the checkpoint path structure: .../<data_label>/<timestamp>/...
        try:
            original_data_label = checkpoint_path.parent.parent.name
            logging.info(f"Original data label from checkpoint path: {original_data_label}")
        except IndexError:
            raise ValueError("Could not determine original data label from checkpoint path structure. Expected '.../<data_label>/<timestamp>/checkpoint.pt'")

        if original_data_label != data_label:
            raise ValueError(
                f"Data label mismatch! Model was trained on '{original_data_label}', "
                f"but trying to fine-tune with '{data_label}'. Halting to prevent model contamination."
            )

    # 5. Initialize model and fine-tune
    logging.info("Initializing model...")
    model_init_start = time.time()
    reg_lm = rlm.RegressLM.from_default(max_input_len=args.max_input_len)
    model_instantiation_time = time.time() - model_init_start
    logging.info(f"Model initialized in {model_instantiation_time:.2f} seconds.")

    if args.resume_from_checkpoint:
        logging.info("Loading model state from checkpoint...")
        reg_lm.model.load_state_dict(torch.load(args.resume_from_checkpoint))

    logging.info("Loading training and evaluation data...")
    train_examples = rlm.load_examples_from_yaml(str(data_dir_path / 'train.yml'))
    eval_examples = rlm.load_examples_from_yaml(str(data_dir_path / 'test.yml'))

    logging.info(f"Starting fine-tuning (epochs={args.max_epochs}, batch_size={args.batch_size})")
    fine_tuning_start = time.time()
    # Fine-tune; retry with batch_size=1 on OOM
    try:
        reg_lm.fine_tune(train_examples, validation_examples=eval_examples,
                         max_epochs=args.max_epochs, batch_size=args.batch_size)
    except RuntimeError as e:
        msg = str(e).lower()
        if 'out of memory' in msg or 'cuda out of memory' in msg:
            logging.warning("OOM detected during fine-tuning, retrying with batch_size=1...")
            # clear GPU cache and retry
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                reg_lm.fine_tune(train_examples, validation_examples=eval_examples, batch_size=1)
            except RuntimeError as e2:
                msg2 = str(e2).lower()
                if 'out of memory' in msg2:
                    logging.error("Still OOM on retry with batch_size=1. Aborting training.")
                    return
                else:
                    raise
        else:
            raise
    fine_tuning_time = time.time() - fine_tuning_start
    logging.info(f"Fine-tuning completed in {fine_tuning_time:.2f} seconds.")

    # 6. Save the model
    model_save_path = model_run_dir / 'checkpoint.pt'
    logging.info(f"Saving model to {model_save_path}")
    torch.save(reg_lm.model.state_dict(), str(model_save_path))

    # 7. Save evaluation results
    logging.info("Generating and saving evaluation results...")
    eval_inputs = [core.ExampleInput(x=e.x) for e in eval_examples]
    
    inference_start = time.time()
    # eval_examplesにはy_samplesが含まれていないので、推論を再実行
    eval_results = reg_lm.sample(eval_inputs, num_samples=args.num_samples)
    inference_time = time.time() - inference_start
    
    for result in eval_results:
        performance_stats = {
            "model_instantiation_sec": model_instantiation_time,
            "fine_tuning_sec": fine_tuning_time,
            "inference_sec": inference_time / len(eval_inputs) # 1サンプルあたりの推論時間
        }
        rlm.save_statistics_and_plot(
            query_text=result.x,
            samples=result.y_samples,
            output_dir=str(output_run_dir),
            performance_stats=performance_stats
        )

    logging.info(f"Evaluation results saved in {output_run_dir}")
    logging.info("Train command finished successfully.")


def handle_infer(args):
    """Handles the 'infer' subcommand."""
    logging.info("Executing infer command...")
    logging.info(f"Arguments: {args}")

    # --- 計測開始 ---
    start_time = time.time()
    model_instantiation_time = None
    inference_time = None

    # 1. Check if checkpoint exists
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint_path}")

    # 2. Determine input text from the provided argument
    input_text = ""
    input_source = ""
    if args.text:
        input_text = args.text
        input_source = f"text: '{args.text}'"
    elif args.input_file:
        input_file = Path(args.input_file)
        if not input_file.is_file():
            raise FileNotFoundError(f"Input file not found: {args.input_file}")
        input_text = input_file.read_text()
        input_source = f"file: {args.input_file}"
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
        
        all_texts = []
        for file_path in sorted(input_dir.glob('*')): # sorted for deterministic order
            if file_path.is_file():
                all_texts.append(file_path.read_text())
        input_text = "".join(all_texts)
        input_source = f"directory: {args.input_dir}"

    if not input_text:
        logging.warning("Input text is empty. Nothing to process.")
        print("Input is empty, cannot run inference.")
        return

    # 3. Initialize model
    logging.info("Initializing model...")
    model_init_start = time.time()
    reg_lm = rlm.RegressLM.from_default(max_input_len=args.max_input_len)
    model_instantiation_time = time.time() - model_init_start
    logging.info(f"Model initialized in {model_instantiation_time:.2f} seconds.")

    # 4. Load model state from checkpoint
    logging.info(f"Loading model state from {args.checkpoint_path}...")
    reg_lm.model.load_state_dict(torch.load(args.checkpoint_path))
    reg_lm.model.eval()  # Set model to evaluation mode

    # 5. Prepare input and run inference
    logging.info(f"Running inference on input from {input_source}")
    query_input = core.ExampleInput(x=input_text)
    
    inference_start = time.time()
    results = reg_lm.sample([query_input], num_samples=args.num_samples)
    inference_time = time.time() - inference_start
    
    result = results[0]

    # 6. Process and print/save results
    predictions = result.y_samples
    if predictions is not None and len(predictions) > 0:
        stats = rlm.calculate_statistics(predictions)
        mean_prediction = stats["mean"]
        
        print(f"Input Source: {input_source}")
        print(f"Predicted Value (Mean): {mean_prediction:.4f}")

        # --output-dir が指定されていればファイルに保存
        if args.output_dir:
            output_dir_path = Path(args.output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)
            
            performance_stats = {
                "model_instantiation_sec": model_instantiation_time,
                "inference_sec": inference_time
            }
            
            # 出力ファイル名を生成 (data_labelとtimestampを使用)
            data_label = Path(args.checkpoint_path).parent.parent.name
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_sub_dir = output_dir_path / data_label / timestamp
            output_sub_dir.mkdir(parents=True, exist_ok=True)

            rlm.save_statistics_and_plot(
                query_text=result.x,
                samples=result.y_samples,
                output_dir=str(output_sub_dir),
                performance_stats=performance_stats
            )
            logging.info(f"Inference results saved in {output_sub_dir}")
            print(f"Full results saved to: {output_sub_dir}")
        # --output-dir がなければコンソールに詳細を出力
        else:
            print("\n--- Statistics ---")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"- {key}: {value:.4f}")
                else:
                    print(f"- {key}: {value}")

    else:
        logging.warning("Inference produced no samples.")
        print(f"Could not generate a prediction for input from: {input_source}")
    
    logging.info("Infer command finished successfully.")


def main():
    """Main function to parse arguments and dispatch to handlers."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="RegressLM Command Line Interface")
    subparsers = parser.add_subparsers(dest='command', required=True, help='Sub-command help')

    # --- Train sub-command ---
    parser_train = subparsers.add_parser('train', help='Train a new model or fine-tune an existing one.')
    parser_train.add_argument('--data-dir', type=str, required=True, help='Directory containing the training data (train.yml, test.yml).')
    parser_train.add_argument('--model-save-dir', type=str, required=True, help='Parent directory to save the trained model.')
    parser_train.add_argument('--output-dir', type=str, required=True, help='Parent directory to save evaluation results (e.g., plots).')
    parser_train.add_argument('--resume-from-checkpoint', type=str, help='Path to a checkpoint file to resume fine-tuning.')
    parser_train.add_argument('--max-input-len', type=int, default=2048, help='Maximum input token length for the model.')
    parser_train.add_argument('--num-samples', type=int, default=128, help='Number of samples to generate for evaluation.')

    # 追加: トレーニング時のエポック数とバッチサイズを指定できるようにする
    parser_train.add_argument('--max-epochs', type=int, default=100, help='Maximum number of epochs for training.')
    parser_train.add_argument('--batch-size', type=int, default=None, help='Batch size for training. If not set, uses full dataset.')
    parser_train.set_defaults(func=handle_train)

    # --- Infer sub-command ---
    parser_infer = subparsers.add_parser('infer', help='Run inference with a trained model.')
    parser_infer.add_argument('--checkpoint-path', type=str, required=True, help='Path to the trained model checkpoint file.')
    
    # Mutually exclusive group for input type
    input_group = parser_infer.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--text', type=str, help='Input text to run inference on.')
    input_group.add_argument('--input-file', type=str, help='Path to a file. The entire content will be used as a single input.')
    input_group.add_argument('--input-dir', type=str, help='Path to a directory. Contents of all files will be concatenated and used as a single input.')

    parser_infer.add_argument('--output-dir', type=str, help='(Optional) Directory to save inference results (text stats and plot). If not provided, results are printed to console.')
    parser_infer.add_argument('--max-input-len', type=int, default=2048, help='Maximum input token length for the model.')
    parser_infer.add_argument('--num-samples', type=int, default=128, help='Number of samples to generate for inference.')
    parser_infer.set_defaults(func=handle_infer)

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
