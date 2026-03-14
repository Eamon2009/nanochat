"""
Unified evaluation script for base models.

Supports three evaluation modes (comma-separated):
  --eval core    : CORE metric (accuracy on ICL tasks)
  --eval bpb     : Bits per byte on train/val splits
  --eval sample  : Generate samples from the model
"""
import os
import csv
import time
import json
import yaml
import shutil
import random
import zipfile
import tempfile
import argparse
import torch

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, autodetect_device_type, download_file_with_lock
from nanochat.tokenizer import HuggingFaceTokenizer, get_token_bytes
from nanochat.checkpoint_manager import load_model
from nanochat.core_eval import evaluate_task
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine

# -----------------------------------------------------------------------------
# HuggingFace loading utilities

class ModelWrapper:
    """Lightweight wrapper to give HuggingFace models a nanochat-compatible interface."""
    def __init__(self, model, max_seq_len=None):
        self.model = model
        self.max_seq_len = max_seq_len

    def __call__(self, input_ids, targets=None, loss_reduction='mean'):
        logits = self.model(input_ids).logits
        if targets is None:
            return logits
        
        # LOGICAL FIX: Shift logits and targets for Causal LM loss calculation
        # Most HF models return logits for the full sequence; we need to align 
        # token n with target n+1.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()
        
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-1,
            reduction=loss_reduction
        )
        return loss

    def get_device(self):
        return next(self.model.parameters()).device


def load_hf_model(hf_path: str, device):
    """Load a HuggingFace model and tokenizer."""
    print0(f"Loading HuggingFace model from: {hf_path}")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(hf_path)
    model.to(device)
    model.eval()
    
    # IMPROVEMENT: Use config-defined sequence length if available
    max_seq_len = getattr(model.config, "max_position_embeddings", 1024)
    model = ModelWrapper(model, max_seq_len=max_seq_len)
    tokenizer = HuggingFaceTokenizer.from_pretrained(hf_path)
    return model, tokenizer


def get_hf_token_bytes(tokenizer, device="cpu"):
    """Compute token_bytes tensor for a HuggingFace tokenizer."""
    vocab_size = tokenizer.tokenizer.get_vocab_size()
    token_bytes = torch.zeros(vocab_size, dtype=torch.int64, device=device)
    for token_id in range(vocab_size):
        # Note: some tokens may not decode to valid utf-8 individually (byte-level BPE)
        # This is a heuristic approximation for BPB
        token_str = tokenizer.tokenizer.decode([token_id])
        token_bytes[token_id] = len(token_str.encode('utf-8'))
    return token_bytes

# -----------------------------------------------------------------------------
# CORE evaluation

EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"


def place_eval_bundle(file_path):
    """Unzip eval_bundle.zip and place it in the base directory."""
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted_bundle_dir = os.path.join(tmpdir, "eval_bundle")
        shutil.move(extracted_bundle_dir, eval_bundle_dir)
    print0(f"Placed eval_bundle directory at {eval_bundle_dir}")


def evaluate_core(model, tokenizer, device, max_per_task=-1):
    """
    Evaluate a base model on the CORE benchmark.
    """
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    if not os.path.exists(eval_bundle_dir):
        download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle)

    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    tasks = config['icl_tasks']

    random_baselines = {}
    with open(eval_meta_data, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            random_baselines[row['Eval Task']] = float(row['Random baseline'])

    results = {}
    centered_results = {}
    
    # MEMORY FIX: Ensure no gradients are tracked during heavy eval
    with torch.no_grad():
        for task in tasks:
            start_time = time.time()
            label = task['label']
            task_meta = {
                'task_type': task['icl_task_type'],
                'dataset_uri': task['dataset_uri'],
                'num_fewshot': task['num_fewshot'][0],
                'continuation_delimiter': task.get('continuation_delimiter', ' ')
            }
            print0(f"Evaluating: {label}... ", end='')

            data_path = os.path.join(data_base_path, task_meta['dataset_uri'])
            with open(data_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line.strip()) for line in f]

            shuffle_rng = random.Random(1337)
            shuffle_rng.shuffle(data)
            if max_per_task > 0:
                data = data[:max_per_task]

            accuracy = evaluate_task(model, tokenizer, data, device, task_meta)
            results[label] = accuracy
            
            rb = random_baselines[label]
            centered_result = (accuracy - 0.01 * rb) / (1.0 - 0.01 * rb)
            centered_results[label] = centered_result
            
            print0(f"acc: {accuracy:.4f} | centered: {centered_result:.4f} | {time.time()-start_time:.2f}s")

    core_metric = sum(centered_results.values()) / len(centered_results)
    return {"results": results, "centered_results": centered_results, "core_metric": core_metric}

# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Base model evaluation")
    parser.add_argument('--eval', type=str, default='core,bpb,sample')
    parser.add_argument('--hf-path', type=str, default=None)
    parser.add_argument('--model-tag', type=str, default=None)
    parser.add_argument('--step', type=int, default=None)
    parser.add_argument('--max_per_task', type=int, default=-1)
    parser.add_argument('--device-batch-size', type=int, default=32)
    parser.add_argument('--split-tokens', type=int, default=40*524288)
    parser.add_argument('--device-type', type=str, default='')
    args = parser.parse_args()

    eval_modes = set(mode.strip() for mode in args.eval.split(','))
    device_type = autodetect_device_type() if args.device_type == '' else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    if args.hf_path:
        model, tokenizer = load_hf_model(args.hf_path, device)
        sequence_len = model.max_seq_len
        token_bytes = get_hf_token_bytes(tokenizer, device=device)
        model_name = args.hf_path
        model_slug = args.hf_path.replace("/", "-")
    else:
        model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=args.model_tag, step=args.step)
        sequence_len = meta["model_config"]["sequence_len"]
        token_bytes = get_token_bytes(device=device)
        model_name = f"base_model (step {meta['step']})"
        model_slug = f"base_model_{meta['step']:06d}"

    print0(f"Evaluating model: {model_name} | Modes: {eval_modes}")

    core_results, bpb_results, samples, unconditioned_samples = None, {}, [], []

    # --- Sampling ---
    if 'sample' in eval_modes and not args.hf_path:
        if ddp_rank == 0:
            print0("\n" + "="*40 + "\nSamples\n" + "="*40)
            engine = Engine(model, tokenizer)
            prompts = ["The capital of France is", "The opposite of hot is", "If 5*x + 3 = 13, then x is"]
            with torch.no_grad():
                for p in prompts:
                    tokens = tokenizer(p, prepend="<|bos|>")
                    sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
                    s_str = tokenizer.decode(sample[0])
                    print0(f"Prompt: {p}\nOutput: {s_str}\n" + "-"*20)
                    samples.append(s_str)
    
    # --- BPB evaluation ---
    if 'bpb' in eval_modes:
        print0("\n" + "="*40 + "\nBPB Eval\n" + "="*40)
        tokens_per_step = args.device_batch_size * sequence_len * ddp_world_size
        
        # LOGICAL FIX: Ensure we perform at least one step
        steps = max(1, args.split_tokens // tokens_per_step)
        
        with torch.no_grad():
            for split in ["train", "val"]:
                loader = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, sequence_len, split, device=device)
                bpb = evaluate_bpb(model, loader, steps, token_bytes)
                bpb_results[split] = bpb
                print0(f"{split} bpb: {bpb:.6f}")

    # --- CORE evaluation ---
    if 'core' in eval_modes:
        # LOGICAL FIX: Avoid redundant work on multi-GPU if evaluate_core isn't sharded
        if ddp_rank == 0:
            print0("\n" + "="*40 + "\nCORE Eval\n" + "="*40)
            core_results = evaluate_core(model, tokenizer, device, max_per_task=args.max_per_task)
            
            out_path = os.path.join(get_base_dir(), "base_eval", f"{model_slug}.csv")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, 'w') as f:
                f.write("Task, Accuracy, Centered\n")
                for k in core_results["results"]:
                    f.write(f"{k}, {core_results['results'][k]:.6f}, {core_results['centered_results'][k]:.6f}\n")
                f.write(f"TOTAL CORE, , {core_results['core_metric']:.6f}\n")
            print0(f"CORE metric: {core_results['core_metric']:.4f}\nResults: {out_path}")

    # --- Log to report ---
    if ddp_rank == 0:
        from nanochat.report import get_report
        report_data = [{"model": model_name}]
        if core_results:
            report_data[0]["CORE metric"] = core_results["core_metric"]
            report_data.append(core_results["centered_results"])
        if bpb_results:
            report_data[0].update({f"{k} bpb": v for k, v in bpb_results.items()})
        get_report().log(section="Base model evaluation", data=report_data)

    compute_cleanup()

if __name__ == "__main__":
    main()
