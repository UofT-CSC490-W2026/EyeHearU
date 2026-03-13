"""
Detailed per-problem evaluation on GSM8K.

Runs inference on every GSM8K test problem and saves per-problem results
(question, ground truth, model response, correctness) to a JSON file
for downstream analysis and clustering.

Usage:
  python -m scripts.gsm8k_detailed_eval --source rl --model-tag d12
  torchrun --standalone --nproc_per_node=4 -m scripts.gsm8k_detailed_eval -- --source rl --model-tag d12
"""

import argparse
import json
import os
import re

import torch
import torch.distributed as dist

from nanochat.common import (
    compute_init, compute_cleanup, print0,
    autodetect_device_type, get_base_dir,
)
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K, extract_answer

parser = argparse.ArgumentParser(description="Detailed per-problem GSM8K evaluation")
parser.add_argument('--source', type=str, required=True, help='Model source: sft|rl')
parser.add_argument('--model-tag', type=str, default=None, help='Model tag to load')
parser.add_argument('--step', type=int, default=None, help='Checkpoint step')
parser.add_argument('--max-new-tokens', type=int, default=512)
parser.add_argument('--temperature', type=float, default=0.0)
parser.add_argument('--top-k', type=int, default=50)
parser.add_argument('--device-type', type=str, default='')
parser.add_argument('--output-filename', type=str, default=None,
                    help='Output JSON filename (saved to report dir)')
args = parser.parse_args()

device_type = autodetect_device_type() if args.device_type == '' else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0

model, tokenizer, meta = load_model(
    args.source, device, phase="eval",
    model_tag=args.model_tag, step=args.step,
)
engine = Engine(model, tokenizer)

task = GSM8K(subset="main", split="test")
print0(f"GSM8K test set: {len(task)} problems")

results = []
num_correct = 0
total = 0

for idx in range(ddp_rank, len(task), ddp_world_size):
    conversation = task[idx]
    tokens = tokenizer.render_for_completion(conversation)
    prefix_length = len(tokens)

    generated_sequences, masks = engine.generate_batch(
        tokens, num_samples=1, max_tokens=args.max_new_tokens,
        temperature=args.temperature, top_k=args.top_k,
    )

    generated_tokens = generated_sequences[0][prefix_length:]
    generated_text = tokenizer.decode(generated_tokens)

    is_correct = task.evaluate(conversation, generated_text)

    question = conversation['messages'][0]['content']
    gt_parts = conversation['messages'][-1]['content']
    gt_last_text = gt_parts[-1]['text']
    gt_answer = extract_answer(gt_last_text)
    pred_answer = extract_answer(generated_text)

    gt_full_text = ''.join(
        p['text'] for p in gt_parts if p.get('type') == 'text'
    )
    gt_tool_calls = [
        p['text'] for p in gt_parts if p.get('type') == 'python'
    ]

    results.append({
        'idx': idx,
        'question': question,
        'gt_answer': gt_answer,
        'gt_full_text': gt_full_text,
        'gt_num_tool_calls': len(gt_tool_calls),
        'gt_tool_calls': gt_tool_calls,
        'pred_answer': pred_answer,
        'model_response': generated_text,
        'is_correct': bool(is_correct),
    })

    total += 1
    num_correct += int(is_correct)
    print(f"\rRank {ddp_rank} | {num_correct}/{total} "
          f"({100*num_correct/total:.1f}%)", end='', flush=True)

print()

if ddp:
    all_results = [None] * ddp_world_size
    dist.all_gather_object(all_results, results)
    if master_process:
        results = [r for rank_results in all_results for r in rank_results]
        results.sort(key=lambda x: x['idx'])

if master_process:
    total = len(results)
    correct = sum(r['is_correct'] for r in results)
    print0(f"Final: {correct}/{total} ({100*correct/total:.2f}%)")

    base_dir = get_base_dir()
    report_dir = os.path.join(base_dir, "report")
    os.makedirs(report_dir, exist_ok=True)

    filename = args.output_filename or f"gsm8k_detailed_{args.source}.json"
    output_path = os.path.join(report_dir, filename)

    with open(output_path, 'w') as f:
        json.dump({
            'source': args.source,
            'model_tag': args.model_tag,
            'total': total,
            'correct': correct,
            'accuracy': correct / total,
            'results': results,
        }, f, indent=2)

    print0(f"Saved {len(results)} detailed results to {output_path}")

compute_cleanup()
