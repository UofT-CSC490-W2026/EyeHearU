"""
Download teknium/OpenHermes-2.5 from HuggingFace and convert to
nanochat CustomJSON JSONL format (one JSON array of messages per line).

Usage:
    python -m scripts.convert_openhermes /path/to/output.jsonl
"""

import json
import sys
from datasets import load_dataset

def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "openhermes_2.5.jsonl"
    print(f"Loading teknium/OpenHermes-2.5 from HuggingFace...")
    ds = load_dataset("teknium/OpenHermes-2.5", split="train")

    role_map = {"human": "user", "gpt": "assistant"}
    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            messages = []
            for turn in ex["conversations"]:
                role = role_map.get(turn["from"])
                if role is None:
                    if turn["from"] == "system" and messages:
                        messages[-1]["content"] = turn["value"] + "\n\n" + messages[-1]["content"]
                    continue
                messages.append({"role": role, "content": turn["value"]})
            if len(messages) >= 2:
                f.write(json.dumps(messages, ensure_ascii=False) + "\n")
                written += 1
            if (i + 1) % 100_000 == 0:
                print(f"  processed {i + 1:,} rows, written {written:,}")

    print(f"Done. Wrote {written:,} conversations to {out_path}")

if __name__ == "__main__":
    main()
