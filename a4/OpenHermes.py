from datasets import load_dataset
import json
import os

out_path = "openhermes_2.5.jsonl"
ds = load_dataset("teknium/OpenHermes-2.5", split="train")

with open(out_path, "w", encoding="utf-8") as f:
    written = 0
    for i, ex in enumerate(ds):
        conv = ex["conversations"]
        messages = []
        for turn in conv:
            role = turn["from"]
            if role == "human":
                r = "user"
            elif role == "gpt":
                r = "assistant"
            elif role == "system":
                if messages and messages[-1]["role"] == "user":
                    messages[-1]["content"] = turn["value"] + "\n\n" + messages[-1]["content"]
                else:
                    messages.append({"role": "user", "content": turn["value"]})
                continue
            else:
                continue
            messages.append({"role": r, "content": turn["value"]})
        if len(messages) >= 2:
            f.write(json.dumps(messages, ensure_ascii=False) + "\n")
            written += 1
        if (i + 1) % 50000 == 0:
            print(f"  processed {i + 1:,} rows, written {written:,}")
print(f"Done. Wrote {written:,} conversations to {out_path}")