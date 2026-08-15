import os
import tiktoken
import numpy as np
from datasets import load_dataset


# ============================================================
# CONFIG
# ============================================================

TARGET_MB = 95
MAX_MB = 100

TARGET_BYTES = TARGET_MB * 1024 * 1024
MAX_BYTES = MAX_MB * 1024 * 1024

OUTPUT_DIR = os.path.dirname(__file__)


# ============================================================
# DOWNLOAD SIMPLESTORIES
# ============================================================

print("Loading SimpleStories in streaming mode...")

dataset = load_dataset(
    "SimpleStories/SimpleStories",
    split="train",
    streaming=True,
)

print("Dataset stream ready.")

parts = []
total_bytes = 0
story_count = 0


# ============================================================
# BUILD ~95 MiB TEXT
# ============================================================

for example in dataset:

    story = example.get("story")

    if story is None:
        continue

    story = str(story).strip()

    if not story:
        continue

    # Just concatenate stories.
    # No EOT token.
    text = story + "\n"

    story_bytes = len(
        text.encode("utf-8")
    )

    # Never exceed 100 MiB.
    if total_bytes + story_bytes > MAX_BYTES:
        break

    parts.append(text)

    total_bytes += story_bytes
    story_count += 1

    if story_count % 1000 == 0:
        print(
            f"stories: {story_count:,} | "
            f"size: {total_bytes / (1024 * 1024):.2f} MiB"
        )

    if total_bytes >= TARGET_BYTES:
        break


# ============================================================
# SAME INPUT THAT NORMAL NANOGPT WOULD READ
# ============================================================

data = "".join(parts)

print()
print(
    f"Stories: {story_count:,}"
)

print(
    f"Dataset size: "
    f"{len(data.encode('utf-8')) / (1024 * 1024):.2f} MiB"
)


# ============================================================
# SAME SPLIT AS STANDARD NANOGPT
# ============================================================

n = len(data)

train_data = data[:int(n * 0.9)]
val_data = data[int(n * 0.9):]


# ============================================================
# SAME GPT-2 TOKENIZER
# ============================================================

enc = tiktoken.get_encoding("gpt2")

train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)

print(
    f"train has {len(train_ids):,} tokens"
)

print(
    f"val has {len(val_ids):,} tokens"
)


# ============================================================
# SAME UINT16 EXPORT AS STANDARD NANOGPT
# ============================================================

train_ids = np.array(
    train_ids,
    dtype=np.uint16
)

val_ids = np.array(
    val_ids,
    dtype=np.uint16
)

train_ids.tofile(
    os.path.join(
        OUTPUT_DIR,
        "train.bin"
    )
)

val_ids.tofile(
    os.path.join(
        OUTPUT_DIR,
        "val.bin"
    )
)

print()
print("Created:")
print(
    os.path.join(
        OUTPUT_DIR,
        "train.bin"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "val.bin"
    )
)
