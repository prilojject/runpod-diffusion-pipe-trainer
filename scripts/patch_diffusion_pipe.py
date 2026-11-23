"""Patch pinned tdrussell/diffusion-pipe after checkout.

The upstream always writes a final Deepspeed checkpoint (optimizer + full
training state). On a Serverless worker that file is large, unused (we only
need the LoRA), and can hang or fill the ephemeral disk. Honor
skip_final_checkpoint in the train TOML.
"""
from __future__ import annotations

import sys
from pathlib import Path

PIPE = Path("/opt/diffusion-pipe/train.py")
MARKER = "skip_final_checkpoint"


def main() -> int:
    if not PIPE.is_file():
        print(f"patch: {PIPE} not found", file=sys.stderr)
        return 1
    text = PIPE.read_text(encoding="utf-8")
    if MARKER in text:
        print("patch: already applied")
        return 0
    old = """    # Save final training state checkpoint and model, unless we just saved them.
    if not checkpointed:
        saver.save_checkpoint(step, examples)
    if not saved:
        saver.save_model(final_model_name)
"""
    new = """    # Save final training state checkpoint and model, unless we just saved them.
    # skip_final_checkpoint: this worker only needs the LoRA; Deepspeed
    # checkpoints are large and can hang or fill ephemeral Serverless disks.
    if not checkpointed and not config.get('skip_final_checkpoint', False):
        saver.save_checkpoint(step, examples)
    if not saved:
        saver.save_model(final_model_name)
"""
    if old not in text:
        print("patch: expected train.py block not found", file=sys.stderr)
        return 1
    PIPE.write_text(text.replace(old, new, 1), encoding="utf-8")
    print("patch: applied skip_final_checkpoint to", PIPE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
