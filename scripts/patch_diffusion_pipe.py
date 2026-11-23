"""Patch pinned tdrussell/diffusion-pipe after checkout.

1) Honor skip_final_checkpoint in the train TOML. Upstream always writes a
   final Deepspeed checkpoint (optimizer + full training state). On a
   Serverless worker that file is large, unused (we only need the LoRA), and
   can hang or fill the ephemeral disk.

2) Python 3.11 compatibility for utils/cache.py: sqlite3.connect(...,
   autocommit=False) is 3.12+ and raises TypeError on this image's 3.11.
"""
from __future__ import annotations

import sys
from pathlib import Path

PIPE = Path("/opt/diffusion-pipe/train.py")
CACHE = Path("/opt/diffusion-pipe/utils/cache.py")
MARKER_CKPT = "skip_final_checkpoint"
MARKER_SQLITE = "py311-sqlite-autocommit"


def _patch_train() -> int:
    if not PIPE.is_file():
        print(f"patch: {PIPE} not found", file=sys.stderr)
        return 1
    text = PIPE.read_text(encoding="utf-8")
    if MARKER_CKPT in text:
        print("patch: skip_final_checkpoint already applied")
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


def _patch_cache() -> int:
    if not CACHE.is_file():
        print(f"patch: {CACHE} not found", file=sys.stderr)
        return 1
    text = CACHE.read_text(encoding="utf-8")
    if MARKER_SQLITE in text:
        print("patch: sqlite autocommit already applied")
        return 0
    old = "        self.con = sqlite3.connect(self.metadata_db, autocommit=False)\n"
    new = (
        "        # py311-sqlite-autocommit: connect(autocommit=...) is Python 3.12+.\n"
        "        # isolation_level default on 3.11 is the equivalent of autocommit=False.\n"
        "        _sql_kw = {\"autocommit\": False} if hasattr(sqlite3, \"LEGACY_TRANSACTION_CONTROL\") else {}\n"
        "        self.con = sqlite3.connect(self.metadata_db, **_sql_kw)\n"
    )
    if old not in text:
        print("patch: expected cache.py sqlite connect line not found", file=sys.stderr)
        return 1
    CACHE.write_text(text.replace(old, new, 1), encoding="utf-8")
    print("patch: applied sqlite 3.11 compat to", CACHE)
    return 0


def main() -> int:
    rc = _patch_train()
    rc2 = _patch_cache()
    return rc or rc2


if __name__ == "__main__":
    raise SystemExit(main())
