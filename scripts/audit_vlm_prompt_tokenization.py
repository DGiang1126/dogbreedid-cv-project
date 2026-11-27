"""Reproduce the tokenizer-only preview for the pinned Task 08.04 prompt banks."""
from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path

from src.vlm import contracts as c
from src.vlm.knowledge_base import audit_prompt_bank, load_prompt_bank


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "knowledge" / "vlm"
TOKEN_LIMIT = 64
BANKS = ("name_prompt_bank.json", "morphology_prompt_bank.json")
PREVIEW_MODELS = (
    ("siglip2-tokenizer-preview", "google/siglip2-base-patch16-224", "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"),
    ("fgclip2-tokenizer-preview", "qihoo360/fg-clip2-base", "430fbc8a912c86fd4de601381b6245a0edab22f0"),
)
SHARED_TOKENIZER_HASHES = {
    "special_tokens_map.json": "baec30ea10906f16adb8c18af7a34023002c1746542612b8b41c9f09e1351351",
    "tokenizer.model": "61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2",
}
MODEL_TOKENIZER_HASHES = {
    "siglip2-tokenizer-preview": {
        "tokenizer.json": "cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322",
        "tokenizer_config.json": "14afe629fe4959b9e0d51e1852b8d9f7ad074f90a1a7125a4fcdd17f06e78fc8",
    },
    "fgclip2-tokenizer-preview": {
        "tokenizer.json": "58a1696e79c9d97937389ed116f552a15c84811d7b8023918b86f4bc5775b1b0",
        "tokenizer_config.json": "607049b4766b302a66e31906a6b4729e01f623597c82262fb52e552f0d4f7fb4",
    },
}


def load_banks() -> tuple[c.PromptSnapshot, ...]:
    mapping_raw = (ROOT / "data" / "metadata" / "class_mapping.json").read_bytes()
    mapping = c.decode_canonical_mapping(mapping_raw, expected_sha256=sha256(mapping_raw).hexdigest())
    name_raw, morphology_raw = ((DATA / name).read_bytes() for name in BANKS)
    name = load_prompt_bank(name_raw, mapping=mapping, expected_sha256=sha256(name_raw).hexdigest())
    # The morphology draft can be tokenized as text while runtime usage awaits source approval.
    morphology = c.decode_record(morphology_raw, c.PromptBank)
    kb_raw = (DATA / "breed_kb.json").read_bytes()
    c.verify_sha256(kb_raw, morphology.kb_sha256)
    if morphology.mapping_sha256 != mapping.mapping_sha256 or any(
        entry.canonical_name != mapping.names[entry.class_id] for entry in morphology.entries
    ):
        raise c.ContractError("morphology mapping mismatch", c.ErrorCode.MAPPING_MISMATCH)
    return name, c.PromptSnapshot(data=morphology, prompt_bank_sha256=sha256(morphology_raw).hexdigest())


def main() -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--siglip2-dir", type=Path, required=True)
    parser.add_argument("--fgclip2-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    banks = load_banks()
    records = []
    for directory, (profile_id, model_id, revision) in zip(
        (args.siglip2_dir, args.fgclip2_dir), PREVIEW_MODELS,
    ):
        path = directory.resolve()
        hashes = SHARED_TOKENIZER_HASHES | MODEL_TOKENIZER_HASHES[profile_id]
        for filename, digest in hashes.items():
            c.verify_sha256((path / filename).read_bytes(), digest)

        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        if not tokenizer.is_fast:
            raise ValueError("expected the native fast tokenizer")
        print(f"{profile_id}: {type(tokenizer).__name__}, revision {revision}")
        profile = c.ApprovedProfile(
            profile_id=profile_id,
            model=c.ModelRef(model_id=model_id, model_revision=revision, processor_revision=revision),
            config_sha256=hashes["tokenizer_config.json"], device="cpu", dtype="tokenizer",
            limits={"prompt_token_limit": TOKEN_LIMIT}, tokenizer_revision=revision,
            approval_ref="preview-awaits-project-profile-approval",
        )

        def count_untruncated(prompt: str) -> int:
            text = prompt.lower()
            full = tokenizer(text, add_special_tokens=True, padding=False, truncation=False)["input_ids"]
            capped = tokenizer(text, add_special_tokens=True, padding=False, truncation=True, max_length=TOKEN_LIMIT)
            if len(full) <= TOKEN_LIMIT and full != capped["input_ids"]:
                raise ValueError("bounded tokenizer altered a prompt within the token limit")
            return len(full)

        for bank in banks:
            audit = audit_prompt_bank(bank, profile=profile, count_untruncated_tokens=count_untruncated)
            counts = [entry.token_count for entry in audit.entries]
            print(f"  {bank.data.bank_version}: {len(counts)} prompts, {min(counts)}-{max(counts)} tokens")
            records.append(audit)
    args.output.write_bytes(c.encode_record(c.PromptTokenAudit(audits=tuple(records))) + b"\n")


if __name__ == "__main__":
    main()
