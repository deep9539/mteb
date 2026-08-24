#!/usr/bin/env python3
"""Prepare the DynamicSuperb/EmotionalVoiceConversion_ESD dataset for MTEB AT2A retrieval.

This script loads the dataset from Hugging Face, structures it into MTEB retrieval format
with `queries`, `corpus`, and `qrels`, and then saves it locally or pushes it to Hugging Face.

Schema of output datasets:
- queries: `id`, `audio` (source), and `text` (instruction)
- corpus: `id` and `audio` (target)
- qrels: `query-id`, `corpus-id`, and binary `score` (all 1)

Usage:
  # Save to a local directory:
  DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib .venv/bin/python scripts/emotional_voice_conversion_esd/data_prep.py --output-dir data/emotional_voice_conversion_esd_at2a

  # Push to Hugging Face:
  DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib HF_TOKEN=... .venv/bin/python scripts/emotional_voice_conversion_esd/data_prep.py --repo-id mteb/emotional-voice-conversion-esd --push
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from datasets import Audio, Dataset, DatasetDict, Features, Value, load_dataset
from huggingface_hub import HfApi, create_repo, dataset_info


def get_filename_stem(path_str: str) -> str:
    """Helper to extract filename stem (without extension) from a path string."""
    return Path(path_str).stem


def get_path_from_audio(audio_obj: Any) -> str | None:
    """Extracts path/filename from an audio object (either dict or AudioDecoder)."""
    if hasattr(audio_obj, "_hf_encoded") and audio_obj._hf_encoded:
        return audio_obj._hf_encoded.get("path")
    if isinstance(audio_obj, dict):
        return audio_obj.get("path")
    return None


def get_audio_dict(audio_obj: Any) -> dict[str, Any] | Any:
    """Extracts the path and bytes payload as a dict from an audio object."""
    if hasattr(audio_obj, "_hf_encoded") and audio_obj._hf_encoded:
        return audio_obj._hf_encoded
    if isinstance(audio_obj, dict):
        return audio_obj
    return audio_obj


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare DynamicSuperb/EmotionalVoiceConversion_ESD for MTEB retrieval"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/emotional_voice_conversion_esd_at2a"),
        help="Local directory to save the structured dataset splits.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Hugging Face repository ID to push the dataset to.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Whether to push the prepared dataset directly to Hugging Face Hub.",
    )
    args = parser.parse_args()

    print("Loading source dataset from Hugging Face...")
    # Load dataset. The caller should set DYLD_FALLBACK_LIBRARY_PATH if torchcodec is used for decoding.
    source_ds_dict = load_dataset("DynamicSuperb/EmotionalVoiceConversion_ESD")
    if "test" not in source_ds_dict:
        raise ValueError(
            "Expected 'test' split in the source dataset, but it was not found."
        )
    test_ds = source_ds_dict["test"]
    print(f"Loaded {len(test_ds)} test rows.")

    # We will build queries, corpus, and qrels.
    # 1. Queries
    # Each query combines the source audio (unedited) with the text instruction.
    query_rows = {
        "id": [],
        "audio": [],
        "text": [],
    }

    # 2. Corpus
    # Collected from unique target audios in 'label', 'wrong_label_1', 'wrong_label_2', 'wrong_label_3'
    # We will key them by their unique filename stem.
    corpus_dict: dict[str, dict[str, Any]] = {}

    # 3. Qrels
    # Binary mapping of query id -> relevant corpus id.
    qrel_rows = {
        "query-id": [],
        "corpus-id": [],
        "score": [],
    }

    # Helper function to track unique target audios
    def add_to_corpus(audio_obj: Any) -> str:
        path = get_path_from_audio(audio_obj)
        if not path:
            raise ValueError(f"Missing file path in audio data: {audio_obj}")
        stem = get_filename_stem(path)
        corpus_id = f"t-{stem}"
        if corpus_id not in corpus_dict:
            audio_dict = get_audio_dict(audio_obj)
            corpus_dict[corpus_id] = {
                "bytes": audio_dict.get("bytes"),
                "path": path,
            }
        return corpus_id

    # Iterate over the test dataset to populate queries, corpus, and qrels
    for idx in range(len(test_ds)):
        row = test_ds[idx]
        
        # Source query audio and instruction
        source_audio_obj = row["audio"]
        source_path = get_path_from_audio(source_audio_obj)
        if not source_path:
            # Fallback to column named 'file' if path is missing
            source_path = row.get("file")
            if source_path and not source_path.endswith(".wav"):
                source_path = f"{source_path}.wav"
        
        if not source_path:
            raise ValueError(f"Could not find path for source audio at row {idx}")
            
        source_stem = get_filename_stem(source_path)
        query_id = f"q-{source_stem}"
        instruction = row["instruction"]

        source_audio_dict = get_audio_dict(source_audio_obj)
        query_rows["id"].append(query_id)
        query_rows["audio"].append({
            "bytes": source_audio_dict.get("bytes") if isinstance(source_audio_dict, dict) else getattr(source_audio_dict, "bytes", None),
            "path": source_path,
        })
        query_rows["text"].append(instruction)

        # Retrieve and add correct/incorrect targets to corpus
        correct_audio = row["label"]
        wrong1_audio = row["wrong_label_1"]
        wrong2_audio = row["wrong_label_2"]
        wrong3_audio = row["wrong_label_3"]

        # Ensure all of them are added to the corpus
        correct_corpus_id = add_to_corpus(correct_audio)
        _ = add_to_corpus(wrong1_audio)
        _ = add_to_corpus(wrong2_audio)
        _ = add_to_corpus(wrong3_audio)

        # Add the positive relation to qrels
        qrel_rows["query-id"].append(query_id)
        qrel_rows["corpus-id"].append(correct_corpus_id)
        qrel_rows["score"].append(1)

    print(f"Extraction complete.")
    print(f"  Total queries extracted: {len(query_rows['id'])}")
    print(f"  Total unique corpus items: {len(corpus_dict)}")
    print(f"  Total qrels: {len(qrel_rows['query-id'])}")

    # Format corpus dict into rows
    corpus_rows = {
        "id": [],
        "audio": [],
    }
    for corpus_id, audio_data in sorted(corpus_dict.items()):
        corpus_rows["id"].append(corpus_id)
        corpus_rows["audio"].append(audio_data)

    # Construct Hugging Face datasets with the proper features and cast columns to Audio()
    print("Constructing Arrow datasets and casting audio columns...")
    queries_ds = Dataset.from_dict(query_rows).cast_column("audio", Audio())
    corpus_ds = Dataset.from_dict(corpus_rows).cast_column("audio", Audio())
    qrels_ds = Dataset.from_dict(
        qrel_rows,
        features=Features(
            {
                "query-id": Value("string"),
                "corpus-id": Value("string"),
                "score": Value("int32"),
            }
        ),
    )

    dataset_dict = {
        "queries": DatasetDict({"test": queries_ds}),
        "corpus": DatasetDict({"test": corpus_ds}),
        "qrels": DatasetDict({"test": qrels_ds}),
    }

    # Save locally or push to the hub
    if args.push:
        if not args.repo_id:
            raise ValueError("--repo-id is required when --push is specified.")
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("Set HF_TOKEN environment variable to push to the Hugging Face Hub.")
            
        print(f"Creating repository: {args.repo_id}...")
        create_repo(args.repo_id, repo_type="dataset", token=token, exist_ok=True)
        
        for config_name, d_dict in dataset_dict.items():
            print(f"Pushing {config_name} subset to {args.repo_id}...")
            d_dict.push_to_hub(
                args.repo_id,
                config_name=config_name,
                token=token,
                max_shard_size="500MB",
            )
            
        print(f"Successfully pushed dataset to {args.repo_id}!")
        try:
            info = dataset_info(args.repo_id, token=token)
            print(f"Repository revision SHA: {info.sha}")
        except Exception as e:
            print(f"Could not fetch repository revision: {e}")
    else:
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving prepared datasets locally to {output_dir}...")
        for config_name, d_dict in dataset_dict.items():
            d_dict.save_to_disk(output_dir / config_name)
        print("Successfully saved structured datasets locally!")


if __name__ == "__main__":
    main()
