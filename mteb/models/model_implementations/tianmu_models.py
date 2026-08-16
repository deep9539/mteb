from __future__ import annotations

import os
import sys
import logging
from typing import Any, TYPE_CHECKING
import torch
import numpy as np
from tqdm.auto import tqdm

from mteb.models.model_implementations.qwen3_vl_embedding_models import Qwen3VLEmbeddingWrapper
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.modality_collators import AudioCollator

if TYPE_CHECKING:
    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import PromptType, Array, BatchedInput
    from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class TianmuEmbUniWrapper(Qwen3VLEmbeddingWrapper):
    """MTEB Wrapper for the TianmuLab/Tianmu-Emb-Uni model.

    Tianmu-Emb-Uni is a unified multimodal embedding model combining:
    - Qwen3-VL-Embedding-8B for text, image, and video modalities.
    - Qwen2.5-Omni-7B audio tower for audio modality connected via trainable adapter/connector modules.

    This wrapper inherits from Qwen3VLEmbeddingWrapper to reuse the Qwen3-VL-Embedding
    modality-processing logic (text, image, video) out-of-the-box, while overriding
    the `encode` method to route audio batches through its custom audio tower and adapters.
    """

    def __init__(
        self,
        model_name: str = "TianmuLab/Tianmu-Emb-Uni",
        revision: str = "fc9106fc68156d2abde24cd046a8c989fa15fdc0",
        device: str | None = None,
        vl_model_name: str = "Qwen/Qwen3-VL-Embedding-8B",
        audio_model_path: str = "Qwen/Qwen2.5-Omni-7B",
        **kwargs: Any,
    ) -> None:
        # 1. Download only the small code files dynamically using huggingface_hub
        from huggingface_hub import hf_hub_download

        for filename in [
            "tianmu_model/__init__.py",
            "tianmu_model/modeling.py",
            "tianmu_model/adapter.py",
            "tianmu_model/prototype.py",
        ]:
            hf_hub_download(repo_id=model_name, revision=revision, filename=filename)

        # Download config.json and locate the snapshot directory
        config_path = hf_hub_download(repo_id=model_name, revision=revision, filename="config.json")
        repo_dir = os.path.dirname(config_path)

        # Add the repository path to sys.path to resolve imports of tianmu_model
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)

        from tianmu_model.modeling import Qwen3vlOmniEmbed

        # Avoid keyword argument collision in Qwen3VLEmbeddingWrapper by popping trust_remote_code
        super_kwargs = kwargs.copy()
        super_kwargs.pop("trust_remote_code", None)

        # 2. Initialize parent class to load Qwen3-VL-Embedding-8B SentenceTransformer
        super().__init__(
            model_name=vl_model_name,
            device=device,
            **super_kwargs,
        )

        # 3. Initialize the audio branch and projection head of Qwen3vlOmniEmbed
        # We pass vl_model_name=None to skip reloading the VL model, and then manually
        # set its vl_backbone to our already-loaded `self.model`.
        self.tianmu_model = Qwen3vlOmniEmbed(
            vl_model_name=None,
            audio_encoder_type="omni",
            audio_model_path=audio_model_path,
            freeze_vl=True,
            freeze_audio_encoder=True,
        )
        self.tianmu_model.vl_backbone = self.model
        self.tianmu_model.vl_backbone_config = self.model.config if hasattr(self.model, "config") else None

        # 4. Load the adapter, audio connector, projection, and prototype weights
        import json
        from transformers.modeling_utils import load_sharded_checkpoint

        # Download the shard index and all referenced shard files
        index_path = hf_hub_download(repo_id=model_name, revision=revision, filename="model.safetensors.index.json")
        with open(index_path) as f:
            index_data = json.load(f)

        # Download unique shard filenames
        shards = sorted(list(set(index_data["weight_map"].values())))
        for shard in shards:
            hf_hub_download(repo_id=model_name, revision=revision, filename=shard)

        # Load sharded state dict into tianmu_model (strict=False as some non-adapter parameters are frozen/unused)
        load_sharded_checkpoint(self.tianmu_model, repo_dir, strict=False)
        self.tianmu_model.to(self.device)
        self.tianmu_model.eval()

        # Load audio processor
        from transformers import AutoProcessor

        self.audio_processor = AutoProcessor.from_pretrained(audio_model_path, trust_remote_code=True)
        self.sampling_rate = self.audio_processor.feature_extractor.sampling_rate

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        # Determine if we are dealing with an audio dataset/batch
        features = inputs.dataset.features
        has_audio = "audio" in features

        if has_audio:
            # Set target sampling rate for audio collation
            inputs.collate_fn = AudioCollator(target_sampling_rate=self.sampling_rate)
            all_embeddings = []

            # Map dataset parser/name to target modality index for soft adapter fusion
            from tianmu_model.adapter import target_modality_from_parser

            target_mod_idx = target_modality_from_parser(task_metadata.name)
            target_modality_tensor = torch.tensor([target_mod_idx], dtype=torch.long, device=self.device)

            for batch in tqdm(inputs, desc="Encoding audio"):
                audios = batch.get("audio")
                # Handle inputs as list of raw audio arrays
                audio_arrays = [
                    a["array"] if isinstance(a, dict) and "array" in a else a
                    for a in audios
                ]
                # Preprocess audio waveforms with Qwen2.5-Omni processor
                processor_outputs = self.audio_processor(
                    audio=audio_arrays,
                    sampling_rate=self.sampling_rate,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)

                with torch.no_grad():
                    # Call encode_audio of the tianmu_model
                    embeddings = self.tianmu_model.encode_audio(
                        input_features=processor_outputs["input_features"],
                        feature_attention_mask=processor_outputs.get("feature_attention_mask"),
                        attention_mask=processor_outputs.get("attention_mask"),
                        target_modality=target_modality_tensor,
                    )
                all_embeddings.append(embeddings.cpu().float().numpy())
            return np.concatenate(all_embeddings, axis=0)

        else:
            # Route text, image, and video modalities through the parents' Qwen3-VL-Embedding encoder
            return super().encode(
                inputs,
                task_metadata=task_metadata,
                hf_split=hf_split,
                hf_subset=hf_subset,
                prompt_type=prompt_type,
                **kwargs,
            )


tianmu_emb_uni = ModelMeta(
    loader=TianmuEmbUniWrapper,
    name="TianmuLab/Tianmu-Emb-Uni",
    languages=["eng-Latn", "zho-Hans"],
    open_weights=True,
    revision="fc9106fc68156d2abde24cd046a8c989fa15fdc0",
    release_date="2026-08-10",
    modalities=["text", "image", "video", "audio"],
    n_parameters=8_964_318_448,
    n_embedding_parameters=622_329_856,
    memory_usage_mb=34000,
    embed_dim=3584,
    license="apache-2.0",
    max_tokens=32768,
    reference="https://huggingface.co/TianmuLab/Tianmu-Emb-Uni",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "Transformers", "safetensors"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    adapted_from="Qwen/Qwen3-VL-Embedding-8B",
    extra_requirements_groups=["multimodal-sbert"],
)
