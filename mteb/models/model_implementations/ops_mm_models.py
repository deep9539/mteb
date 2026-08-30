from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from tqdm.auto import tqdm

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.model_implementations.ops_colqwen3_models import multilingual_langs

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType

logger = logging.getLogger(__name__)


class OpsMMEmbeddingWrapper(AbsEncoder):
    """MTEB Wrapper for OpenSearch-AI/Ops-MM-embedding-v1 models."""

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        torch_dtype: torch.dtype | str | None = None,
        attn_implementation: str | None = None,
        max_length: int | None = None,
        fps: float | None = 2.0,
        max_frames: int | None = 64,
        num_frames: int | None = None,
        **kwargs: Any,
    ):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from transformers.utils.import_utils import is_flash_attn_2_available

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.max_length = max_length
        self.fps = fps
        self.max_frames = max_frames
        self.num_frames = num_frames

        if attn_implementation is None and is_flash_attn_2_available():
            attn_implementation = "flash_attention_2"

        if torch_dtype is None:
            self.torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        elif isinstance(torch_dtype, str):
            self.torch_dtype = getattr(torch, torch_dtype)
        else:
            self.torch_dtype = torch_dtype

        trust_remote_code = kwargs.pop("trust_remote_code", True)

        self.base_model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
            **kwargs,
        ).to(self.device)

        self.base_model.eval()

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=trust_remote_code,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        )
        self.processor.tokenizer.padding_side = "left"

    def encode_input(self, inputs: dict[str, Any]) -> torch.Tensor:
        hidden_states = self.base_model(**inputs, return_dict=True, output_hidden_states=True)
        hidden_states = hidden_states.hidden_states[-1]
        pooled_output = self._pooling(hidden_states)
        return pooled_output

    def _pooling(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        batch_size = last_hidden_state.shape[0]
        reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), -1, :]
        reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps

    def embed_batch(
        self,
        texts: list[str] | None = None,
        images: list[Any] | None = None,
        instruction: str | None = None,
    ) -> torch.Tensor:
        if texts is None and images is None:
            raise ValueError("Either texts or images must be provided")

        batch_size = len(texts) if texts is not None else len(images)  # type: ignore
        inst = instruction or "You are a helpful assistant."

        input_texts, input_images = [], []
        for i in range(batch_size):
            text = texts[i] if texts is not None else None
            image = images[i] if images is not None else None

            input_str = ""
            if image is not None:
                if isinstance(image, torch.Tensor) and image.ndim == 4:
                    import torchvision.transforms.functional as F
                    processed_image = [F.to_pil_image(frame) for frame in image]
                else:
                    processed_image = [image] if not isinstance(image, list) else image
                input_str += "<|vision_start|><|image_pad|><|vision_end|>" * len(processed_image)
                input_images.append(processed_image)
            else:
                input_images.append(None)

            if text is not None:
                input_str += text

            msg = f"<|im_start|>system\n{inst}<|im_end|>\n<|im_start|>user\n{input_str}<|im_end|>\n<|im_start|>assistant\n<|endoftext|>"
            input_texts.append(msg)

        processed_images = input_images if any(img is not None for img in input_images) else None

        processor_kwargs = {
            "text": input_texts,
            "padding": True,
            "truncation": True,
            "max_length": self.max_length,
            "return_tensors": "pt",
        }
        if processed_images is not None:
            normalized_images = [img if img is not None else [] for img in processed_images]
            if not all(len(img) == 0 for img in normalized_images):
                processor_kwargs["images"] = normalized_images

        inputs = self.processor(**processor_kwargs)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            embeddings = self.encode_input(inputs)

        return embeddings

    def get_text_embeddings(
        self,
        texts: DataLoader[BatchedInput] | list[str],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> Array:
        instruction = kwargs.get("instruction")
        if hasattr(texts, "__iter__") and not isinstance(texts, list):
            all_embeddings = []
            for batch in tqdm(texts, disable=not show_progress_bar, desc="Encoding texts"):
                batch_texts = batch["text"]
                emb = self.embed_batch(texts=batch_texts, instruction=instruction)
                all_embeddings.append(emb.cpu().to(torch.float32))
            return torch.cat(all_embeddings, dim=0).numpy()
        else:
            emb = self.embed_batch(texts=texts, instruction=instruction)
            return emb.cpu().to(torch.float32).numpy()

    def get_image_embeddings(
        self,
        images: DataLoader[BatchedInput] | list[Any],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> Array:
        instruction = kwargs.get("instruction")
        if hasattr(images, "__iter__") and not isinstance(images, list):
            all_embeddings = []
            for batch in tqdm(images, disable=not show_progress_bar, desc="Encoding images"):
                batch_images = batch["image"]
                emb = self.embed_batch(images=batch_images, instruction=instruction)
                all_embeddings.append(emb.cpu().to(torch.float32))
            return torch.cat(all_embeddings, dim=0).numpy()
        else:
            emb = self.embed_batch(images=images, instruction=instruction)
            return emb.cpu().to(torch.float32).numpy()

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
        instruction = self.get_task_instruction(task_metadata, prompt_type)

        features = inputs.dataset.features
        has_text = "text" in features
        has_image = "image" in features
        has_video = "video" in features

        if not (has_text or has_image or has_video):
            raise ValueError("No text, image, or video features found in inputs.")

        if has_video:
            from mteb.models.modality_collators import VideoCollator
            inputs.collate_fn = VideoCollator(
                target_sampling_rate=16000,
                fps=self.fps,
                max_frames=self.max_frames,
                num_frames=self.num_frames,
            )

        all_embeddings = []
        with torch.no_grad():
            for batch in inputs:
                texts = batch["text"] if has_text else None
                images = batch["image"] if has_image else (batch["video"] if has_video else None)
                emb = self.embed_batch(texts=texts, images=images, instruction=instruction)
                all_embeddings.append(emb.cpu().to(torch.float32))

        return torch.cat(all_embeddings, dim=0).numpy()


OPS_MM_EMBEDDING_CITATION = """
@misc{ops_mm_embedding_v1,
  author       = {OpenSearch-AI},
  title        = {Ops-MM-embedding-v1: State-of-the-Art Multimodal Embedding Models},
  year         = {2025},
  url          = {https://huggingface.co/OpenSearch-AI/Ops-MM-embedding-v1-2B},
}
"""

ops_mm_embedding_v1_2b = ModelMeta(
    loader=OpsMMEmbeddingWrapper,
    name="OpenSearch-AI/Ops-MM-embedding-v1-2B",
    model_type=["dense"],
    languages=multilingual_langs,
    open_weights=True,
    revision="69c23ac5595bb7fb3a9959df96fe155b3650c01e",
    release_date="2025-07-03",
    modalities=["image", "text", "video"],
    n_parameters=2_208_985_600,
    n_embedding_parameters=233_373_696,
    memory_usage_mb=4418,
    embed_dim=1536,
    license="apache-2.0",
    max_tokens=32768,
    reference="https://huggingface.co/OpenSearch-AI/Ops-MM-embedding-v1-2B",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["PyTorch", "Transformers", "safetensors"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    citation=OPS_MM_EMBEDDING_CITATION,
)

ops_mm_embedding_v1_7b = ModelMeta(
    loader=OpsMMEmbeddingWrapper,
    name="OpenSearch-AI/Ops-MM-embedding-v1-7B",
    model_type=["dense"],
    languages=multilingual_langs,
    open_weights=True,
    revision="e9179f601f9bf21d2fd20136c11eb4ae2ea42859",
    release_date="2025-07-03",
    modalities=["image", "text", "video"],
    n_parameters=8_291_375_616,
    n_embedding_parameters=544_997_376,
    memory_usage_mb=16583,
    embed_dim=3584,
    license="apache-2.0",
    max_tokens=32768,
    reference="https://huggingface.co/OpenSearch-AI/Ops-MM-embedding-v1-7B",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["PyTorch", "Transformers", "safetensors"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    citation=OPS_MM_EMBEDDING_CITATION,
)
