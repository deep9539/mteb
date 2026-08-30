from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import torch
from PIL import Image

from mteb.models.model_implementations.ops_mm_models import (
    OpsMMEmbeddingWrapper,
)


@patch("transformers.AutoModelForImageTextToText.from_pretrained")
@patch("transformers.AutoProcessor.from_pretrained")
def test_ops_mm_embedding_wrapper(mock_processor_class, mock_model_class):
    # Mocking AutoModel and AutoProcessor
    mock_model = MagicMock()
    mock_model_class.return_value = mock_model
    mock_model.to.return_value = mock_model

    mock_processor = MagicMock()
    mock_processor_class.return_value = mock_processor

    # We mock the return dict of the model's forward
    mock_output = MagicMock()
    # hidden_states should return a list, the last of which has the actual shape (batch_size, seq_len, hidden_size)
    mock_hidden_state = torch.randn(2, 10, 128)
    mock_output.hidden_states = [mock_hidden_state]
    mock_model.return_value = mock_output

    # Create wrapper with mock model and processor
    wrapper = OpsMMEmbeddingWrapper(
        model_name="OpenSearch-AI/Ops-MM-embedding-v1-2B",
        device="cpu",
        torch_dtype=torch.float32,
    )

    # Setup mock return for processor
    mock_processor.return_value = {
        "input_ids": torch.ones(2, 10, dtype=torch.long),
        "attention_mask": torch.ones(2, 10, dtype=torch.long),
    }

    # 1. Test embed_batch with text-only
    texts = ["Hello", "World"]
    embs = wrapper.embed_batch(texts=texts)
    assert isinstance(embs, torch.Tensor)
    assert embs.shape == (2, 128)

    # 2. Test embed_batch with image-only
    img = Image.new("RGB", (100, 100), color="red")
    embs_img = wrapper.embed_batch(images=[img, img])
    assert isinstance(embs_img, torch.Tensor)

    # 3. Test embed_batch with 4D video tensor (shape: [T, C, H, W] - e.g. 3 frames of 3x100x100)
    video_tensor = torch.zeros((3, 3, 100, 100))
    embs_video = wrapper.embed_batch(images=[video_tensor, video_tensor])
    assert isinstance(embs_video, torch.Tensor)

    # 4. Test embed_batch with mixed images and text
    embs_mixed = wrapper.embed_batch(texts=["Describe this", "Describe that"], images=[img, img])
    assert isinstance(embs_mixed, torch.Tensor)
