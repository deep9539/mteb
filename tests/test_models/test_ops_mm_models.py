from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import torch

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

    # Test embed_batch
    texts = ["Hello", "World"]
    embs = wrapper.embed_batch(texts=texts)
    assert isinstance(embs, torch.Tensor)
    assert embs.shape == (2, 128)

    # Test get_text_embeddings
    text_embeddings = wrapper.get_text_embeddings(texts)
    assert isinstance(text_embeddings, np.ndarray)
    assert text_embeddings.shape == (2, 128)
