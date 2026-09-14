"""Forward pass shape checks, no NaNs."""
import torch

from falldet.model import AttentionPooling, FallDetectionModel


def test_attention_pooling_shapes_and_weights_sum_to_one():
    pooling = AttentionPooling(input_dim=8, attention_dim=4)
    sequence = torch.randn(3, 10, 8)
    pooled, weights = pooling(sequence)
    assert pooled.shape == (3, 8)
    assert weights.shape == (3, 10)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3), atol=1e-5)


def test_model_forward_pass_shape():
    model = FallDetectionModel(input_dim=16, hidden_dim=32, num_layers=2, num_classes=2)
    x = torch.randn(4, 20, 16)
    logits = model(x)
    assert logits.shape == (4, 2)
    assert torch.isfinite(logits).all()


def test_model_forward_pass_no_nans_with_dropout_in_train_mode():
    model = FallDetectionModel(input_dim=16, hidden_dim=32, num_layers=2, dropout=0.5)
    model.train()
    x = torch.randn(8, 15, 16)
    logits = model(x)
    assert torch.isfinite(logits).all()


def test_model_single_layer_no_lstm_dropout_warning():
    import warnings

    model = FallDetectionModel(input_dim=8, hidden_dim=16, num_layers=1, dropout=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model(torch.randn(2, 5, 8))


def test_model_unidirectional_output_dim():
    model = FallDetectionModel(input_dim=8, hidden_dim=16, num_layers=1, bidirectional=False)
    logits = model(torch.randn(2, 5, 8))
    assert logits.shape == (2, 2)


def test_model_return_attention_weights():
    model = FallDetectionModel(input_dim=8, hidden_dim=16, num_layers=1)
    logits, weights = model(torch.randn(2, 5, 8), return_attention=True)
    assert logits.shape == (2, 2)
    assert weights.shape == (2, 5)


def test_model_from_config():
    config = {
        "model": {
            "hidden_dim": 32,
            "num_layers": 1,
            "bidirectional": True,
            "attention_dim": 16,
            "dropout": 0.1,
            "num_classes": 2,
        }
    }
    model = FallDetectionModel.from_config(config, input_dim=167)
    logits = model(torch.randn(2, 30, 167))
    assert logits.shape == (2, 2)


def test_model_backward_pass_produces_gradients():
    model = FallDetectionModel(input_dim=8, hidden_dim=16, num_layers=1)
    x = torch.randn(4, 5, 8)
    labels = torch.tensor([0, 1, 0, 1])
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
