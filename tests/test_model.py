import pathlib

import torch

from hex_game.ai.model import HexNet


def test_model_shapes():
    n = 11
    model = HexNet(n=n, n_res_block=2)
    batch_size = 4
    x = torch.randn(batch_size, 1, n, n)

    policy, value = model(x)

    assert policy.shape == (batch_size, n * n)
    assert value.shape == (batch_size, 1)


def test_model_save_load(tmp_path):
    n = 5
    model = HexNet(n=n, n_res_block=1)
    save_path = tmp_path / "model.pth"

    # Save weights
    torch.save(model.state_dict(), save_path)
    assert pathlib.Path(save_path).exists()

    # Load weights
    model2 = HexNet(n=n, n_res_block=1)
    model2.load_state_dict(torch.load(save_path))

    # Check if parameters are equal
    for p1, p2 in zip(model.parameters(), model2.parameters(), strict=True):
        assert torch.equal(p1, p2)
