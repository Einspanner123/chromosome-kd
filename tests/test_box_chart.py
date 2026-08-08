import torch

from ldmdet.diffusion.box_chart import ValidBoxChart


def random_valid_boxes(n=4096, dtype=torch.float64):
    gaps_x = torch.distributions.Dirichlet(torch.ones(3, dtype=dtype)).sample((n,))
    gaps_y = torch.distributions.Dirichlet(torch.ones(3, dtype=dtype)).sample((n,))
    return torch.stack(
        (gaps_x[:, 0], gaps_y[:, 0], gaps_x[:, :2].sum(-1), gaps_y[:, :2].sum(-1)),
        dim=-1,
    )


def test_decode_is_strictly_valid_for_finite_latents():
    chart = ValidBoxChart().double()
    latent = torch.randn(20000, 4, dtype=torch.float64) * 8
    boxes = chart.decode(latent)
    assert torch.all(boxes[:, 0] >= 0)
    assert torch.all(boxes[:, 1] >= 0)
    assert torch.all(boxes[:, 2] <= 1)
    assert torch.all(boxes[:, 3] <= 1)
    assert torch.all(boxes[:, 2] > boxes[:, 0])
    assert torch.all(boxes[:, 3] > boxes[:, 1])


def test_interior_round_trip():
    chart = ValidBoxChart(eps=1e-12).double()
    boxes = random_valid_boxes()
    reconstructed = chart.from_chart(chart.to_chart(boxes))
    torch.testing.assert_close(reconstructed, boxes, atol=1e-10, rtol=1e-10)


def test_fit_whitens_first_two_moments():
    chart = ValidBoxChart().double()
    boxes = random_valid_boxes(20000)
    chart.fit(boxes)
    latent = chart.encode(boxes)
    mean = latent.mean(0)
    centered = latent - mean
    covariance = centered.T @ centered / latent.shape[0]
    torch.testing.assert_close(mean, torch.zeros_like(mean), atol=1e-10, rtol=0)
    torch.testing.assert_close(covariance, torch.eye(4, dtype=latent.dtype), atol=1e-8, rtol=1e-8)


def test_decode_gradient_is_finite_away_from_float_saturation():
    chart = ValidBoxChart().double()
    latent = torch.randn(128, 4, dtype=torch.float64, requires_grad=True)
    boxes = chart.decode(latent)
    boxes.square().sum().backward()
    assert torch.isfinite(latent.grad).all()
    assert (latent.grad.abs().sum(dim=-1) > 0).all()
