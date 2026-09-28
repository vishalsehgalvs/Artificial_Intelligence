"""Tiny 1D DDPM: learn to remove noise from numbers near -2 and +2."""

import torch


def run():
    torch.manual_seed(42)
    steps = 20
    beta = torch.linspace(0.001, 0.04, steps)
    alpha = 1 - beta
    retained = torch.cumprod(alpha, dim=0)
    clean = torch.where(torch.rand(2048, 1) < 0.5, -2.0, 2.0)
    clean += 0.15 * torch.randn_like(clean)
    network = torch.nn.Sequential(torch.nn.Linear(2, 32), torch.nn.SiLU(), torch.nn.Linear(32, 32), torch.nn.SiLU(), torch.nn.Linear(32, 1))
    optimizer = torch.optim.Adam(network.parameters(), lr=0.003)
    losses = []
    for iteration in range(400):
        indices = torch.randint(0, len(clean), (128,))
        times = torch.randint(0, steps, (128,))
        noise = torch.randn(128, 1)
        keep = retained[times, None]
        noisy = keep.sqrt() * clean[indices] + (1 - keep).sqrt() * noise
        inputs = torch.cat((noisy, times[:, None].float() / steps), dim=1)
        loss = torch.nn.functional.mse_loss(network(inputs), noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert torch.isfinite(loss) and sum(losses[-50:]) / 50 < sum(losses[:50]) / 50
    print("Noise prediction loss, first/last 50:", round(sum(losses[:50]) / 50, 3), round(sum(losses[-50:]) / 50, 3))
    samples = torch.randn(32, 1)
    network.eval()
    with torch.no_grad():
        for time in reversed(range(steps)):
            inputs = torch.cat((samples, torch.full_like(samples, time / steps)), dim=1)
            samples = (samples - beta[time] / (1 - retained[time]).sqrt() * network(inputs)) / alpha[time].sqrt()
            if time:
                samples += beta[time].sqrt() * torch.randn_like(samples)
    assert samples.shape == (32, 1) and torch.isfinite(samples).all()
    print("First ten generated numbers:", samples[:10, 0].tolist())


if __name__ == "__main__":
    run()