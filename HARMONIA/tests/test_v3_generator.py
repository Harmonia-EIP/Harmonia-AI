"""v3 generator building blocks: preset codecs, diffusion sampler and text -> sound prior."""

import numpy as np
import torch

from src.presets import dx7_render
from src.synth import v3_params as P
from src.v3 import codecs
from src.v3.diffusion import Diffusion
from src.v3.prior import Prior, prior_loss
from tests.test_preset_readers import _bulk, _packed_voice


def test_analog_codec_round_trips_and_clamps():
    physical = P.physical_from_dict({"osc_1_waveform": 2, "filter_cutoff": 1500, "unison_voices": 5})
    vector = codecs.analog_encode(physical)
    assert vector.shape == (codecs.ANALOG_DIM,) and vector.min() >= -1 and vector.max() <= 1
    np.testing.assert_allclose(codecs.analog_decode(vector), physical, rtol=1e-6, atol=1e-6)
    wild = codecs.analog_decode(np.full(codecs.ANALOG_DIM, 3.0))
    assert np.all(wild <= np.array([s.end for s in P.SPECS]))


def test_dx7_codec_round_trips_through_the_patch(tmp_path):
    from src.presets import dx7

    path = tmp_path / "cart.syx"
    path.write_bytes(_bulk([_packed_voice(algorithm=17)] * 32))
    voice = next(dx7.read_file(path))
    vector = codecs.dx7_encode(voice)
    assert vector.shape == (codecs.DX7_DIM,)
    decoded = codecs.dx7_decode(vector, voice["name"])
    assert dx7_render.to_patch(decoded) == dx7_render.to_patch(voice)
    assert dx7_render.to_patch(dx7_render.from_patch(dx7_render.to_patch(voice))) == dx7_render.to_patch(voice)
    noisy = codecs.dx7_decode(vector + np.random.default_rng(0).normal(0, 0.3, vector.shape))
    dx7_render.to_patch(noisy)  # any vector decodes to a valid voice


def test_diffusion_learns_a_conditional_toy_distribution():
    torch.manual_seed(0)
    diffusion = Diffusion.create(dim=2, cond_dim=4, width=64, depth=2)
    opt = torch.optim.Adam(diffusion.model.parameters(), lr=3e-3)
    conds = torch.eye(4)[:2]
    centres = torch.tensor([[0.6, -0.6], [-0.6, 0.6]])
    for _ in range(600):
        k = torch.randint(0, 2, (128,))
        x0 = centres[k] + 0.05 * torch.randn(128, 2)
        loss = diffusion.loss(x0, conds[k])
        opt.zero_grad()
        loss.backward()
        opt.step()
    samples = diffusion.sample(conds[[0] * 64], steps=30, guidance=2.0)
    assert torch.linalg.norm(samples.mean(0) - centres[0]) < 0.2


def test_prior_outputs_unit_vectors_and_its_loss_prefers_the_right_pair():
    prior = Prior(dim=8, width=16, depth=1)
    out = prior(torch.randn(5, 8))
    torch.testing.assert_close(out.norm(dim=-1), torch.ones(5))
    audio = torch.nn.functional.normalize(torch.randn(5, 8), dim=-1)
    assert prior_loss(audio, audio) < prior_loss(audio, audio.roll(1, 0))


def test_variation_sampling_starts_from_the_given_presets():
    torch.manual_seed(0)
    diffusion = Diffusion.create(dim=6, cond_dim=3, width=32, depth=1)
    cond = torch.zeros(4, 3)
    start = torch.full((4, 6), 0.8)
    near = diffusion.sample(cond, steps=20, start=start, strength=0.05)
    far = diffusion.sample(cond, steps=20, start=start, strength=1.0)
    assert (near - start).abs().mean() < 0.15 < (far - start).abs().mean()
    # strength 1 without a start is the plain sampler
    g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    torch.testing.assert_close(diffusion.sample(cond, steps=10, generator=g1),
                               diffusion.sample(cond, steps=10, generator=g2, strength=1.0))
