"""Synth vocabulary (types read from human labels) and the synth space losses."""

import numpy as np
import torch

from src.v3 import vocabulary as V
from src.v3.synth_space import SynthSpace, contrastive_loss, type_loss


def test_types_are_read_from_categories_then_names():
    assert V.preset_types({"name": "Unison Fear", "category": "Leads"}) == ["lead"]
    assert V.preset_types({"name": "E.PIANO 1", "aliases": [], "category": ""}) == ["keys"]
    assert V.preset_types({"name": "LAURIE", "aliases": ["JUPE STRGS"], "category": ""}) == ["strings"]
    assert V.preset_types({"name": "Say Again.", "aliases": [], "category": ""}) == []
    assert V.preset_types({"name": "Strings", "category": "Jacky Ligon\\Keys"}) == ["keys"]


def test_multi_word_types_win_over_single_words():
    assert V.words_to_types("bass drum") == ["drums"]
    assert V.words_to_types("synth bass") == ["bass"]
    assert V.words_to_types("electric piano and pipe organ") == ["keys", "organ"]
    assert V.fsd_types("Bass_drum,Drum,Percussion,Musical_instrument") == ["drums"]
    assert V.fsd_types("Speech,Human_voice") == []
    np.testing.assert_array_equal(V.multi_hot(["pad", "fx"]).nonzero()[0], [V.TYPES.index("pad"), V.TYPES.index("fx")])


def test_contrastive_loss_counts_identical_texts_as_positives():
    z = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    scale = torch.tensor(20.0)
    distinct = contrastive_loss(z, z, torch.arange(4), scale)
    # presets 0 and 1 share a text: pairing text 0 with sound 1 is no longer an error
    swapped = z.clone()
    swapped[[0, 1]] = z[[1, 0]]
    shared = torch.tensor([0, 0, 2, 3])
    assert contrastive_loss(z, swapped, shared, scale) < contrastive_loss(z, swapped, torch.arange(4), scale)
    assert distinct < contrastive_loss(z, swapped, torch.arange(4), scale)


def test_type_loss_ignores_unlabelled_samples_and_space_outputs_unit_vectors():
    model = SynthSpace(dim=16)
    z = model.audio(torch.randn(3, 512))
    torch.testing.assert_close(z.norm(dim=-1), torch.ones(3))
    logits = model.type_logits(z)
    none = torch.zeros(3, len(V.TYPES))
    assert type_loss(logits, none).item() == 0.0
    one = none.clone()
    one[0, 2] = 1
    assert torch.isclose(type_loss(logits, one), type_loss(logits[:1], one[:1]))
