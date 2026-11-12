"""JUCE state codec and the DX7 voice layout used by the v3 renderers (no plugin needed)."""

import numpy as np
import pytest

from src.presets import dx7_render, juce_state
from tests.test_preset_readers import _bulk, _packed_voice


def test_juce_base64_roundtrip_and_known_value():
    data = bytes(range(256)) + b"OB-Xf"
    assert juce_state.b64_decode(juce_state.b64_encode(data)) == data
    assert juce_state.b64_encode(b"\x01") == "1.A."  # 6 low bits first, alphabet starts with "."


def test_component_state_is_replaced_inside_pedalboard_wrapper():
    inner = juce_state.binary_xml('<VST3PluginState><IComponent>3.0zB</IComponent></VST3PluginState>')
    replaced = juce_state.with_component_state(inner, b"sub3-patch")
    assert juce_state.get_component_state(replaced) == b"sub3-patch"


def test_dx7_patch_layout_matches_the_voice_edit_buffer(tmp_path):
    path = tmp_path / "cart.syx"
    path.write_bytes(_bulk([_packed_voice()] * 32))
    from src.presets import dx7

    voice = next(dx7.read_file(path))
    patch = dx7_render.to_patch(voice)
    assert len(patch) == 156
    assert patch[16] == 90  # op6 output level comes first
    assert patch[5 * 21 + 16] == 85  # op1 last
    assert patch[134] == 4 and patch[135] == 6 and patch[136] == 1  # algorithm, feedback, key sync
    assert patch[145:155] == b"E.PIANO 1 "


@pytest.mark.skipif(not dx7_render.LIBRARY.exists(), reason="run scripts/v3/build_dx7.py first")
def test_dx7_single_sine_operator_is_in_tune(tmp_path):
    path = tmp_path / "cart.syx"
    path.write_bytes(_bulk([_packed_voice()] * 32))
    from src.presets import dx7

    voice = next(dx7.read_file(path))
    voice.update(algorithm=31, feedback=0, transpose=24, lfo_pitch_mod_depth=0, pitch_eg_levels=[50] * 4)
    for i, op in enumerate(voice["operators"]):
        op.update(output_level=99 if i == 0 else 0, freq_coarse=1, freq_fine=0, detune=7, osc_mode=0,
                  rates=[99] * 4, levels=[99, 99, 99, 0], kbd_left_depth=0, kbd_right_depth=0)
    audio = dx7_render.render(voice, note=69, total_seconds=1.0)
    spectrum = np.abs(np.fft.rfft(audio[4800:4800 + 32768] * np.hanning(32768)))
    assert abs(np.fft.rfftfreq(32768, 1 / 48000)[spectrum.argmax()] - 440.0) < 2.0


@pytest.mark.skipif(not dx7_render.LIBRARY.exists(), reason="run scripts/v3/build_dx7.py first")
def test_dx7_velocity_follows_the_dx7_keyboard_range(tmp_path):
    path = tmp_path / "cart.syx"
    path.write_bytes(_bulk([_packed_voice()] * 32))
    from src.presets import dx7

    voice = next(dx7.read_file(path))
    scaled = dx7_render.render(voice, velocity=127, total_seconds=0.5)
    raw = dx7_render.render(voice, velocity=int(127 * dx7_render.DX_VELOCITY), total_seconds=0.5, dx_velocity=False)
    np.testing.assert_array_equal(scaled, raw)
