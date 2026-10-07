"""Readers of third-party preset files (DX7 sysex, OB-Xf / Surge .fxp), on hand-built files."""

import struct

from src.presets import dx7, fxp


def _packed_voice(name=b"E.PIANO 1 ", algorithm=4, feedback=6):
    ops = []
    for op in range(6):  # op6 first in the sysex
        ops += [99, 50, 30, 60, 99, 80, 0, 0,  # rates, levels
                39, 0, 10, (2 << 2) | 1,  # breakpoint, depths, curves
                (7 << 3) | 2, (3 << 2) | 1,  # detune 7 / rate scaling 2, velocity 3 / ams 1
                90 - op, (14 << 1), 0]  # output level, coarse 14 (ratio), fine
    glob = [99, 99, 99, 99, 50, 50, 50, 50, algorithm, (1 << 3) | feedback, 35, 0, 0, 0, (3 << 4) | (2 << 1) | 1,
            24]
    return bytes(ops + glob) + name


def _bulk(voices):
    data = b"".join(voices)
    checksum = (-sum(data)) & 0x7F
    return bytes([0xF0, 0x43, 0x00, 0x09, 0x20, 0x00]) + data + bytes([checksum, 0xF7])


def test_dx7_bulk_dump_is_decoded_op1_first(tmp_path):
    path = tmp_path / "cart.syx"
    path.write_bytes(_bulk([_packed_voice()] * 32))
    voices = list(dx7.read_file(path))
    assert len(voices) == 32
    v = voices[0]
    assert v["name"] == "E.PIANO 1" and v["algorithm"] == 4 and v["feedback"] == 6 and v["osc_key_sync"] == 1
    assert [op["output_level"] for op in v["operators"]] == [85, 86, 87, 88, 89, 90]  # op1..op6
    op = v["operators"][0]
    assert (op["detune"], op["rate_scaling"], op["key_vel_sens"], op["amp_mod_sens"]) == (7, 2, 3, 1)
    assert (op["kbd_left_curve"], op["kbd_right_curve"], op["freq_coarse"]) == (1, 2, 14)
    assert (v["lfo_key_sync"], v["lfo_wave"], v["pitch_mod_sens"], v["transpose"]) == (1, 2, 3, 24)


def test_dx7_out_of_range_voice_is_dropped(tmp_path):
    bad = bytearray(_packed_voice())
    bad[0] = 120  # op6 rate 1 > 99
    path = tmp_path / "cart.syx"
    path.write_bytes(_bulk([bytes(bad)] + [_packed_voice()] * 31))
    assert [v["slot"] for v in dx7.read_file(path)] == list(range(1, 32))


def test_dx7_algorithm_table_matches_the_dx7_chart():
    assert dx7.carriers(0) == [1, 3]  # algorithm 1
    assert dx7.carriers(4) == [1, 3, 5]  # algorithm 5
    assert dx7.carriers(15) == [1]  # algorithm 16
    assert dx7.carriers(31) == [1, 2, 3, 4, 5, 6]  # algorithm 32
    assert dx7.feedback_operator(31) == 6


def _fxp(plugin: bytes, chunk: bytes) -> bytes:
    header = b"CcnK" + struct.pack(">I", 0) + b"FPCh" + struct.pack(">I", 1) + plugin + struct.pack(">II", 1, 1)
    header += b"Patch".ljust(28, b"\x00") + struct.pack(">I", len(chunk))
    return header + chunk


def test_obxf_patch_parameters_and_metadata(tmp_path):
    xml = b'<?xml version="1.0"?><OB-Xf FilterCutoff="0.44" Osc2Pitch="0.75" programName="Brass" author="A" ' \
          b'license="CC0" category="Brass"/>'
    path = tmp_path / "p.fxp"
    path.write_bytes(_fxp(b"OBXf", b"VC2!" + struct.pack("<I", len(xml)) + xml + b"\x00"))
    patch = fxp.read_obxf(path)
    assert patch["params"] == {"FilterCutoff": 0.44, "Osc2Pitch": 0.75}
    assert patch["meta"]["programName"] == "Brass" and patch["meta"]["license"] == "CC0"


def test_surge_patch_parameters_and_modulation(tmp_path):
    xml = (b'<?xml version="1.0"?><patch revision="15"><meta name="Organ" category="Keys" author="B"/>'
           b'<parameters><a_osc1_type type="0" value="2"/><a_filter1_cutoff type="2" value="3.5">'
           b'<modrouting source="17" depth="0.25" muted="0" source_index="0"/>'
           b'<modrouting source="18" depth="1.0" muted="1"/></a_filter1_cutoff></parameters></patch>')
    chunk = b"sub3" + struct.pack("<I", len(xml)) + struct.pack("<6I", 0, 0, 0, 0, 0, 0) + xml
    path = tmp_path / "s.fxp"
    path.write_bytes(_fxp(b"cjs3", chunk))
    patch = fxp.read_surge(path)
    assert patch["params"] == {"a_osc1_type": 2, "a_filter1_cutoff": 3.5}
    assert patch["modulation"] == [{"target": "a_filter1_cutoff", "source": 17, "index": 0, "depth": 0.25}]
    assert patch["meta"]["name"] == "Organ"
