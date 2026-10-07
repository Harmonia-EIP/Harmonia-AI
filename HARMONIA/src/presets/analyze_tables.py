"""Surge XT enum tables at the pinned commit, shared by the analysis and the converter."""

# Surge XT enums at the pinned commit (src/common/SurgeStorage.h, ModulationSource.h, sst-filters).
SURGE_OSC = ["classic", "sine", "wavetable", "shnoise", "audioinput", "fm3", "fm2", "window", "modern",
             "string", "twist", "alias"]
SURGE_UNISON_OSC = {0, 1, 2, 7, 8, 11}  # oscillator types whose param6 is the unison voice count
SURGE_LFO_SOURCES = set(range(17, 29))  # voice LFO 1-6, scene LFO 1-6
SURGE_EG_SOURCES = {15, 16}  # amp EG, filter EG
SURGE_FILTER_FAMILY = {
    1: "lp", 2: "lp", 3: "lp", 10: "lp", 11: "lp", 12: "lp", 13: "lp", 15: "lp", 16: "lp", 28: "lp",
    4: "hp", 5: "hp", 14: "hp", 17: "hp", 20: "hp", 29: "hp",
    6: "bp", 19: "bp", 22: "bp", 23: "bp", 31: "bp",
    7: "notch", 18: "notch", 21: "notch", 24: "notch", 30: "notch",
    8: "comb", 25: "comb",
}
SURGE_FILTER_24DB = {2, 3, 5, 10, 12, 15, 23, 24}
SURGE_FX_FAMILY = {
    1: "delay", 30: "delay", 2: "reverb", 11: "reverb", 27: "reverb", 31: "reverb",
    9: "chorus", 20: "chorus", 3: "phaser", 12: "flanger", 4: "rotary",
    5: "distortion", 18: "distortion", 25: "distortion", 28: "distortion", 24: "distortion",
    6: "eq", 16: "eq", 8: "eq", 13: "ringmod", 7: "freqshift", 17: "resonator", 21: "comb",
}
