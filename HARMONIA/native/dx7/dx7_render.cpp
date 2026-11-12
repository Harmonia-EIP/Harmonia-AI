// Render one DX7 voice with the msfa FM core (see README.md).
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>

#include "controllers.h"
#include "dx7note.h"
#include "env.h"
#include "exp2.h"
#include "fm_core.h"
#include "freqlut.h"
#include "lfo.h"
#include "pitchenv.h"
#include "porta.h"
#include "sin.h"
#include "tuning.h"

namespace {

// 12-TET: (1 << 24) * log2(frequency), as in Dexed's standard tuning.
struct StandardTuning : public TuningState {
    int32_t midinote_to_logfreq(int midinote) override { return 50857777 + ((1 << 24) / 12) * midinote; }
};

double g_sample_rate = 0.0;

void init_tables(double sample_rate) {
    static bool tables = false;
    if (!tables) {
        Exp2::init();
        Tanh::init();
        Sin::init();
        tables = true;
    }
    if (sample_rate != g_sample_rate) {
        Freqlut::init(sample_rate);
        Lfo::init(sample_rate);
        PitchEnv::init(sample_rate);
        Env::init_sr(sample_rate);
        Porta::init_sr(sample_rate);
        g_sample_rate = sample_rate;
    }
}

}  // namespace

std::shared_ptr<TuningState> createStandardTuning() { return std::make_shared<StandardTuning>(); }

extern "C" int dx7_render(const uint8_t *patch, int midinote, int velocity, int hold_samples, int total_samples,
                          int sample_rate, float *out) {
    if (!patch || !out || total_samples <= 0 || sample_rate <= 0) return -1;
    init_tables(sample_rate);

    FmCore core;
    Controllers controllers;
    controllers.values_[kControllerPitch] = 0x2000;
    controllers.values_[kControllerPitchRangeUp] = 3;
    controllers.values_[kControllerPitchRangeDn] = 3;
    controllers.values_[kControllerPitchStep] = 0;
    controllers.masterTune = 0;
    controllers.modwheel_cc = 0;
    controllers.foot_cc = 0;
    controllers.breath_cc = 0;
    controllers.aftertouch_cc = 0;
    controllers.portamento_enable_cc = false;
    controllers.portamento_cc = 0;
    controllers.core = &core;
    controllers.refresh();

    uint8_t data[156];
    std::memcpy(data, patch, 156);
    Lfo lfo{};  // value-initialized: the sample-and-hold state starts at 0 instead of stack garbage
    lfo.reset(data + 137);
    lfo.keydown();

    // Dx7Note's constructor leaves the operator feedback buffer uninitialized (Dexed reuses voices);
    // build it in zeroed storage so a render does not depend on leftover memory.
    alignas(Dx7Note) unsigned char storage[sizeof(Dx7Note)];
    std::memset(storage, 0, sizeof(storage));
    Dx7Note &note = *new (storage) Dx7Note(createStandardTuning(), nullptr);
    note.init(data, midinote, velocity, 1, &controllers);
    if (data[136]) note.oscSync();

    // Dexed applies MIDI events on 64-sample block boundaries.
    int keyup_at = ((hold_samples + N - 1) / N) * N;
    float dc_in = 0.0f, dc_out = 0.0f;
    const float dc_r = 1.0f - (126.0f / static_cast<float>(sample_rate));
    AlignedBuf<int32_t, N> buf;
    for (int start = 0; start < total_samples; start += N) {
        if (start == keyup_at) note.keyup();
        std::fill(buf.get(), buf.get() + N, 0);
        int32_t lfo_value = lfo.getsample();
        int32_t lfo_delay = lfo.getdelay();
        note.compute(buf.get(), lfo_value, lfo_delay, &controllers);
        for (int j = 0; j < N && start + j < total_samples; ++j) {
            int32_t val = buf.get()[j] >> 4;
            int clip = val < -(1 << 24) ? 0x8000 : val >= (1 << 24) ? 0x7fff : val >> 9;
            float f = std::max(-1.0f, std::min(1.0f, static_cast<float>(clip) / static_cast<float>(0x8000)));
            float filtered = f - dc_in + dc_r * dc_out;  // Dexed's DC filter (PluginFx)
            dc_in = f;
            dc_out = filtered;
            out[start + j] = filtered;
        }
    }
    note.~Dx7Note();
    return 0;
}
