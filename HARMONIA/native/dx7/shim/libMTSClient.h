// Stand-in for the MTS-ESP client: no tuning master, standard tuning only.
#ifndef HARMONIA_MTS_STUB_H
#define HARMONIA_MTS_STUB_H
struct MTSClient;
inline bool MTS_HasMaster(const MTSClient *) { return false; }
inline double MTS_NoteToFrequency(const MTSClient *, char, char) { return 440.0; }
#endif
