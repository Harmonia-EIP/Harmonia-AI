"""Cleaning of the uploaders' descriptions (src/v4/descriptions.py)."""

from src.v4.descriptions import clean_description, shared_sentences, usable, without


# Sentences about the sound are kept word for word; recording gear, links and thanks are dropped.
def test_keeps_the_sound_drops_the_recording():
    text = ("The slow rattling of a cardboard matchbox filled with wooden matches. Soft rattling. "
            "Recorded on a tascam DR-40")
    assert clean_description(text) == "The slow rattling of a cardboard matchbox filled with wooden matches. Soft rattling"


def test_drops_links_licences_and_thanks():
    text = ('Large wooden door slamming shut.\n<a href="http://freesound.org/people/x">Check my other sounds</a>\n'
            "You may use these samples for commercial projects. Thanks for listening!")
    assert clean_description(text) == "Large wooden door slamming shut"


def test_drops_file_names_settings_and_other_languages():
    assert clean_description("44.1kHz 24bit stereo") == ""
    assert clean_description("un camion se acerca y luego se aleja") == ""
    assert clean_description("door_slam_01_take2.wav") == ""
    assert clean_description(None) == ""


# At most two sentences, without repeats.
def test_two_sentences_at_most():
    text = "Glass breaking. Glass breaking. Small shards falling on a tiled floor. A dog barks far away."
    assert clean_description(text) == "Glass breaking. Small shards falling on a tiled floor"


# A pack's boilerplate, repeated across many recordings, is removed from each of them.
def test_shared_sentences_are_removed():
    texts = [f"Part of the sounds dataset. Hit number {i} on a snare" for i in range(5)]
    shared = shared_sentences(texts, max_clips=3)
    assert shared == {"part of the sounds dataset"}
    assert without(texts[0], shared) == "Hit number 0 on a snare"


def test_usable_needs_three_words():
    assert usable("leaky faucet dripping")
    assert not usable("wind")
