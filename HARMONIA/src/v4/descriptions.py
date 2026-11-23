"""Human descriptions of real recordings, cleaned for training (Harmonia v4, #35).

Freesound uploaders describe their sounds in their own words ("The slow rattling of a cardboard matchbox
filled with wooden matches"), but the same text also holds how the sound was made ("Recorded on a tascam
DR-40"), links, licence notes and thanks. Only the sentences about the sound itself are kept: a sentence
naming recording gear, software, file formats, links, licences or the uploader's other work is dropped.

The text stays the uploader's: sentences are kept or dropped whole, never rewritten.
"""

from __future__ import annotations

import html
import re
from typing import Iterable, Optional

# A sentence naming one of these is about the recording, the file or the uploader, not the sound.
NOT_THE_SOUND = re.compile(
    r"\b("
    r"record(ed|ing|ings|er|ers|ist)?|captured|field ?recorder|mic|mics|microphones?|preamps?|"
    r"zoom|tascam|h[1-6]n?|dr-?\d+|rode|ntg\d*|sennheiser|mkh\d*|shure|sm\d+|sony|pcm|marantz|olympus|"
    r"roland|edirol|r-?\d{2}|audio-?technica|neumann|akg|"
    r"edit(ed|ing)?|processed|audacity|cubase|ableton|reaper|logic|pro ?tools|fl ?studio|garageband|"
    r"soundforge|sound forge|izotope|adobe|audition|"
    r"normali[sz]ed|eq|eq'?d|equali[sz]ed|compress(ed|ion|or)?|limit(ed|er)|denois(ed|ing)|noise reduction|"
    r"khz|hz|kbps|bits?|bit-?depth|stereo|mono|wav|mp3|flac|ogg|aiff|sample ?rate|"
    r"bpm|tempo|loop(ed)?|"
    r"licen[sc]e[ds]?|attribution|credit(ed)?|commercial|royalty|copyright|cc0|creative commons|public domain|"
    r"freesound|download(ed|s)?|upload(ed|s)?|please|thanks?|thank you|enjoy|comments?|rate|"
    r"check out|check my|my other|other sounds|follow me|subscribe|donations?|appreciated|"
    r"pack|collection|series|library|project|university|course|class|assignment|students?|"
    r"plugins?|vst|daw|synth(esi[sz]ed|esizer)? patch|"
    r"https?|www|\.com|@"
    r")\b",
    re.IGNORECASE,
)
# Function words of the other languages common on Freesound: a sentence using two of them and no English
# one is not English (the text encoder is taught English, then French through translation).
FOREIGN = {"el", "los", "las", "una", "y", "con", "del", "por", "para", "que", "se", "es", "muy", "pero",
           "le", "les", "des", "du", "et", "avec", "est", "une", "dans", "sur", "der", "die", "das", "und",
           "mit", "ein", "eine", "ist", "von", "il", "di", "della", "il", "che", "em", "um", "uma", "het", "een",
           "met", "van"}
ENGLISH = {"the", "a", "an", "of", "and", "with", "in", "on", "is", "are", "at", "from", "to", "by", "it",
           "this", "that", "into", "some", "being", "while", "then", "for", "its", "as"}
SENTENCE = re.compile(r"(?<=[.!?])\s+|\s*\n+\s*|\s+-\s+|\s*;\s*")
MIN_WORDS, MAX_WORDS, MAX_SENTENCES = 2, 30, 2


def _sentences(text: str) -> Iterable[str]:
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    for sentence in SENTENCE.split(text):
        yield re.sub(r"\s+", " ", sentence).strip(" .,:*\"'()[]")


def _about_the_sound(sentence: str) -> bool:
    words = sentence.split()
    if not MIN_WORDS <= len(words) <= MAX_WORDS:
        return False
    letters = sum(c.isalpha() for c in sentence)
    if letters < 0.75 * len(sentence.replace(" ", "")):  # file names, numbers, lists of settings
        return False
    if NOT_THE_SOUND.search(sentence):
        return False
    lower = [w.strip(".,!?'\"()").lower() for w in words]
    return not (sum(w in FOREIGN for w in lower) >= 2 and not any(w in ENGLISH for w in lower))


def clean_description(text: Optional[str]) -> str:
    """The first sentences of an uploader's description that are about the sound itself ("" if none)."""
    if not text:
        return ""
    kept = []
    for sentence in _sentences(text):
        if _about_the_sound(sentence) and sentence.lower() not in (k.lower() for k in kept):
            kept.append(sentence)
            if len(kept) == MAX_SENTENCES:
                break
    return ". ".join(kept)


def shared_sentences(descriptions: Iterable[str], max_clips: int = 3) -> set:
    """Sentences found in more than `max_clips` descriptions: a pack's boilerplate ("Part of the ... dataset")
    says nothing about each sound."""
    counts: dict = {}
    for description in descriptions:
        for sentence in {s.lower() for s in description.split(". ") if s}:
            counts[sentence] = counts.get(sentence, 0) + 1
    return {sentence for sentence, n in counts.items() if n > max_clips}


def without(description: str, shared: set) -> str:
    """The description without the shared sentences."""
    return ". ".join(s for s in description.split(". ") if s and s.lower() not in shared)


def usable(description: str, min_words: int = 3) -> bool:
    """Long enough to say something a title would not."""
    return len(description.split()) >= min_words
