"""Concept -> MusicGen prompt mapping (shared with backend_v1)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ConceptPrompt:
    target_bpm: int
    template: str


_SUFFIX = (
    "Key of {key}, {bpm} BPM, perfectly quantized, instrumental, "
    "no vocals, no singing, studio quality production"
)

CONCEPT_PROMPTS: dict[str, ConceptPrompt] = {
    "powerful": ConceptPrompt(
        target_bpm=138,
        template=(
            "Powerful peak-time techno with massive pounding kick drums and an "
            "aggressive driving sub-bass, raw energetic warehouse rave atmosphere, "
            "hard-hitting percussion, " + _SUFFIX
        ),
    ),
    "groove": ConceptPrompt(
        target_bpm=124,
        template=(
            "Groovy funky techno with a rolling syncopated bassline and tight "
            "swung percussion, danceable funk-driven rhythm, warm analog groove, "
            + _SUFFIX
        ),
    ),
    "hypnotic": ConceptPrompt(
        target_bpm=130,
        template=(
            "Hypnotic minimal techno with repetitive looping arpeggios and a "
            "trance-inducing four-on-the-floor pulse, deep meditative atmosphere, "
            "gradually evolving textures, " + _SUFFIX
        ),
    ),
    "dark": ConceptPrompt(
        target_bpm=134,
        template=(
            "Dark industrial techno with distorted metallic textures and an "
            "ominous brooding atmosphere, raw mechanical rhythm, shadowy "
            "soundscape, " + _SUFFIX
        ),
    ),
    "melodic": ConceptPrompt(
        target_bpm=126,
        template=(
            "Melodic techno with ethereal atmospheric synth layers and emotional "
            "evolving chord progressions, cinematic and spiritual mood, " + _SUFFIX
        ),
    ),
    "minimal": ConceptPrompt(
        target_bpm=122,
        template=(
            "Minimal deep techno with a sparse stripped-back arrangement and "
            "subtle textures, spacious deep groove, understated and atmospheric, "
            + _SUFFIX
        ),
    ),
}

CONCEPT_IDS = list(CONCEPT_PROMPTS.keys())


def build_prompt(concept_id: str, key: str) -> str:
    concept = CONCEPT_PROMPTS[concept_id]
    return concept.template.format(key=key, bpm=concept.target_bpm)
