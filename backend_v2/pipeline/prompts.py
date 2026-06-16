"""Concept -> MusicGen prompt mapping (v2)."""

from dataclasses import dataclass

_BASE = (
    "hypnotic driving techno, four-on-the-floor kick, deep analog bassline, "
    "hypnotic synth stabs, warehouse reverb, no vocals, 128 bpm"
)

@dataclass(frozen=True)
class ConceptPrompt:
    target_bpm: int
    template: str


CONCEPT_PROMPTS: dict[str, ConceptPrompt] = {
    "powerful": ConceptPrompt(
        target_bpm=128,
        template=_BASE,
    ),
    "groove": ConceptPrompt(
        target_bpm=128,
        template=_BASE + ", syncopated groove, swung percussion, funky rhythm",
    ),
    "hypnotic": ConceptPrompt(
        target_bpm=128,
        template=_BASE + ", deep repetitive loops, meditative trance pulse",
    ),
    "dark": ConceptPrompt(
        target_bpm=128,
        template=_BASE + ", dark industrial textures, distorted metallic sounds",
    ),
    "melodic": ConceptPrompt(
        target_bpm=128,
        template=_BASE + ", melodic atmospheric synths, emotional chord progressions",
    ),
    "minimal": ConceptPrompt(
        target_bpm=128,
        template=_BASE + ", minimal sparse arrangement, spacious deep groove",
    ),
}

CONCEPT_IDS = list(CONCEPT_PROMPTS.keys())


def build_prompt(concept_id: str) -> str:
    return CONCEPT_PROMPTS[concept_id].template
