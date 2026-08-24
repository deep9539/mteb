from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class EmotionalVoiceConversionESDRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="EmotionalVoiceConversionESDRetrieval",
        description=(
            "Composed audio retrieval built from the DynamicSuperb EmotionalVoiceConversion_ESD "
            "dataset. Given a source audio and an instruction (text) describing the target emotion, "
            "the goal is to retrieve the target emotional audio among 400 unique candidates. "
            "The correct target is marked as positive, and all other candidates are negative."
        ),
        reference="https://arxiv.org/abs/2105.14762",
        dataset={
            "path": "deep9539/emotional_voice_conversion_esd",
            "revision": "main",
        },
        type="Any2AnyRetrieval",
        category="at2a",
        modalities=["audio", "text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="hit_rate_at_1",
        date=("2021-01-01", "2022-12-31"),
        domains=["Spoken"],
        task_subtypes=["Emotional Speech Retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@article{zhou2022emotional,
  title={Emotional voice conversion: Theory, databases and esd},
  author={Zhou, Kun and Sisman, Berrak and Liu, Rui and Li, Haizhou},
  journal={Speech communication},
  volume={137},
  pages={1--18},
  year={2022},
  publisher={Elsevier}
}
""",
        prompt={
            "query": "Given the source audio and an instruction describing the target emotion, retrieve the corresponding emotional target audio."
        },
    )
