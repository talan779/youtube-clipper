"""Plain data types shared across the pipeline, with JSON (de)serialization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass
class Transcript:
    language: str
    segments: list[Segment]

    @property
    def words(self) -> list[Word]:
        return [w for s in self.segments for w in s.words]

    def words_between(self, start: float, end: float) -> list[Word]:
        return [w for w in self.words if w.end > start and w.start < end]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "Transcript":
        data = json.loads(path.read_text())
        segments = [
            Segment(
                start=s["start"],
                end=s["end"],
                text=s["text"],
                words=[Word(**w) for w in s.get("words", [])],
            )
            for s in data["segments"]
        ]
        return cls(language=data.get("language", "en"), segments=segments)


@dataclass
class Clip:
    start: float
    end: float
    title: str
    hook: str
    description: str
    hashtags: list[str]
    virality_score: int
    reason: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def save_clips(clips: list[Clip], path: Path) -> None:
    path.write_text(json.dumps([asdict(c) for c in clips], indent=2))


def load_clips(path: Path) -> list[Clip]:
    return [Clip(**c) for c in json.loads(path.read_text())]
