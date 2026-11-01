"""Validated source-to-output timing shared by rendering evidence and captions."""

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RetainedPart:
    source_start: float
    source_end: float
    speed: float = 1.0

    def __post_init__(self):
        if not all(math.isfinite(value) for value in (self.source_start, self.source_end, self.speed)):
            raise ValueError("Non-finite edit timing")
        if self.source_start < 0 or self.source_end <= self.source_start or self.speed <= 0:
            raise ValueError("Invalid retained part")

    @property
    def output_duration(self):
        return (self.source_end - self.source_start) / self.speed

    def output_time(self, source_time: float) -> float:
        if not math.isfinite(source_time) or not self.source_start <= source_time <= self.source_end:
            raise ValueError("Source timestamp outside retained part")
        return (source_time - self.source_start) / self.speed

    def to_dict(self):
        return {**asdict(self), "output_start": 0.0, "output_end": self.output_duration}
