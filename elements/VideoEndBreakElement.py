from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class VideoEndBreakElement:
    """Событие окончания видеопотока."""
    source: str
