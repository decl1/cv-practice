from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Detection:
    label: str
    confidence: float
    box: Tuple[int, int, int, int]


class BaseDetector:
    def __init__(self, conf_threshold: float = 0.3, nms_threshold: float = 0.45):
        self.conf_threshold = float(conf_threshold)
        self.nms_threshold = float(nms_threshold)

    def detect(self, image) -> List[Detection]:
        raise NotImplementedError