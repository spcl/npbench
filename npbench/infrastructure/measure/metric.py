from abc import ABC, abstractmethod

class Metric(ABC):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def execute(self, **kwargs):
        pass
