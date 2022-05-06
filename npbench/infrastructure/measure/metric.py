from abc import ABC, abstractmethod

class Metric(ABC):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def benchmark(
        self,
        stmt,
        setup="pass",
        out_text="",
        repeat=1,
        context={},
        output=None,
        verbose=True
    ):
        pass
