from typing import Any


class Agent:
    name: str = "base"
    weight: float = 0.0

    def score(self, candidate: dict, context: dict) -> dict[str, Any]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<Agent {self.name} weight={self.weight}>"
