from dataclasses import dataclass
from event import time_to_minutes
from typing import Optional

@dataclass
class Memory:
    id: str
    npc_id: str
    content: str
    timestamp: int
    importance: float
    event_id: Optional[str] = None


if __name__ == "__main__":
    memory = Memory(
        id="memory_001",
        npc_id="npc001",
        content="Met the player in the village square",
        timestamp=time_to_minutes("23:00"),
        importance=0.8,
        event_id = "e001"
    )

    print(memory)