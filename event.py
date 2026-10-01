from dataclasses import dataclass
from typing import Optional

@dataclass
class Event:
    id: str
    time: int
    actor: str
    action: str
    location: str
    target: Optional[str] = None


def time_to_minutes(time_string: str) -> int:
    hour, minute = map(int, time_string.split(":"))
    return hour * 60 + minute

if __name__ == "__main__":

    event = Event(
        id="e001",
        time=time_to_minutes("23:10"),
        actor="barbie",
        action="meet",
        target="edward",
        location="rose_garden"
    )

    print(event)