from personality import Personality
from memory import Memory
from event import time_to_minutes

class NPC:
    def __init__(self, npc_id, name, role, personality):
        self.npc_id = npc_id
        self.name = name
        self.role = role
        self.personality = personality
        self.goals = []
        self.secrets = []
        self.memories = []
    
    def add_goal(self, goal):
        self.goals.append(goal)

    def add_secret(self, secret):
        self.secrets.append(secret)

    def add_memory(self, memory):
        self.memories.append(memory)

if __name__ == "__main__":
    emma_personality = Personality(honesty=0.8, courage=0.6, intelligence=0.7, kindness=0.9, suspicion=0.2, confidence=0.5)
    emma = NPC(npc_id="npc001", name="Emma", role="villager", personality=emma_personality)
    memory = Memory(
        id="memory_001",
        npc_id="npc001",
        content="A",
        timestamp=time_to_minutes("23:00"),
        importance=0.8,
        event_id = "e001"
    )
    emma.add_goal("Find the lost cat")
    emma.add_secret("Has a hidden stash of gold")
    emma.add_memory(memory.content)
    print("NPC ID:", emma.npc_id)
    print("Name:", emma.name)
    print("Role:", emma.role)
    print("Personality:", emma.personality)

    print("Goals:", emma.goals)
    print("Secrets:", emma.secrets)
    print("Memories:", emma.memories)