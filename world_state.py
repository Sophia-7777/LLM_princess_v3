from event import Event, time_to_minutes
class world_state:
    def __init__(self):
        #当前时间
        self.current_time = 0
        self.event = []
        self.npc_locations = {}

    def add_event(self, event):
        self.event.append(event)
    
    def get_events(self):
        return self.event
    
    def get_events_between(self, start_time, end_time):
        return [event for event in self.event if start_time <= event.time <= end_time]
    
    def set_npc_location(self, npc_id, location):
        return self.npc_locations.update({npc_id: location})
    
    def get_npc_location(self, npc_id):
        return self.npc_locations.get(npc_id, None)
    
if __name__ == "__main__":
    world = world_state()
    event1 = Event(
            id="e001",
            time=time_to_minutes("23:10"),
            actor="barbie",
            action="meet",
            target="edward",
            location="rose_garden"
        )
    event2 = Event(
            id="e002",
            time=time_to_minutes("05:10"),
            actor="barbie",
            action="meet",
            target="edward",
            location="rose_garden"
        )
    world.add_event(event1)
    world.add_event(event2)
    print(world.get_events())
    print(world.get_events_between(time_to_minutes("06:00"), time_to_minutes("23:59")))
    world.set_npc_location('barbie', 'rose_garden')
    print(world.get_npc_location('barbie'))