import enum


class TickCounter:
    def __init__(self) -> None:
        self._tick = 0

    @property
    def tick(self):
        return self._tick

    def update(self) -> None:
        self._tick += 1

    def reset(self) -> None:
        self._tick = 0


class EventType(enum.Enum):
    SPAWN_FOOD_EVENT = 1


class TickEvent:
    def __init__(self, event_type: EventType, tick: int) -> None:
        self._event_type = event_type
        self._tick = tick
        self._tick_counter = TickCounter()

    @property
    def event_type(self):
        return self._event_type

    @property
    def tick(self):
        return self._tick

    def trigger(self):
        return self._tick == self._tick_counter.tick

    def update(self):
        self._tick_counter.update()
        if self._tick < self._tick_counter.tick:
            self._tick_counter.reset()


class TickEvents:
    def __init__(self) -> None:
        self._event_list = []

    def set_tick_event(self, event_type: EventType, tick: int):
        self._event_list.append(TickEvent(event_type, tick))

    def get(self) -> list:
        return [event.event_type for event in self._event_list if event.trigger()]

    def update(self) -> None:
        for event in self._event_list:
            event.update()
