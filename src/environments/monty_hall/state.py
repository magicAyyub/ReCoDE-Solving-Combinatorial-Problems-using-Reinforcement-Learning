from enum import Enum, IntEnum, auto

class DoorState(IntEnum):
    """State of each door in the observation vector."""

    CLOSED = 0  # Unopened & unchosen
    GOAT = 1  # Opened and reveals a goat
    CAR = 2  # Opened and reveals a car (a win)
    CHOSEN = 3  # Still closed but currently selected by the player


class Phase(Enum):
    """Progress phase of an episode."""

    AWAITING_FIRST_PICK = auto()
    AFTER_REVEAL = auto()
    DONE = auto()
