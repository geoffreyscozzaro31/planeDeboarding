from enum import IntEnum

class PassengerState(IntEnum):
    UNDEFINED = 0
    SEATED = 1
    STAND_UP_FROM_SEAT = 2
    MOVE_WAIT = 3
    MOVE_FROM_ROW = 4
    DISEMBARKED = 5