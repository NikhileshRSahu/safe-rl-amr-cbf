from __future__ import annotations


def hesitation_speed_factor(tick: int) -> float:
    """Causal, human-like stop/restart speed profile for the hesitation case.

    Real pedestrians generally expose a short deceleration cue before stopping
    and accelerate again after deciding to continue. The previous benchmark
    teleported velocity from cruise to zero, making the event unobservable from
    motion history. This profile keeps the same stop/restart intent but makes
    acceleration finite and observable to every controller.
    """
    tick = int(tick)
    if tick < 20:
        return 1.0
    if tick <= 28:
        return max(0.0, (28.0 - tick) / 8.0)
    if tick < 42:
        return 0.0
    if tick <= 50:
        return min(1.0, (tick - 41.0) / 9.0)
    if tick < 56:
        return 1.0
    if tick <= 62:
        return max(0.0, (62.0 - tick) / 6.0)
    if tick < 68:
        return 0.0
    return 1.0
