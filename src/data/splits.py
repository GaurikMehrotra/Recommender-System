"""Per-user chronological leave-last-positive-out with explicit tie rules."""
from dataclasses import dataclass
import pandas as pd

@dataclass
class Split:
    """Train contains all earlier ratings; relevance is defined separately."""
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    excluded_users: int


def leave_last_out(events: pd.DataFrame, positive_threshold: float = 4.0) -> Split:
    """Hold out final two positive timestamp groups per eligible user.

    Training ends strictly before validation timestamp; test occurs strictly later.
    Intermediate/held-out low ratings are deliberately discarded. Users with fewer
    than three distinct positive timestamps are excluded from this benchmark.
    Global chronology is NOT enforced: other users may train on later dates. This
    is a per-user benchmark and must not be presented as historical deployment.
    Duplicate user-item events are rejected rather than silently leaking repeats.
    """
    if events.duplicated(["user", "item"]).any():
        raise ValueError("Repeated user-item interactions require an explicit repeat policy")
    trains, vals, tests = [], [], []
    excluded = 0
    for _, group in events.groupby("user", sort=True):
        times = sorted(group.loc[group.rating >= positive_threshold, "timestamp"].unique())
        if len(times) < 3:
            excluded += 1
            continue
        val_time, test_time = times[-2:]
        trains.append(group[group.timestamp < val_time])
        vals.append(group[(group.timestamp == val_time) & (group.rating >= positive_threshold)])
        tests.append(group[(group.timestamp == test_time) & (group.rating >= positive_threshold)])
    if not trains:
        raise ValueError("No users with three positive timestamps")
    return Split(*(pd.concat(parts, ignore_index=True) for parts in [trains, vals, tests]), excluded)


def assert_no_leakage(split: Split) -> None:
    """Check disjoint pairs and strict per-user chronology."""
    parts = [split.train, split.validation, split.test]
    sets = [set(zip(p.user, p.item)) for p in parts]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    latest_train = split.train.groupby("user").timestamp.max()
    earliest_val = split.validation.groupby("user").timestamp.min()
    latest_val = split.validation.groupby("user").timestamp.max()
    earliest_test = split.test.groupby("user").timestamp.min()
    assert (latest_train.loc[earliest_val.index] < earliest_val).all()
    assert (latest_val.loc[earliest_test.index] < earliest_test).all()
