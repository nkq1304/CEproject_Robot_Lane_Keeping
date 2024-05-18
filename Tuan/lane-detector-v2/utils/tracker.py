import time

from typing import List


class TrackerManager:
    trackers: List["Tracker"] = []

    @staticmethod
    def add(tracker: "Tracker"):
        TrackerManager.trackers.append(tracker)

    @staticmethod
    def stop():
        for tracker in TrackerManager.trackers:
            tracker.log()


class Tracker:
    def __init__(self, name: str) -> None:
        self.name = name
        self.prev_time = 0
        self.new_time = 0
        self.elapsed_times = []

        TrackerManager.add(self)

    def start(self) -> None:
        self.new_time = time.time()

    def end(self) -> None:
        self.prev_time = self.new_time
        self.new_time = time.time()
        elapsed_time = self.new_time - self.prev_time
        self.elapsed_times.append(elapsed_time)

    def fps(self) -> float:
        return 1 / (self.new_time - self.prev_time)

    def log(self) -> None:
        mean_time = sum(self.elapsed_times) / len(self.elapsed_times)
        print(f"{self.name} mean time: {mean_time:.6f}s")
