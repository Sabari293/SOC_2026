import numpy as np

class VolumeFilter:
    def __init__(self, volume: np.ndarray, window: int = 20, c: float = 1.5):
        self.volume = volume
        self.window = window
        self.c=c
        self.sma = np.full(len(volume), np.nan)
        self.valid = np.full(len(volume), False)
        rolling_sum = np.sum(self.volume[:window])

        avg = rolling_sum / window
        self.sma[window-1] = avg
        self.valid[window-1] = self.volume[window-1] > self.c * avg

        for i in range(window, len(volume)):
            rolling_sum += self.volume[i] - self.volume[i-window]

            avg = rolling_sum / window

            self.sma[i] = avg
            self.valid[i] = self.volume[i] > self.c * avg

    def volumenew(self, current_volume):
        self.volume = np.append(self.volume, current_volume)

        if len(self.volume)>=self.window:
            avg = np.mean(self.volume[-self.window:])
        else:
            avg = np.mean(self.volume)

        self.sma = np.append(self.sma, avg)

        is_valid = current_volume > self.c * avg
        self.valid = np.append(self.valid, is_valid)

        return avg, is_valid

    def filter(self):
        return self.valid