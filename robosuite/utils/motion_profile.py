import matplotlib.pyplot as plt
import numpy as np

class MotionProfile:
    def __init__(self, default_control_freq=20, intervals=None, integers_only=False):
        self.default_control_freq = default_control_freq
        self.intervals = intervals if intervals is not None else []
        self.integers_only = integers_only

    def add_interval(self, start_percent, end_percent, control_freq, end_control_freq=None, easing=None):
        """
        Adds an interval with optional interpolation between two control frequencies.
        
        :param start_percent: Start percentage of the interval.
        :param end_percent: End percentage of the interval.
        :param control_freq: Control frequency at the start of the interval.
        :param end_control_freq: Control frequency at the end of the interval (for interpolation).
        :param easing: Easing function to use for interpolation. Options are:
                      'linear' (default), 'ease_in', 'ease_out', 'ease_in_out'
        """
        self.intervals.append((start_percent, end_percent, control_freq, end_control_freq, easing))
        self.intervals.sort(key=lambda x: x[0])

    def _ease(self, t, easing):
        if easing is None or easing == 'linear':
            return t
        elif easing == 'ease_in':
            return t * t
        elif easing == 'ease_out':
            return t * (2 - t)
        elif easing == 'ease_in_out':
            return 2 * t * t if t < 0.5 else -1 + (4 - 2 * t) * t
        else:
            raise ValueError(f"Unknown easing function: {easing}")

    def get_control_frequency(self, current_step, total_steps):
        current_percent = (current_step / total_steps) * 100

        for start, end, start_freq, end_freq, easing in self.intervals:
            if start <= current_percent <= end:
                if end_freq is not None:
                    # Interpolate between start_freq and end_freq with easing
                    percent_within_interval = (current_percent - start) / (end - start)
                    eased_percent = self._ease(percent_within_interval, easing)
                    freq = start_freq + eased_percent * (end_freq - start_freq)
                    return int(freq) if self.integers_only else freq
                return int(start_freq) if self.integers_only else start_freq
        return int(self.default_control_freq) if self.integers_only else self.default_control_freq

    def generate_graph(self, total_steps):
        """
        Generates a graph of the control frequency over the given number of steps.
        
        :param total_steps: Total number of steps to simulate.
        :return: A matplotlib figure object.
        """
        steps = np.arange(total_steps)
        frequencies = [self.get_control_frequency(step, total_steps) for step in steps]

        fig, ax = plt.subplots()
        ax.plot(steps, frequencies, label='Control Frequency')
        ax.set_xlabel('Step')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title('Motion Profile')
        ax.legend()
        return fig