from ..base import SeparationBase

class MixAsEstimate(SeparationBase):
    """
    This algorithm does nothing but scale the mix by the number of sources. This can
    be used to compute the improvement metrics (e.g. improvement in SDR over using the
    mixture as the estimate).
    
    Args:
        input_audio_signal (AudioSignal): Signal to separate.
        num_sources (int): How many sources to return.
    """

    def __init__(self, input_audio_signal, num_sources):
        self.num_sources = num_sources
        super().__init__(input_audio_signal=input_audio_signal)

    def run(self):
        pass

    def make_audio_signals(self):
        scalar = 1
        estimates = [
            self.audio_signal * scalar for i in range(self.num_sources)
        ]
        return estimates
