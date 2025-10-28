from functools import partial

import librosa
import numpy as np
import scipy


class Sequential:
    def __init__(self, *args):
        self.transforms = args

    def __call__(self, inp: np.ndarray):
        res = inp
        for transform in self.transforms:
            res = transform(res)
        return res


class Windowing:
    def __init__(self, window_size=1024, hop_length=None):
        self.window_size = window_size
        self.hop_length = hop_length if hop_length else self.window_size // 2

    def __call__(self, waveform):
        left_pad = self.window_size // 2
        right_pad = self.window_size // 2
        padded_waveform = np.pad(
            waveform, (left_pad, right_pad), mode="constant", constant_values=0
        )
        windows = []
        for start in range(
            0, len(padded_waveform) - self.window_size + 1, self.hop_length
        ):
            window = padded_waveform[start : start + self.window_size]
            windows.append(window)
        return np.array(windows)


class Hann:
    def __init__(self, window_size=1024):
        self.window_size = window_size
        self.hann_window = scipy.signal.windows.hann(window_size, sym=False)

    def __call__(self, windows):
        return windows * self.hann_window


class DFT:
    def __init__(self, n_freqs=None):
        self.n_freqs = n_freqs

    def __call__(self, windows):
        spec = np.fft.rfft(windows)
        if self.n_freqs is not None:
            spec = spec[:, : self.n_freqs]
        return np.abs(spec)


class Square:
    def __call__(self, array):
        return np.square(array)


class Mel:
    def __init__(self, n_fft, n_mels=80, sample_rate=22050):
        self.mel_filter_matrix = librosa.filters.mel(
            sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=1, fmax=sample_rate // 2
        )
        self.inverse_mel_filter_matrix = np.linalg.pinv(self.mel_filter_matrix)

    def __call__(self, spec):
        return np.matmul(spec, self.mel_filter_matrix.T)

    def restore(self, mel):
        return np.matmul(mel, self.inverse_mel_filter_matrix.T)


class GriffinLim:
    def __init__(self, window_size=1024, hop_length=None, n_freqs=None):
        self.griffin_lim = partial(
            librosa.griffinlim,
            n_iter=32,
            hop_length=hop_length,
            win_length=window_size,
            n_fft=window_size,
            window="hann",
        )

    def __call__(self, spec):
        return self.griffin_lim(spec.T)


class Wav2Spectrogram:
    def __init__(self, window_size=1024, hop_length=None, n_freqs=None):
        self.windowing = Windowing(window_size=window_size, hop_length=hop_length)
        self.hann = Hann(window_size=window_size)
        self.fft = DFT(n_freqs=n_freqs)
        # self.square = Square()
        self.griffin_lim = GriffinLim(
            window_size=window_size, hop_length=hop_length, n_freqs=n_freqs
        )

    def __call__(self, waveform):
        return self.fft(self.hann(self.windowing(waveform)))

    def restore(self, spec):
        return self.griffin_lim(spec)


class Wav2Mel:
    def __init__(
        self,
        window_size=1024,
        hop_length=None,
        n_freqs=None,
        n_mels=80,
        sample_rate=22050,
    ):
        self.wav_to_spec = Wav2Spectrogram(
            window_size=window_size, hop_length=hop_length, n_freqs=n_freqs
        )
        self.spec_to_mel = Mel(
            n_fft=window_size, n_mels=n_mels, sample_rate=sample_rate
        )

    def __call__(self, waveform):
        return self.spec_to_mel(self.wav_to_spec(waveform))

    def restore(self, mel):
        return self.wav_to_spec.restore(self.spec_to_mel.restore(mel))


class TimeReverse:
    def __call__(self, mel):
        return mel[::-1, :]


class Loudness:
    def __init__(self, loudness_factor):
        self.loudness_factor = loudness_factor

    def __call__(self, mel):
        return mel * self.loudness_factor


class PitchUp:
    def __init__(self, num_mels_up):
        self.num_mels_up = num_mels_up

    def __call__(self, mel):
        # mel: (time, n_mels)
        mel_shifted = np.roll(
            mel, self.num_mels_up, axis=1
        )  # move low freq to high freq
        mel_shifted[:, : self.num_mels_up] = 0.0
        return mel_shifted


class PitchDown:
    def __init__(self, num_mels_down):
        self.num_mels_down = num_mels_down

    def __call__(self, mel):
        # mel: (time, n_mels)
        mel_shifted = np.roll(mel, -self.num_mels_down, axis=1)
        mel_shifted[:, -self.num_mels_down :] = 0.0
        return mel_shifted


class SpeedUpDown:
    def __init__(self, speed_up_factor=1.0):
        self.speed_up_factor = speed_up_factor

    def __call__(self, mel):
        n_new = max(1, int(round(mel.shape[0] / self.speed_up_factor)))
        return scipy.signal.resample(mel, n_new)


class FrequenciesSwap:
    def __call__(self, mel):
        return mel[:, ::-1]


class WeakFrequenciesRemoval:
    def __init__(self, quantile=0.05):
        self.quantile = quantile

    def __call__(self, mel):
        threshold_per_freq = np.quantile(mel, self.quantile, axis=0)
        mel_filtered = mel.copy()
        for i in range(mel.shape[1]):
            mel_filtered[:, i] = np.where(
                mel[:, i] < threshold_per_freq[i], 0.0, mel[:, i]
            )
        return mel_filtered


class Cringe1:
    def __init__(self):
        # Your code here
        raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^

    def __call__(self, mel):
        # Your code here
        raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^


class Cringe2:
    def __init__(self):
        # Your code here
        raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^

    def __call__(self, mel):
        # Your code here
        raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^
