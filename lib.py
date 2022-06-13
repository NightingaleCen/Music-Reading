import numpy as np
from numpy import fft, hsplit
from scipy import signal
import array


def export_mp3(audio, data, name):
    """
    Export a numpy array as a mp3 file.
    Parameters
    ----------
    audio:
        Format from Pydub AudioSegment.
    data:
        Numpy array to be converted.
    name:
        File name to be saved.
    Return
    -------

    """
    data = data * 32767 / np.max(np.abs(data))
    data = data.reshape(-1).real.astype(np.int16)
    export_audio = audio._spawn(array.array(audio.array_type, data))
    export_audio.export(name + ".mp3", format="mp3")


def add_noise(audio_data: np.ndarray, mu: int = 0, sigma: int = 1):
    """
    Add Gaussian noise with a mean of `mu` and standard deviation of `sigma` to `audio_data`

    Parameters
    ----------
    audio_data : ndarray
        Input data of arbitary dimentions.
    mu: int
        Mean of the Gaussian noise.
    sigma : int
        Standard deviation of the Gaussian noise.

    Return
    ------
    out : ndarray
        The output array with added noise.
    """
    noised_data = audio_data + mu + sigma * np.random.randn(*audio_data.shape)
    return np.ceil(noised_data)


def SS_denoise(raw_data: np.ndarray, frame_rate: int = 44100, alpha: int = 5, beta: int = 0.005):
    """
    Apply spectral substraction to the last axis of `raw_data`.

    Parameters
    ----------
    raw_data : ndarray
        Input data of arbitary dimentions.
    frame_rate: int
        Frame rate (sample rate) of the input signal.
    alpha : int
        The desired value of parameter alpha at SNR=0 dB, suggested value is 3-6.
    beta : int
        The spectral floor parameter beta, suggested value is 0.005-0.1.

    Return
    ------
    out : ndarray
        The denoised version of `raw_data`.
    """
    return np.apply_along_axis(
        __spectral_subtraction, -1, raw_data, frame_rate, alpha, beta)


def SS_pitch_extraction(raw_data: np.ndarray, frame_rate: int = 44100, piece_size: float = 0.1):
    """
    Extract pitch at the last axis of `raw_data`.

    Parameters
    ----------
    raw_data : ndarray
        Input data of arbitary dimentions.
    frame_rate: int, default 44100 Hz
        Frame rate (sample rate) of the input signal.
    piece_size: int, default 0.1s
        Short-time Fourier transform period.

    Return
    ------
    out : ndarray
        A frequency list at each time slice.
        :param piece_size: Short-time Fourier transform period.
    """
    return np.apply_along_axis(
        __pitch_extraction, -1, raw_data, frame_rate, piece_size)


def SS_tune_export(fft_data_list: list):
    """
    Convert data from frequency to pitch, do re mi fa so la ti do.
    This function is always used after SS_pitch_extraction.

    Parameters
    ----------
    fft_data_list: list
        Output of SS_pitch_extraction.

    Return
    ------
    out : list
        A two-dimentional list of tune name.
    """
    tune_result = []
    duplicate = []
    for i in fft_data_list:
        if i:
            tune = __search_tune(i)
            if len(tune) <= 5 and tune != duplicate:
                tune_result.append(tune)
                duplicate = tune

    return tune_result


def __search_tune(piece: list):
    """
    Convert data from frequency to pitch, do re mi fa so la ti do.
    This function should not be called. Please use SS_tune_export.

    Parameters
    ----------
    piece: list
        A piece of FFT result.
    Return
    ------
    out : list
        A chord of tune name.
    """
    # 保存音高，共有6个八度。一个八度为频率乘2._A_表示升A
    _A = np.array([55.0 * (2 ** i) for i in range(6)])
    _A_ = np.array([58.27 * (2 ** i) for i in range(6)])
    _B = np.array([61.735 * (2 ** i) for i in range(6)])
    _C = np.array([32.703 * (2 ** i) for i in range(6)])
    _C_ = np.array([34.648 * (2 ** i) for i in range(6)])
    _D = np.array([36.708 * (2 ** i) for i in range(6)])
    _D_ = np.array([38.891 * (2 ** i) for i in range(6)])
    _E = np.array([41.203 * (2 ** i) for i in range(6)])
    _F = np.array([43.654 * (2 ** i) for i in range(6)])
    _F_ = np.array([46.249 * (2 ** i) for i in range(6)])
    _G = np.array([49.0 * (2 ** i) for i in range(6)])
    _G_ = np.array([51.913 * (2 ** i) for i in range(6)])
    tune_list = np.array([_A, _A_, _B, _C, _C_, _D, _D_,
                          _E, _F, _F_, _G, _G_])
    ret = []
    for i in piece:
        # assert type(i) == float
        if not (i >= 32.703 and i <= 1975.5):
            continue
        minindex = 0
        minret = (tune_list[0][0] - i) ** 2
        for index1 in range(12):
            for index2 in range(6):
                nowret = (tune_list[index1][index2] - i) ** 2
                if nowret < minret:
                    minret = nowret
                    minindex = index1
        if minindex == 0 and "A" not in ret:
            ret.append("A")
        if minindex == 1 and "A#" not in ret:
            ret.append("A#")
        if minindex == 2 and "B" not in ret:
            ret.append("B")
        if minindex == 3 and "C" not in ret:
            ret.append("C")
        if minindex == 4 and "C#" not in ret:
            ret.append("C#")
        if minindex == 5 and "D" not in ret:
            ret.append("D")
        if minindex == 6 and "D#" not in ret:
            ret.append("D#")
        if minindex == 7 and "E" not in ret:
            ret.append("E")
        if minindex == 8 and "F" not in ret:
            ret.append("F")
        if minindex == 9 and "F#" not in ret:
            ret.append("F#")
        if minindex == 10 and "G" not in ret:
            ret.append("G")
        if minindex == 11 and "G#" not in ret:
            ret.append("G#")
    return ret


def __pitch_extraction(raw_data: np.ndarray, frame_rate: int, piece: float):
    """
    Extract pitch at the last axis of `raw_data`.

    Parameters
    ----------
    raw_data : ndarray
        Input data of arbitary dimentions.
    frame_rate: int
        Frame rate (sample rate) of the input signal.
    piece: int
        Short-time Fourier transform period.

    Return
    ------
    out : ndarray
        A list of frequency.
    """

    assert raw_data.ndim == 1, "Input must be one demensional!"

    # padding to raw data
    original_size = raw_data.shape[0]
    frame_size = int(np.ceil(frame_rate * piece))
    raw_data = np.pad(
        raw_data, (0, int(frame_size - (original_size % frame_size))))

    # 50% overlap
    overlap_size = frame_size // 2  # 50% overlap
    raw_data = raw_data.reshape(-1, overlap_size)
    overlap_data = np.hstack((raw_data, np.vstack(
        (raw_data[1:], np.zeros((1, raw_data.shape[1]))))))

    # now we're ready to do the FFT
    timestep = 1.0 / frame_rate
    fft_data = np.fft.fft(overlap_data, axis=-1)
    n = overlap_data[0].size
    freq = np.fft.fftfreq(n, timestep)
    freqs = np.fft.fftshift(freq, axes=-1)
    fft_data = np.fft.fftshift(fft_data, axes=-1)
    power_spectrum = np.abs(fft_data) ** 2

    # clear some weak frequency -- if a frequency is too weak, it's likely to be a noise.
    zero_index = np.where(np.abs(power_spectrum) <= 1000)
    power_spectrum[zero_index] = 0

    # analyze the best freqs for every piece of fft_data
    index = np.where(np.abs(power_spectrum) >=
                     (0.4 * (np.max(np.abs(power_spectrum), axis=-1).reshape(-1, 1))))
    result_for_frame = freqs[index[1]]     # result of each FFT frame

    # convert array index to frequency
    result = [[]]
    for i in range(len(index[0])):
        if result_for_frame[i] > 0:
            result[index[0][i]].append(result_for_frame[i])
        result.append([])

    return result


def __spectral_subtraction(raw_data: np.ndarray,
                           frame_rate: int, alpha: int, beta: int):
    """
    Apply spectral substraction to `raw_data`.\\
    You should always use `SS_denoise` instead of this function!

    Parameters
    ----------
    raw_data : ndarray
        Input data of arbitary dimentions.
    frame_rate: int
        Frame rate (sample rate) of the input signal.
    alpha : int
        The desired value of parameter alpha at SNR=0 dB, suggested value is 3-6.
    beta : int
        The spectral floor parameter beta, suggested value is 0.005-0.1.

    Return
    ------
    out : ndarray
        The denoised version of `raw_data`.
    """

    assert raw_data.ndim == 1, "Input must be one demensional!"

    original_size = raw_data.shape[0]
    frame_size = int(np.ceil(20 * frame_rate / 1000))
    # padding at the end
    raw_data = np.pad(
        raw_data, (0, int(frame_size - (original_size % frame_size))))

    # calculate the average noise power spectrum of the first 5 frames
    noise_power_sectrum = np.mean(
        np.abs(fft.fft(raw_data[: 5 * frame_size].reshape(5, -1))) ** 2, axis=0)

    # apply window function and overlap to the raw data
    window = signal.windows.hamming(frame_size)
    overlap_size = frame_size // 2  # 50% overlap
    raw_data = raw_data.reshape(-1, overlap_size)
    overlap_data = np.hstack((raw_data, np.vstack(
        (raw_data[1:], np.zeros((1, raw_data.shape[1]))))))
    preprocessd_data = overlap_data * window

    # now we're ready to do the spectral subtraction
    fft_data = fft.fft(preprocessd_data, axis=-1)
    power_spectrum = np.abs(fft_data) ** 2
    phase = np.angle(fft_data)  # save the phase for the fourier inversion

    # calculate the signal-noise ratio for each frame
    snr = 10 * np.log10((np.sum(power_spectrum, axis=-1) + 1e-4) /
                        (np.sum(noise_power_sectrum) + 1e-4))

    # calculate alpha for each frame
    alpha_array = np.where(snr > 20, 1, alpha - snr * (1 - alpha) / 20)
    alpha_array = np.where(snr < -5, 5, alpha_array).reshape(-1, 1)
    result_power_spectrum = np.where(
        power_spectrum >= alpha_array * noise_power_sectrum,
        power_spectrum - alpha_array * noise_power_sectrum, beta * noise_power_sectrum)
    overlap_result = fft.ifft(np.sqrt(result_power_spectrum) *
                              np.exp(1j * phase))

    # restore the overlap
    left_half, right_half = hsplit(overlap_result, 2)
    result = (np.vstack((left_half, np.zeros((1, overlap_size)))) +
              np.vstack((np.zeros((1, overlap_size)), right_half)
                        )).reshape(-1)[:original_size]

    return result
