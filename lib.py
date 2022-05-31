from curses import raw
import numpy as np
from numpy import fft, hsplit
from scipy import signal


def add_noise(audio_data: np.ndarray, mu: int = 0, sigma: int = 1):
    '''
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
    '''
    noised_data = audio_data + mu + sigma * np.random.randn(*audio_data.shape)
    return np.ceil(noised_data)


def SS_denoise(raw_data: np.ndarray, frame_rate: int, alpha: int, beta: int):
    '''
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
    '''
    return np.apply_along_axis(
        __spectral_subtraction, -1, raw_data, frame_rate, alpha, beta)


def __spectral_subtraction(raw_data: np.ndarray,
                           frame_rate: int, alpha: int, beta: int):
    '''
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
    '''

    assert raw_data.ndim == 1, "Input must be one demensional!"

    original_size = raw_data.shape[0]
    frame_size = int(np.ceil(20 * frame_rate / 1000))
    # padding at the end
    raw_data = np.pad(
        raw_data, (0, int(frame_size - (original_size % frame_size))))
    frame_number = int(raw_data.shape[0] / frame_size)

    # calculate the average noise power spectrum of the first 5 frames
    noise_power_sectrum = np.mean(
        np.abs(fft.fft(raw_data[: 5 * frame_size].reshape(5, -1))) ** 2, axis=0)

    # apply window function and overlap to the raw data
    window = signal.windows.hamming(frame_size)
    overlap_size = frame_size // 2   # 50% overlap
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
