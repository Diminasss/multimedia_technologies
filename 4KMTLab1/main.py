"""
Вариант №7

Синусоидальный сигнал, громкость которого линейно увеличивается от
нуля в начале записи до максимума в конце записи, одновременно в обоих
каналах. Частота сигнала не изменяется на всем протяжении записи
"""

import sys
import os
from os.path import abspath

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian


def generate_signal(
        frequency_hz: int,
        duration_s: float,
        sample_rate: int,
        save_file: bool = False,
        filename: str = "generated.mp3"
):
    """
    Функция, которая генерирует и записывает в файл то, что дано по варианту
    :param save_file: флаг сохранения звука в файл
    :param frequency_hz: частота звука (Гц)
    :param duration_s: продолжительность звука (сек)
    :param sample_rate: частота дискретизации (Гц)
    :param filename: имя файла для сохранения
    :return:
    """
    N: int = int(round(sample_rate * duration_s))
    t = np.arange(N) / sample_rate

    sin_signal = np.sin(2 * np.pi * frequency_hz * t)
    loudness = np.linspace(0, 1, N, dtype=float)
    mono_signal = sin_signal * loudness

    stereo_signal = np.vstack((mono_signal, mono_signal))

    # Нормировка – это обязательно всегда!:
    norm = np.max(np.abs(stereo_signal))
    if norm != 0:
        stereo_signal = stereo_signal / norm
    if save_file:
        filename = "result/" + filename
        sf.write(filename, stereo_signal.T, sample_rate, format="mp3")
        print(f"Saved: {abspath(filename)} (sr={sample_rate}, duration={duration_s}s)")
    return stereo_signal


def analyze_generated_audio(filename: str):
    """
    Анализ аудиофайла + сохранение всех графиков в папку result/
    """

    os.makedirs("result", exist_ok=True)

    data, Fd = sf.read(filename, always_2d=True)
    model_signal = data.T

    N = model_signal.shape[1]
    T = N / Fd
    t = np.linspace(0, T, N)

    print(f"Loaded: {filename}")
    print(f"Sample rate = {Fd}, duration = {T:.2f}s")

    plt.close('all')

    # Полная волна
    plt.figure(figsize=(8, 4))
    plt.plot(t, model_signal[0, :])
    plt.title("Waveform (full signal)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("result/waveform_full.png", dpi=300)

    # Фрагмент сигнала
    fragment_time = 0.01
    frag_N = int(fragment_time * Fd)

    plt.figure(figsize=(8, 3))
    plt.plot(t[:frag_N], model_signal[0, :frag_N])
    plt.title("Waveform fragment")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("result/waveform_fragment.png", dpi=300)

    # Амплитудный спектр

    Spectr_input = np.fft.fft(model_signal[0, :])

    AS_input = np.abs(Spectr_input)
    eps = np.max(AS_input) * 1.0e-9
    S_dB_input = 20 * np.log10(AS_input + eps)

    f = np.arange(0, Fd / 2, Fd / N)
    S_dB_input = S_dB_input[:len(f)]

    plt.figure(figsize=(6, 4))
    plt.semilogx(f, S_dB_input)

    plt.grid(True)
    plt.minorticks_on()
    plt.grid(True, which='major', color='#444', linewidth=1)
    plt.grid(True, which='minor', color='#aaa', ls=':')

    Max_dB = np.ceil(np.max(S_dB_input) / 20) * 20
    plt.axis([10, Fd / 2, Max_dB - 100, Max_dB])

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Level (dB)')
    plt.title('Amplitude Spectrum')
    plt.tight_layout()
    plt.savefig("result/amplitude_spectrum.png", dpi=300)

    # Спектрограмма (ShortTimeFFT)

    g_std = 0.01 * Fd
    wind = gaussian(round(2 * g_std), std=g_std, sym=True)

    SFT = ShortTimeFFT(
        wind,
        hop=round(0.1 * Fd),
        fs=Fd,
        scale_to='magnitude'
    )

    Sx = SFT.stft(model_signal[0, :])
    print("STFT memory size:", sys.getsizeof(Sx))

    fig1, ax1 = plt.subplots(figsize=(6, 4))

    t_lo, t_hi = SFT.extent(N)[:2]

    ax1.set_title(
        rf"STFT ({SFT.m_num * SFT.T:g}$\,s$ Gauss window,"
        rf"$\sigma_t={g_std * SFT.T}\,$s)"
    )

    ax1.set(
        xlabel="Time t (s)",
        ylabel="Frequency (Hz)",
        xlim=(t_lo, t_hi)
    )

    im1 = ax1.imshow(
        abs(Sx),
        origin='lower',
        aspect='auto',
        extent=SFT.extent(N),
        cmap='viridis'
    )

    fig1.colorbar(im1, label="Magnitude |Sx(t,f)|")

    ax1.semilogy()
    ax1.set_xlim([0, T])
    ax1.set_ylim([10, Fd / 2])

    ax1.grid(which='major', color='#bbbbbb', linewidth=0.5)
    ax1.grid(which='minor', color='#999999', linestyle=':', linewidth=0.5)
    ax1.minorticks_on()

    plt.tight_layout()
    plt.savefig("result/spectrogram.png", dpi=300)

    plt.show()


def main() -> int:
    # Частота генерируемого звука
    generation_frequency_hz: int = 2000
    # Продолжительность генерируемого звука
    generation_duration_s: float = 10.0
    # Частота дискретизации
    sample_rate_hz: int = 44100
    generate_signal(
        generation_frequency_hz,
        generation_duration_s,
        sample_rate_hz,
        save_file=True)

    analyze_generated_audio("result/generated.mp3")

    return 0


if __name__ == '__main__':
    main()
