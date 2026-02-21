"""
Вариант №7

Синусоидальный сигнал, громкость которого линейно увеличивается от
нуля в начале записи до максимума в конце записи, одновременно в обоих
каналах. Частота сигнала не изменяется на всем протяжении записи
"""
import numpy as np
import soundfile as sf
from os.path import abspath


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


def main() -> int:
    generation_frequency_hz: int = 2000  # Частота генерируемого звука
    generation_duration_s: float = 10.0  # Продолжительность генерируемого звука
    sample_rate_hz: int = 44100  # Частота дискретизации
    stereo_signal = generate_signal(generation_frequency_hz, generation_duration_s, sample_rate_hz, save_file=True)

    return 0


if __name__ == '__main__':
    main()
