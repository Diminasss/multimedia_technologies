"""
Лабораторная работа №3
Фильтрация аудиосигналов во временной области.
Применение рекурсивных цифровых фильтров.

Вариант 17: поднять уровень сигнала в диапазонах 20..250 Гц и 5..10 кГц в 3 раза.

"""

import os
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy import signal

plt.close('all')  # Очистка памяти

#  МЕТОД: signal = input + (K-1)*output_BP
#  Для усиления в N раз:  output = input_signal + (N-1)*BP_filtered
#  Важно: order=1 (фильтр Баттерворта 1-го порядка), иначе фазовые
#  искажения приведут к "провалам" вместо подъёмов АЧХ.


# Диапазоны усиления (f_low Гц, f_high Гц, коэффициент K)
BOOST_BANDS = [
    (20, 250, 3.0),
    (5000, 10000, 3.0),
]

# Порядок фильтра-прототипа Баттерворта
# ВАЖНО: при методе сложения сигналов обязательно order=1!
ORDER = 1

# Пути к файлам
SOUNDS_DIR = 'sounds'
INPUT_FILE = os.path.join(SOUNDS_DIR, 'input.wav')
OUTPUT_FILE = os.path.join(SOUNDS_DIR, 'output_rcf.wav')
MODEL_INPUT = os.path.join(SOUNDS_DIR, 'model_input.wav')
MODEL_OUTPUT = os.path.join(SOUNDS_DIR, 'model_output_rcf.wav')

# Параметры модельного сигнала
MODEL_SR = 44100  # Гц
MODEL_DURATION = 3.0  # секунды


def compute_filter_response(Fd: int) -> tuple:
    """
    Рассчитывает комплексную частотную характеристику H(f)
    составного фильтра, реализованного методом сложения сигналов.

    Для каждого диапазона (f_lo, f_hi, K) добавляется:
        H += (K-1) * H_BP_i

    Итоговый фильтр: H_total = 1 + sum_i (K_i - 1)*H_BP_i

    Возвращает (f, H_total), где f — массив частот в Гц.
    """
    f, H = signal.freqz([0], worN=Fd, whole=False)  # нулевая база
    H = np.zeros_like(f, dtype=complex)  # начинаем с нуля

    for f_lo, f_hi, K in BOOST_BANDS:
        sos = signal.butter(ORDER,
                            Wn=(f_lo / (Fd / 2), f_hi / (Fd / 2)),
                            btype='bandpass', output='sos')
        f, H_bp = signal.sosfreqz(sos, worN=Fd, whole=False, fs=Fd)
        H += (K - 1) * H_bp

    H_total = 1 + H  # добавляение сквозного канала (вх. сигнал без изменений)
    return f, H_total


def rcf_filter(input_stereo: np.ndarray, Fd: int) -> np.ndarray:
    """
    Рекурсивная фильтрация стерео-сигнала формы (2, N).
    output = input + sum_i (K_i - 1) * sosfilt(BP_i, input)
    Возвращает выходной сигнал той же формы (2, N).
    """
    output = input_stereo.copy().astype(float)

    for f_lo, f_hi, K in BOOST_BANDS:
        sos = signal.butter(ORDER,
                            Wn=(f_lo / (Fd / 2), f_hi / (Fd / 2)),
                            btype='bandpass', output='sos')
        filtered_band = signal.sosfilt(sos, input_stereo)  # РЦФ-фильтрация
        output += (K - 1) * filtered_band  # суммирование

    return output


def generate_model_signal():
    """
    Для варианта 17 генерируем смесь четырёх тонов:
      -   80 Гц — внутри первой полосы усиления (20–250 Гц)  => x3
      -  500 Гц — вне обеих полос                            => x1
      - 3000 Гц — вне обеих полос                            => x1
      - 7000 Гц — внутри второй полосы усиления (5–10 кГц)   => x3
    """
    DEMO_FREQS = [
        (80, 'в полосе 20–250 Гц   => усилится x3'),
        (500, 'вне полос            => без изменений'),
        (3000, 'вне полос            => без изменений'),
        (7000, 'в полосе 5–10 кГц   => усилится x3'),
    ]

    N = int(MODEL_SR * MODEL_DURATION)
    t = np.linspace(0, MODEL_DURATION, N, endpoint=False)

    sig = np.zeros(N)
    for f, desc in DEMO_FREQS:
        sig += np.sin(2 * np.pi * f * t)
        print(f'    {f:5d} Гц — {desc}')

    sig /= np.max(np.abs(sig) + 1e-9)
    stereo = np.vstack((sig, sig))  # (2, N)
    return stereo, MODEL_SR


def load_as_stereo(path: str):
    """Загружает WAV, всегда возвращает (2, N) float64 и Fd."""
    raw, Fd = sf.read(path)
    if raw.ndim == 1:
        stereo = np.vstack((raw, raw))
    else:
        stereo = raw.T
        if stereo.shape[0] > 2:
            stereo = stereo[:2]
    return stereo.astype(float), Fd


def save_stereo(path: str, sig: np.ndarray, Fd: int):
    """Сохраняет (2, N) в WAV PCM 16-bit, нормируя при необходимости."""
    mx = np.max(np.abs(sig))
    if mx > 1.0:
        sig = sig / mx
    sf.write(path, sig.T, Fd, subtype='PCM_16')
    print(f'  Сохранён: {path}')


def prelimit(sig: np.ndarray, max_gain: float) -> np.ndarray:
    """
    Превентивно ослабляет сигнал, чтобы после усиления x max_gain
    не возникло цифрового клиппинга (перегрузки).
    """
    if max_gain > 1.0:
        print(f'  Предварительное ослабление в {max_gain:.1f} раз (защита от перегрузки)')
        return sig / max_gain
    return sig


def plot_filter_response(Fd: int):
    """
    Показывает подъём в заданных диапазонах.
    """
    f, H = compute_filter_response(Fd)
    eps = 1e-10
    L = 20 * np.log10(np.abs(H) + eps)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogx(f, L, color='seagreen', lw=1.8)
    ax.axhline(20 * np.log10(3.0), color='gray', ls='--', lw=0.9,
               label=f'x3 = {20 * np.log10(3):.1f} dB (target gain)')
    ax.axhline(0, color='gray', ls=':', lw=0.9, label='x1 = 0 dB (no change)')
    ax.set_xlim([10, Fd / 2])
    ax.set_ylim([-20, 20])
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Level [dB]')
    ax.set_title(f'Digital Filter Frequency Response — Variant 7 '
                 f'(Butterworth order={ORDER}, boost x3 at 20–250 Hz and 5–10 kHz)')
    ax.grid(True, which='both', axis='both')
    ax.minorticks_on()
    ax.grid(True, which='major', color='#444', linewidth=1)
    ax.grid(True, which='minor', color='#aaa', ls=':')
    ax.legend()
    plt.tight_layout()
    return fig


def plot_spectra(Fd: int, sig_in: np.ndarray, sig_out: np.ndarray, title: str):
    """
    амплитудные спектры до и после РЦФ-фильтрации.
    """
    Sp_in = np.fft.fft(sig_in)
    Sp_out = np.fft.fft(sig_out)

    Max_A = max(np.max(np.abs(Sp_in)), np.max(np.abs(Sp_out)))
    eps = Max_A * 1e-9
    Max_dB = np.ceil(np.log10(Max_A + 1e-30)) * 20

    N = sig_in.shape[1]
    f = np.arange(0, Fd / 2, Fd / N)
    nf = len(f)

    S_in = 20 * np.log10(np.abs(Sp_in[0, :nf]) + eps)
    S_out = 20 * np.log10(np.abs(Sp_out[0, :nf]) + eps)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogx(f, S_in, color='royalblue', lw=1.4, label='Input spectrum')
    ax.semilogx(f, S_out, color='crimson', lw=1.4, label='Output spectrum (RCF)')
    ax.set_xlim([10, Fd / 2])
    ax.set_ylim([Max_dB - 120, Max_dB + 5])
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Level (dB)')
    ax.set_title(f'Amplitude Spectrums — Input vs Output  [{title}]')
    ax.grid(True, which='major', color='#555', lw=0.8)
    ax.grid(True, which='minor', color='#bbb', ls=':')
    ax.minorticks_on()
    ax.legend()
    plt.tight_layout()
    return fig


def plot_signals(t: np.ndarray,
                 sig_in: np.ndarray,
                 sig_out: np.ndarray,
                 title: str,
                 zoom_s: float = 0.02):
    """
    сигналы во времени + крупный план выходного.
    """
    fig, axes = plt.subplots(3, 1, figsize=(9, 9))

    axes[0].plot(t, sig_in[0, :], color='royalblue', lw=0.5)
    axes[0].set_xlim([t[0], t[-1]])
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title(f'Input Audio Signal  [{title}]')
    axes[0].grid(True, alpha=0.4)

    axes[1].plot(t, sig_out[0, :], color='crimson', lw=0.5)
    axes[1].set_xlim([t[0], t[-1]])
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title(f'Output Audio Signal (RCF)  [{title}]')
    axes[1].grid(True, alpha=0.4)

    zoom_end = min(zoom_s, t[-1] * 0.1)
    mask = t <= zoom_end
    axes[2].plot(t[mask], sig_out[0, mask], color='darkorange', lw=1.2)
    axes[2].set_xlim([0, zoom_end])
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Amplitude')
    axes[2].set_title(f'Output Signal — Zoom [0 … {zoom_end:.4f} s]  [{title}]')
    axes[2].grid(True, alpha=0.4)

    plt.tight_layout()
    return fig


def plot_spectrogram(sig_in: np.ndarray,
                     sig_out: np.ndarray,
                     Fd: int,
                     title: str):
    """Спектрограммы до и после РЦФ-фильтрации."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 7))

    for ax, sig, label, cmap in [
        (axes[0], sig_in[0], f'Input Spectrogram  [{title}]', 'Blues'),
        (axes[1], sig_out[0], f'Output Spectrogram (RCF)  [{title}]', 'Blues'),
    ]:
        ax.specgram(sig, Fs=Fd, cmap=cmap, NFFT=2048, noverlap=1024, scale='dB')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(label)
        ax.set_ylim([0, min(Fd // 2, 12000)])

    plt.tight_layout()
    return fig


def main():
    print(f'=== Лабораторная работа №3 | Вариант 17 ===')
    print(f'  Метод: РЦФ (рекурсивный цифровой фильтр, семейство Баттерворта)')
    for f_lo, f_hi, K in BOOST_BANDS:
        print(f'  Усиление x{K} в диапазоне {f_lo}–{f_hi} Гц')
    print(f'  Порядок фильтра-прототипа: {ORDER}')
    print()

    os.makedirs(SOUNDS_DIR, exist_ok=True)

    max_gain = max(K for _, _, K in BOOST_BANDS)

    # ── 1. МОДЕЛЬНЫЙ СИГНАЛ ──────────────────────────────────
    print('── Генерация модельного сигнала ──')
    print('  Тоны:')
    model_in, model_Fd = generate_model_signal()

    model_in = prelimit(model_in, max_gain)
    save_stereo(MODEL_INPUT, model_in, model_Fd)

    print('\n── РЦФ-фильтрация модельного сигнала ──')
    model_out = rcf_filter(model_in, model_Fd)
    save_stereo(MODEL_OUTPUT, model_out, model_Fd)

    N_m = model_in.shape[1]
    t_m = np.linspace(0, MODEL_DURATION, N_m, endpoint=False)

    print('\n── Графики модельного сигнала ──')
    fig_mf = plot_filter_response(model_Fd)
    fig_ms = plot_spectra(model_Fd, model_in, model_out, 'Model Signal')
    fig_mt = plot_signals(t_m, model_in, model_out, 'Model Signal', zoom_s=0.02)
    fig_msg = plot_spectrogram(model_in, model_out, model_Fd, 'Model Signal')

    for fig, name in [(fig_mf, 'model_filter_response.png'),
                      (fig_ms, 'model_spectra_rcf.png'),
                      (fig_mt, 'model_signals_rcf.png'),
                      (fig_msg, 'model_spectrogram_rcf.png')]:
        fig.savefig(os.path.join(SOUNDS_DIR, name), dpi=150)
        print(f'  Сохранён: sounds/{name}')

    # ── 2. РЕАЛЬНЫЙ АУДИОФАЙЛ ────────────────────────────────
    if not os.path.isfile(INPUT_FILE):
        print(f'\n  Файл {INPUT_FILE!r} не найден — обработка реального файла пропущена.')
        print('  Поместите sounds/input.wav и запустите снова.')
    else:
        print(f'\n── Загрузка: {INPUT_FILE} ──')
        real_in, Fd = load_as_stereo(INPUT_FILE)
        N_r = real_in.shape[1]
        print(f'  Fd={Fd} Гц | Длит.={N_r / Fd:.2f} с | Отсчётов={N_r}')

        real_in = prelimit(real_in, max_gain)

        print('\n── РЦФ-фильтрация реального сигнала ──')
        real_out = rcf_filter(real_in, Fd)
        save_stereo(OUTPUT_FILE, real_out, Fd)

        t_r = np.linspace(0, N_r / Fd, N_r, endpoint=False)

        print('\n── Графики реального сигнала ──')
        fig_rs = plot_spectra(Fd, real_in, real_out, 'Real Signal')
        fig_rt = plot_signals(t_r, real_in, real_out, 'Real Signal', zoom_s=0.02)
        fig_rsg = plot_spectrogram(real_in, real_out, Fd, 'Real Signal')

        for fig, name in [(fig_rs, 'real_spectra_rcf.png'),
                          (fig_rt, 'real_signals_rcf.png'),
                          (fig_rsg, 'real_spectrogram_rcf.png')]:
            fig.savefig(os.path.join(SOUNDS_DIR, name), dpi=150)
            print(f'  Сохранён: sounds/{name}')

    print('\n=== Готово! Отображаем графики... ===')
    plt.show()


if __name__ == '__main__':
    main()
