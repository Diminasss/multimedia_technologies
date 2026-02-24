"""
Лабораторная работа №2
Фурье-фильтрация аудиосигналов средствами Python.

Вариант 17: поднять уровень сигнала в диапазонах 20..250 Гц и 5..10 кГц в 3 раза.

Зависимости: pip install numpy matplotlib soundfile
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

# ─────────────────────────────────────────────────────────────
#  ВАРИАНТ 17
#   Диапазоны усиления: 20–250 Гц  ×3  и  5000–10000 Гц  ×3
#   Остальные частоты: ×1 (без изменений)
# ─────────────────────────────────────────────────────────────

VARIANT = 17

# Диапазоны вида (f_low_Hz, f_high_Hz, gain).
# Частоты вне этих диапазонов остаются с коэффициентом 1.0.
BOOST_BANDS = [
    (20, 250, 3.0),
    (5000, 10000, 3.0),
]

BUTTERWORTH_ORDER = 10  # порядок фильтра Баттерворта

# Пути к файлам
SOUNDS_DIR = 'sounds'
INPUT_FILE = os.path.join(SOUNDS_DIR, 'input.wav')
OUTPUT_FILE = os.path.join(SOUNDS_DIR, 'output.wav')
MODEL_INPUT = os.path.join(SOUNDS_DIR, 'model_input.wav')
MODEL_OUTPUT = os.path.join(SOUNDS_DIR, 'model_output.wav')

# Параметры модельного сигнала
MODEL_SR = 44100  # Гц
MODEL_DURATION = 3.0  # секунды


# ═════════════════════════════════════════════════════════════
#  Ядро: функции Баттерворта и сборка передаточной функции W
# ═════════════════════════════════════════════════════════════

def hz_to_idx(freq_hz: float, N: int, Fd: int) -> int:
    """Перевод частоты в Гц → целочисленный индекс массива БПФ."""
    return round(N * freq_hz / Fd)


def _mirror(W1: np.ndarray) -> np.ndarray:
    """
    Строит «зеркальную» половину W для выполнения условия
    комплексно-сопряжённой симметрии (2.2).
    W2[0] = 0, далее W2[k] = W1[N-k]  (flip без крайнего элемента).
    Длина W2 == длина W1.
    """
    return np.concatenate(([0.0], np.flip(W1)[:-1]))


def _assemble(W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
    """
    Объединяет прямую и зеркальную половины с учётом чётности N,
    чтобы итоговый массив имел длину ровно N (без off-by-one).

    Для чётного N:
        Wn = [W1[0..N/2], W2[N/2+1..N-1]]   -> длина: (N/2+1) + (N/2-1) = N
    Для нечётного N:
        Wn = [W1[0..N//2-1], W2[N//2..N-1]] -> длина: N//2 + (N - N//2) = N
    """
    N = len(W1)
    if N % 2 == 0:  # чётное N
        return np.concatenate((W1[:N // 2 + 1], W2[N // 2 + 1:]))
    else:  # нечётное N
        return np.concatenate((W1[:N // 2], W2[N // 2:]))


def butterworth_bandpass_1d(N: int, n_dn: int, n_up: int, deg: int) -> np.ndarray:
    """
    Одномерная передаточная функция полосового фильтра Баттерворта (формула 2.3).
    Возвращает вещественный массив длиной N, удовлетворяющий условию (2.2).
    """
    n0 = np.sqrt(n_dn * n_up)  # центральная частота (индекс)
    d = (n_up - n_dn) / 2.0  # полуширина полосы (индексы)
    idx = np.arange(N, dtype=float)
    W1 = 1.0 / np.sqrt(1.0 + (np.abs(idx - n0) / d) ** (2 * deg))
    return _assemble(W1, _mirror(W1))


def build_filter_W(N: int, Fd: int) -> np.ndarray:
    """
    Строит передаточную функцию для варианта 17 (два диапазона усиления).

    Идея: W = 1  +  sum_i (gain_i - 1) * BP_i
    Там, где ни один диапазон не активен: W = 1 (сигнал без изменений).
    В i-м диапазоне: W ≈ 1 + (gain_i - 1) * 1 = gain_i.
    Переходы плавные (Баттерворт), поэтому условие (2.2) соблюдается.
    """
    Wn = np.ones(N, dtype=float)  # начало: всё x1

    for f_lo, f_hi, gain in BOOST_BANDS:
        n_dn = hz_to_idx(f_lo, N, Fd)
        n_up = hz_to_idx(f_hi, N, Fd)
        # Защита от вырожденных индексов
        n_dn = max(n_dn, 1)
        n_up = min(n_up, N // 2 - 1)
        if n_dn >= n_up:
            print(f"  [!] Диапазон {f_lo}–{f_hi} Гц слишком узкий для N={N}, пропускаем.")
            continue
        bp = butterworth_bandpass_1d(N, n_dn, n_up, BUTTERWORTH_ORDER)
        Wn += (gain - 1.0) * bp  # добавляем «поднятие»

    # Размножаем на 2 канала стерео: shape (2, N)
    return np.vstack((Wn, Wn))


# ═════════════════════════════════════════════════════════════
#  Фурье-фильтрация
# ═════════════════════════════════════════════════════════════

def fourier_filter(signal_stereo: np.ndarray, Fd: int):
    """
    Фурье-фильтрация стерео-сигнала формы (2, N).
    Возвращает (output, W, Spectr_input, Spectr_output).
    """
    N = signal_stereo.shape[1]

    Spectr_input = np.fft.fft(signal_stereo)  # шаг 1: спектр входного
    W = build_filter_W(N, Fd)  # передаточная функция
    Spectr_output = Spectr_input * W  # шаг 2: формула (2.1)

    # Проверка условия (2.2): мнимая часть обратного БПФ должна быть ≈ 0
    ifft_val = np.fft.ifft(Spectr_output)
    ratio = (np.max(np.abs(np.imag(ifft_val))) /
             (np.max(np.abs(np.real(ifft_val))) + 1e-30))
    status = "OK" if ratio < 1e-4 else "ОШИБКА — условие (2.2) нарушено!"
    print(f"  Проверка (2.2): im/re = {ratio:.2e}  [{status}]")

    output = np.real(ifft_val)  # шаг 3: обратное БПФ
    return output, W, Spectr_input, Spectr_output


# ═════════════════════════════════════════════════════════════
#  Модельный сигнал
# ═════════════════════════════════════════════════════════════

def generate_model_signal():
    """
    Для варианта 17 генерируем смесь четырёх тонов:
      - 80 Гц   — внутри первой полосы усиления (20–250 Гц)
      - 500 Гц  — вне обеих полос (средние частоты)
      - 3000 Гц — вне обеих полос (верхняя середина)
      - 7000 Гц — внутри второй полосы усиления (5–10 кГц)

    После фильтрации 80 Гц и 7000 Гц должны стать втрое громче,
    а 500 Гц и 3000 Гц — остаться без изменений.
    """
    DEMO_FREQS = [
        (80, "в полосе 20–250 Гц   => усилится x3"),
        (500, "вне полос            => без изменений"),
        (3000, "вне полос            => без изменений"),
        (7000, "в полосе 5–10 кГц   => усилится x3"),
    ]

    N = int(MODEL_SR * MODEL_DURATION)
    t = np.linspace(0, MODEL_DURATION, N, endpoint=False)

    signal = np.zeros(N)
    for f, desc in DEMO_FREQS:
        signal += np.sin(2 * np.pi * f * t)
        print(f"    {f:5d} Гц — {desc}")

    signal /= np.max(np.abs(signal) + 1e-9)
    stereo = np.vstack((signal, signal))  # (2, N)
    return stereo, MODEL_SR


# ═════════════════════════════════════════════════════════════
#  Графики
# ═════════════════════════════════════════════════════════════

def _freq_axis(N: int, Fd: int) -> np.ndarray:
    return np.linspace(0, Fd / 2, N // 2, endpoint=False)


def plot_spectra(Fd: int, Sp_in: np.ndarray, Sp_out: np.ndarray, title: str):
    """Рис. 2.1 / 2.4 — амплитудные спектры до и после фильтрации."""
    N = Sp_in.shape[1]
    eps = np.max(np.abs(Sp_in)) * 1e-9

    f = _freq_axis(N, Fd)
    nf = len(f)

    S_in = 20 * np.log10(np.abs(Sp_in[0, :nf]) + eps)
    S_out = 20 * np.log10(np.abs(Sp_out[0, :nf]) + eps)

    Max_A = max(np.max(np.abs(Sp_in)), np.max(np.abs(Sp_out)))
    Max_dB = np.ceil(np.log10(Max_A + 1e-30)) * 20

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogx(f, S_out, color='crimson', lw=1.4, label='Output spectrum')
    ax.semilogx(f, S_in, color='royalblue', lw=1.4, label='Input spectrum')
    ax.set_xlim([10, Fd / 2])
    ax.set_ylim([Max_dB - 120, Max_dB + 5])
    ax.grid(True, which='major', color='#555', lw=0.8)
    ax.grid(True, which='minor', color='#bbb', ls=':')
    ax.minorticks_on()
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Level (dB)')
    ax.set_title(f'Amplitude Spectrums — Input vs Output  [{title}]')
    ax.legend()
    plt.tight_layout()
    return fig


def plot_signals(t: np.ndarray,
                 sig_in: np.ndarray,
                 sig_out: np.ndarray,
                 title: str,
                 zoom_s: float = 0.02):
    """Рис. 2.2 / 2.5 — сигналы во времени + крупный план выхода (Рис. 2.3)."""
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
    axes[1].set_title(f'Output Audio Signal  [{title}]')
    axes[1].grid(True, alpha=0.4)

    # Крупный план — первые zoom_s секунд выходного сигнала (Рис. 2.3)
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
    """Спектрограммы до и после фильтрации."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 7))

    for ax, sig, label, cmap in [
        (axes[0], sig_in[0], f'Input Spectrogram  [{title}]', 'Blues'),
        (axes[1], sig_out[0], f'Output Spectrogram  [{title}]', 'Blues'),
    ]:
        ax.specgram(sig, Fs=Fd, cmap=cmap, NFFT=2048, noverlap=1024, scale='dB')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(label)
        ax.set_ylim([0, min(Fd // 2, 12000)])

    plt.tight_layout()
    return fig


def plot_transfer_function(N: int, Fd: int):
    """График передаточной функции W(f) — дополнительно для наглядности."""
    W = build_filter_W(N, Fd)
    f = _freq_axis(N, Fd)
    nf = len(f)
    Wn = W[0, :nf]

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.semilogx(f, Wn, color='seagreen', lw=1.8)
    ax.axhline(3.0, color='gray', ls='--', lw=0.9, label='x3 (target gain)')
    ax.axhline(1.0, color='gray', ls=':', lw=0.9, label='x1 (no change)')
    ax.set_xlim([10, Fd / 2])
    ax.set_ylim([0, 4.0])
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('|W(f)|')
    ax.set_title(f'Transfer Function — Variant {VARIANT} '
                 f'(boost x3 at 20–250 Hz and 5–10 kHz)')
    ax.grid(True, which='major', color='#555', lw=0.8)
    ax.grid(True, which='minor', color='#bbb', ls=':')
    ax.minorticks_on()
    ax.legend()
    plt.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════
#  Утилиты ввода/вывода
# ═════════════════════════════════════════════════════════════

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


def save_stereo(path: str, signal: np.ndarray, Fd: int):
    """Сохраняет (2, N) в WAV PCM 16-bit. Нормирует при необходимости."""
    mx = np.max(np.abs(signal))
    if mx > 1.0:
        signal = signal / mx
    sf.write(path, signal.T, Fd, subtype='PCM_16')
    print(f"  Сохранён: {path}")


def prelimit(signal: np.ndarray, max_gain: float) -> np.ndarray:
    """
    Превентивно ослабляет сигнал перед усилением,
    чтобы избежать цифрового клиппинга.
    """
    if max_gain > 1.0:
        print(f"  Предварительное ослабление в {max_gain:.1f} раз (защита от перегрузки)")
        return signal / max_gain
    return signal


# ═════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════

def main():
    print(f"=== Лабораторная работа №2 | Вариант {VARIANT} ===")
    for f_lo, f_hi, g in BOOST_BANDS:
        print(f"  Усиление x{g} в диапазоне {f_lo}–{f_hi} Гц")
    print(f"  Порядок фильтра Баттерворта: {BUTTERWORTH_ORDER}")
    os.makedirs(SOUNDS_DIR, exist_ok=True)

    max_gain = max(g for _, _, g in BOOST_BANDS)

    # ── 1. МОДЕЛЬНЫЙ СИГНАЛ ──────────────────────────────────
    print("\n── Генерация модельного сигнала ──")
    print("  Тоны:")
    model_in, model_Fd = generate_model_signal()

    # Ослабляем заранее: после усиления x3 сигнал не выйдет за [−1, 1]
    model_in = prelimit(model_in, max_gain)
    save_stereo(MODEL_INPUT, model_in, model_Fd)

    print("\n── Фурье-фильтрация модельного сигнала ──")
    model_out, _, Sp_m_in, Sp_m_out = fourier_filter(model_in, model_Fd)
    model_out_s = np.vstack((model_out[0], model_out[1]))
    save_stereo(MODEL_OUTPUT, model_out_s, model_Fd)

    N_m = model_in.shape[1]
    t_m = np.linspace(0, MODEL_DURATION, N_m, endpoint=False)

    print("\n── Графики модельного сигнала ──")
    fig_mw = plot_transfer_function(N_m, model_Fd)
    fig_ms = plot_spectra(model_Fd, Sp_m_in, Sp_m_out, "Model Signal")
    fig_mt = plot_signals(t_m, model_in, model_out_s, "Model Signal", zoom_s=0.02)
    fig_msg = plot_spectrogram(model_in, model_out_s, model_Fd, "Model Signal")

    for fig, name in [(fig_mw, 'model_transfer.png'),
                      (fig_ms, 'model_spectra.png'),
                      (fig_mt, 'model_signals.png'),
                      (fig_msg, 'model_spectrogram.png')]:
        fig.savefig(os.path.join(SOUNDS_DIR, name), dpi=150)
        print(f"  Сохранён: sounds/{name}")

    # ── 2. РЕАЛЬНЫЙ АУДИОФАЙЛ ────────────────────────────────
    if not os.path.isfile(INPUT_FILE):
        print(f"\n  Файл {INPUT_FILE!r} не найден — обработка пропущена.")
        print("  Поместите sounds/input.wav и запустите снова.")
    else:
        print(f"\n── Загрузка: {INPUT_FILE} ──")
        real_in, Fd = load_as_stereo(INPUT_FILE)
        N_r = real_in.shape[1]
        print(f"  Fd={Fd} Гц | Длит.={N_r / Fd:.2f} с | Отсчётов={N_r}")

        real_in = prelimit(real_in, max_gain)

        print("\n── Фурье-фильтрация реального сигнала ──")
        real_out, _, Sp_r_in, Sp_r_out = fourier_filter(real_in, Fd)
        real_out_s = np.vstack((real_out[0], real_out[1]))
        save_stereo(OUTPUT_FILE, real_out_s, Fd)

        t_r = np.linspace(0, N_r / Fd, N_r, endpoint=False)

        print("\n── Графики реального сигнала ──")
        fig_rs = plot_spectra(Fd, Sp_r_in, Sp_r_out, "Real Signal")
        fig_rt = plot_signals(t_r, real_in, real_out_s, "Real Signal", zoom_s=0.02)
        fig_rsg = plot_spectrogram(real_in, real_out_s, Fd, "Real Signal")

        for fig, name in [(fig_rs, 'real_spectra.png'),
                          (fig_rt, 'real_signals.png'),
                          (fig_rsg, 'real_spectrogram.png')]:
            fig.savefig(os.path.join(SOUNDS_DIR, name), dpi=150)
            print(f"  Сохранён: sounds/{name}")

    print("\n=== Готово! Отображаем графики... ===")
    plt.show()


if __name__ == '__main__':
    main()
