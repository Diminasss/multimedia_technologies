"""
Лабораторная работа №4.
Тема: Применение формирующих фильтров для создания шумоподобных аудиосигналов
Имитация звука морского прибоя

Структура программы:
  - три формирующих фильтра (ФНЧ + полосовой + медленный ФНЧ)
  - периодический конверт наката волн (быстрый фронт + экспоненциальный спад)
  - стерео-смещение огибающей между каналами
  - тихий фоновый шум дальних волн
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import soundfile as sf

# ============================================================
# Параметры
# ============================================================
Fd = 44100  # частота дискретизации, Гц
T = 20  # длительность, с
N = round(Fd * T)
t = np.linspace(0, T, N)

# ============================================================
# Источники белого шума (равномерное распределение, как в методичке)
# ============================================================
x1 = np.random.uniform(-1.0, 1.0, (2, N))  # для тела волны (ФНЧ)
x2 = np.random.uniform(-1.0, 1.0, (2, N))  # для шипения пены (полосовой)
x3 = np.random.uniform(-1.0, 1.0, (2, N))  # для случайных вариаций высоты волн
x_bg = np.random.uniform(-1.0, 1.0, (2, N))  # фоновый шум дальних волн

# ============================================================
# Формирующие фильтры
# ============================================================

# 1. ФНЧ — «тело» волны: глухой низкочастотный шум (до 180 Гц, порядок 4)
sos_lp = signal.butter(4,
                       Wn=180 / (Fd / 2),
                       btype='lowpass', output='sos')

# 2. Полосовой — «шипение» пены при накате (1000–4000 Гц, порядок 3)
sos_bp = signal.butter(3,
                       Wn=(1000 / (Fd / 2), 4000 / (Fd / 2)),
                       btype='bandpass', output='sos')

# 3. Очень медленный ФНЧ — вариации «высоты» каждой волны
sos_slow = signal.butter(2,
                         Wn=0.05 / (Fd / 2),
                         btype='lowpass', output='sos')

# 4. Фоновый ФНЧ — отдалённые волны, тихий постоянный шум (до 600 Гц)
sos_bg = signal.butter(2,
                       Wn=600 / (Fd / 2),
                       btype='lowpass', output='sos')

# ============================================================
# Фильтрация
# ============================================================
y_body = signal.sosfilt(sos_lp, x1)
y_foam = signal.sosfilt(sos_bp, x2)
y_slow = signal.sosfilt(sos_slow, x3)
y_bg = signal.sosfilt(sos_bg, x_bg)

# ============================================================
# Периодический конверт наката волн
# (аналог y2[:,n] = exp(-6*(n%M)/M) из листинга 4.2 методички,
#  с добавлением нарастающего фронта для реалистичности)
# ============================================================
wave_period_sec = 5.0
M = int(Fd * wave_period_sec)

envelope = np.zeros((2, N))
for n in range(N):
    phase = (n % M) / M

    # Первые 20% периода — нарастающий фронт (накат волны)
    if phase < 0.20:
        amp = phase / 0.20
    else:
        # Экспоненциальный спад (откат воды, как в листинге 4.2)
        amp = np.exp(-5.0 * (phase - 0.20) / 0.80)
    envelope[0, n] = amp

    # Правый канал — смещён на 30% периода (стерео-эффект)
    phase_r = ((n + int(0.30 * M)) % M) / M
    if phase_r < 0.20:
        amp_r = phase_r / 0.20
    else:
        amp_r = np.exp(-5.0 * (phase_r - 0.20) / 0.80)
    envelope[1, n] = amp_r

# ============================================================
# Случайные вариации «высоты» волн (нестационарность, как в природе)
# y_slow — медленный случайный процесс, аналог y2 из листинга 4.1
# ============================================================
wave_height = 1.0 + 0.45 * y_slow

# ============================================================
# Микширование (по аналогии с листингом 4.3)
# ============================================================
# -------------------------------------------------------
y = (y_body * envelope * wave_height * 0.70 +
     y_foam * envelope * wave_height * 0.30)
y += y_bg * 0.07  # тихий фоновый шум
# -------------------------------------------------------

# Нормировка
Norm = np.max(np.abs(y))
if Norm != 0:
    y = y / Norm

# ============================================================
# Графики
# ============================================================

# Амплитудный спектр
Spectr_y = np.fft.fft(y)
AS_y = np.abs(Spectr_y)
eps = np.max(AS_y) * 1.0e-9
S_dB_y = 20 * np.log10(AS_y + eps)
f = np.arange(0, Fd / 2, Fd / N)
S_dB_y = S_dB_y[:, :len(f)]

plt.figure(figsize=(8, 4))
plt.semilogx(f, S_dB_y[0, :], color='steelblue', label='Левый канал')
plt.semilogx(f, S_dB_y[1, :], color='coral', label='Правый канал', alpha=0.75)
plt.minorticks_on()
plt.grid(True, which='major', color='#444', linewidth=1)
plt.grid(True, which='minor', color='#aaa', ls=':')
Max_dB = np.ceil(np.log10(np.max(np.abs(Spectr_y)))) * 20
plt.axis([10, Fd / 2, Max_dB - 80, Max_dB])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Level (dB)')
plt.title('Amplitude Spectrum — Ocean Waves')
plt.legend()
plt.tight_layout()
plt.savefig('lab4_spectrum.png', dpi=150)
plt.show()

# Временной сигнал
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(t, y[0, :], color='steelblue', linewidth=0.3)
plt.xlim([0, T])
plt.xlabel('Time (s)');
plt.ylabel('Amplitude')
plt.title('Left channel — Ocean Waves')

plt.subplot(2, 1, 2)
plt.plot(t, y[1, :], color='coral', linewidth=0.3)
plt.xlim([0, T])
plt.xlabel('Time (s)');
plt.ylabel('Amplitude')
plt.title('Right channel — Ocean Waves')
plt.tight_layout()
plt.savefig('lab4_waveform.png', dpi=150)
plt.show()

# Конверт огибающей
plt.figure(figsize=(10, 3))
plt.plot(t, envelope[0, :], color='steelblue', linewidth=0.8, label='Левый канал')
plt.plot(t, envelope[1, :], color='coral', linewidth=0.8, label='Правый канал', alpha=0.8)
plt.xlim([0, T])
plt.xlabel('Time (s)');
plt.ylabel('Envelope')
plt.title('Конверт периодического наката волн (период 5 с)')
plt.legend();
plt.grid(True)
plt.tight_layout()
plt.savefig('lab4_envelope.png', dpi=150)
plt.show()

# ============================================================
# Запись аудиофайла через soundfile
# ============================================================
sf.write('lab4_ocean_waves.wav', np.transpose(y), Fd)

print(f"  Аудио:   lab4_ocean_waves.wav  ({T} с, стерео, {Fd} Гц)")
print("  Графики: lab4_spectrum.png, lab4_waveform.png, lab4_envelope.png")
