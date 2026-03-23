import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import restoration
import os

real_image_path = 'смазана.jpg'

if not os.path.exists(real_image_path):
    print(f"ОШИБКА: Файл '{real_image_path}' не найден в текущей директории!")
    print("Текущая рабочая директория:", os.getcwd())
    print("Пожалуйста, убедитесь, что файл 'смазана.jpg' находится в папке со скриптом.")
    exit()

real_image = cv2.imread(real_image_path)
if real_image is None:
    print(
        f"ОШИБКА: Не удалось загрузить '{real_image_path}'. Возможно, файл поврежден или имеет неподдерживаемый формат.")
    exit()

print(f"Размер изображения: {real_image.shape[1]}x{real_image.shape[0]} пикселей")
print(f"Количество каналов: {real_image.shape[2] if len(real_image.shape) > 2 else 1}")

if len(real_image.shape) == 2 or real_image.shape[2] == 1:
    print("Изображение в оттенках серого. Конвертируем в цветное BGR...")
    real_image = cv2.cvtColor(real_image, cv2.COLOR_GRAY2BGR)

real_image_norm = real_image.astype(np.float32) / 255.0


def blur_mask(size, angle):
    if size % 2 == 0:
        size += 1

    k = np.zeros((size, size), dtype=np.float32)
    k[(size - 1) // 2, :] = np.ones(size, dtype=np.float32)

    rotation_matrix = cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), angle, 1.0)
    k = cv2.warpAffine(k, rotation_matrix, (size, size))

    if np.sum(k) > 0:
        k = k / np.sum(k)

    return k


def diagonal_blur_mask(size):
    if size % 2 == 0:
        size += 1

    k = np.zeros((size, size), dtype=np.float32)

    for i in range(size):
        k[i, i] = 1.0

    k = k / np.sum(k)
    return k


print("\n" + "=" * 60)
print("ВИЗУАЛЬНАЯ ОЦЕНКА ПАРАМЕТРОВ СМАЗА")
print("=" * 60)
print("1. Откройте изображение в любом просмотрщике и увеличьте фрагмент,")
print("   где видны отдельные элементы (например, буквы или цифры).")
print("2. Оцените направление смаза (угол в градусах):")
print("   - 0°: горизонтальный смаз (слева направо)")
print("   - 45°: диагональный смаз (слева-направо и сверху-вниз)")
print("   - 90°: вертикальный смаз (сверху вниз)")
print("   - 135°: диагональный смаз (справа-налево и сверху-вниз)")
print("3. Оцените длину смаза (size) - примерно сколько пикселей 'размазана' точка.")
print("   Например, если буква 'растянута' на 10-15 пикселей, size = 11 или 15.")
print("=" * 60)

estimated_size = 11
estimated_angle = 30

use_simple_diagonal = False

print(f"\nИспользуемые параметры:")
print(f"  - Размер маски (size): {estimated_size}")
print(f"  - Угол смаза (angle): {estimated_angle}°")
print(f"  - Тип маски: {'Упрощенная диагональная' if use_simple_diagonal else 'Повернутая линия'}")

if use_simple_diagonal:
    PSF_estimate = diagonal_blur_mask(estimated_size)
else:
    PSF_estimate = blur_mask(estimated_size, estimated_angle)

print("\nМаска искажений (первые 5x5 элементов):")
print(PSF_estimate[:5, :5])

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(PSF_estimate, cmap='hot', interpolation='nearest')
plt.title(f'Маска смаза (size={estimated_size}, angle={estimated_angle}°)')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(real_image, cv2.COLOR_BGR2RGB))
plt.title('Исходное смазанное изображение')
plt.axis('off')
plt.tight_layout()
plt.show()

noise_var_values = [0.005, 0.01, 0.03, 0.05, 0.6]

print("\n" + "=" * 60)
print("ПРИМЕНЕНИЕ ФИЛЬТРА ВИНЕРА С РАЗНЫМИ ПАРАМЕТРАМИ")
print("=" * 60)

plt.figure(figsize=(18, 10))

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(real_image, cv2.COLOR_BGR2RGB))
plt.title('Исходное смазанное изображение')
plt.axis('off')

for idx, nv in enumerate(noise_var_values[:5]):
    deblurred = real_image_norm.copy()

    try:
        for i in range(3):
            deblurred[:, :, i] = restoration.wiener(real_image_norm[:, :, i], PSF_estimate, nv)

        deblurred = np.clip(deblurred, 0., 1.)

        plt.subplot(2, 3, idx + 2)
        plt.imshow(cv2.cvtColor(deblurred, cv2.COLOR_BGR2RGB))
        plt.title(f'Винера, noise_var={nv}')
        plt.axis('off')

        if nv == 0.01:
            best_wiener = (deblurred * 255).astype(np.uint8)
            cv2.imwrite('wiener_best_variant7.jpg', best_wiener)

    except Exception as e:
        print(f"Ошибка при noise_var={nv}: {e}")

plt.tight_layout()
plt.show()

iterations_list = [5, 10, 20, 30, 50]

print("\n" + "=" * 60)
print("ПРИМЕНЕНИЕ МЕТОДА ЛЮСИ-РИЧАРДСОНА С РАЗНЫМ ЧИСЛОМ ИТЕРАЦИЙ")
print("=" * 60)
print("ВНИМАНИЕ: Метод Люси-Ричардсона может работать медленно на больших изображениях.")
print("Если изображение большое, процесс может занять некоторое время...")

plt.figure(figsize=(18, 10))
plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(real_image, cv2.COLOR_BGR2RGB))
plt.title('Исходное')
plt.axis('off')

for idx, it in enumerate(iterations_list[:5]):
    deblurred_lucy = real_image_norm.copy()

    try:
        print(f"  Выполняется итерация {it}...")

        for i in range(3):
            deblurred_lucy[:, :, i] = restoration.richardson_lucy(
                real_image_norm[:, :, i], PSF_estimate, num_iter=it
            )

        deblurred_lucy = np.clip(deblurred_lucy, 0., 1.)

        plt.subplot(2, 3, idx + 2)
        plt.imshow(cv2.cvtColor(deblurred_lucy, cv2.COLOR_BGR2RGB))
        plt.title(f'Люси-Ричардсона, итер={it}')
        plt.axis('off')

        if it == 20:
            best_lucy = (deblurred_lucy * 255).astype(np.uint8)
            cv2.imwrite('lucy_best_variant7.jpg', best_lucy)

    except Exception as e:
        print(f"Ошибка при итерациях {it}: {e}")

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("РЕКОМЕНДАЦИИ ПО ПОДБОРУ ПАРАМЕТРОВ")
print("=" * 60)
print("1. Если результат слишком размытый - увеличьте размер маски (size)")
print("2. Если появились сильные волнистые артефакты - увеличьте noise_var")
print("3. Если метод Люси-Ричардсона создает шум - уменьшите число итераций")
print("4. Попробуйте разные углы с шагом 5-10 градусов")
print("5. Лучшие результаты сохранены как 'wiener_best_variant7.jpg' и 'lucy_best_variant7.jpg'")
print("=" * 60)

print("\nЗадание 3 выполнено!")
