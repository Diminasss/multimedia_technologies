import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_and_prepare_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Изображение {image_path} не найдено!")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    luminance = lab[:, :, 0].astype(np.float32) / 255.0
    return img, img_rgb, luminance


def compute_spectrum(image):
    f_transform = np.fft.fft2(image)
    f_shifted = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shifted)
    eps = np.max(magnitude) * 1e-9
    magnitude_db = 20 * np.log10(magnitude + eps)
    return f_shifted, magnitude, magnitude_db, eps


def display_with_spectrum(original, filtered, spectrum_db, title, cutoff, filename=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Исходное изображение')
    axes[0].axis('off')
    axes[1].imshow(filtered, cmap='gray')
    axes[1].set_title(f'{title}\nсрез={cutoff}')
    axes[1].axis('off')
    axes[2].imshow(spectrum_db, cmap='jet')
    axes[2].set_title('Амплитудный спектр (дБ)')
    axes[2].axis('off')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()


class FrequencyFilter:
    def __init__(self, shape):
        self.h, self.w = shape
        self.cy, self.cx = self.h // 2, self.w // 2
        y, x = np.ogrid[:self.h, :self.w]
        self.distance = np.sqrt((x - self.cx) ** 2 + (y - self.cy) ** 2)

    def ideal_highpass(self, cutoff):
        mask = np.ones((self.h, self.w))
        mask[self.distance <= cutoff] = 0
        return mask

    def butterworth_highpass(self, cutoff, order):
        d = np.maximum(self.distance, 1e-10)
        mask = (d / cutoff) ** order / np.sqrt(1 + (d / cutoff) ** (2 * order))
        mask[self.cy, self.cx] = 0
        return mask


def main():
    print("=" * 60)
    print("ЛАБОРАТОРНАЯ РАБОТА: Фурье-фильтрация изображений")
    print("=" * 60)

    try:
        img_bgr, img_rgb, luminance = load_and_prepare_image('суслик.jpg')
        print(f"\nИзображение 'суслик.jpg' загружено успешно")
    except FileNotFoundError as e:
        print(f"Ошибка: {e}")
        return

    h, w = luminance.shape
    print(f"Размер изображения: {w}x{h}")
    print(f"Диапазон яркости: [{luminance.min():.3f}, {luminance.max():.3f}]")

    plt.figure(figsize=(8, 6))
    plt.imshow(img_rgb)
    plt.title('Исходное изображение (суслик)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('01_original_suslik.png', dpi=150, bbox_inches='tight')
    plt.show()

    spectrum_shifted, spectrum_mag, spectrum_db, eps = compute_spectrum(luminance)

    plt.figure(figsize=(8, 6))
    plt.imshow(spectrum_db, cmap='jet')
    plt.title('Амплитудный спектр исходного изображения (дБ)')
    plt.axis('off')
    plt.colorbar(label='Амплитуда (дБ)')
    plt.tight_layout()
    plt.savefig('02_original_spectrum.png', dpi=150, bbox_inches='tight')
    plt.show()

    filter_bank = FrequencyFilter(luminance.shape)

    print("\n" + "=" * 60)
    print("ЧАСТЬ 1: Идеальный ФВЧ")
    print("=" * 60)

    ideal_cutoffs = [20, 35, 50, 65]
    ideal_results = []

    for cutoff in ideal_cutoffs:
        print(f"\n--- Идеальный ФВЧ, частота среза: {cutoff} ---")
        mask = filter_bank.ideal_highpass(cutoff)
        spectrum_filtered = spectrum_shifted * mask
        img_recovered = np.fft.ifft2(np.fft.ifftshift(spectrum_filtered))
        imag_ratio = np.max(np.abs(np.imag(img_recovered))) / np.max(np.abs(np.real(img_recovered)))
        print(f"  Отношение мнимой/действительной части: {imag_ratio:.2e}")
        img_filtered = np.abs(img_recovered.real)
        img_norm = cv2.normalize(img_filtered, None, 0, 1, cv2.NORM_MINMAX)
        img_inv = 1 - img_norm
        spectrum_filtered_db = 20 * np.log10(np.abs(spectrum_filtered) + eps)
        display_with_spectrum(img_rgb, img_inv, spectrum_filtered_db, f'Идеальный ФВЧ', cutoff,
                              filename=f'03_ideal_hpf_cutoff_{cutoff}.png')
        ideal_results.append((cutoff, img_inv))

    print("\n" + "=" * 60)
    print("ЧАСТЬ 2: ФВЧ Баттерворта")
    print("=" * 60)

    K = 4
    butter_cutoffs = [10, 25, 40, 45, 55]
    butter_results = []

    for cutoff in butter_cutoffs:
        print(f"\n--- ФВЧ Баттерворта, частота среза: {cutoff}, K={K} ---")
        mask = filter_bank.butterworth_highpass(cutoff, K)
        spectrum_filtered = spectrum_shifted * mask
        img_recovered = np.fft.ifft2(np.fft.ifftshift(spectrum_filtered))
        imag_ratio = np.max(np.abs(np.imag(img_recovered))) / np.max(np.abs(np.real(img_recovered)))
        print(f"  Отношение мнимой/действительной части: {imag_ratio:.2e}")
        img_filtered = np.abs(img_recovered.real)
        img_norm = cv2.normalize(img_filtered, None, 0, 1, cv2.NORM_MINMAX)
        img_inv = 1 - img_norm
        spectrum_filtered_db = 20 * np.log10(np.abs(spectrum_filtered) + eps)
        spectrum_filtered_db[filter_bank.cy, filter_bank.cx] = 20 * np.log10(np.max(spectrum_mag))
        display_with_spectrum(img_rgb, img_inv, spectrum_filtered_db,
                              f'ФВЧ Баттерворта K={K}', cutoff,
                              filename=f'04_butter_hpf_cutoff_{cutoff}.png')
        butter_results.append((cutoff, img_inv))
        if cutoff == 45:
            best_butter_result = img_inv.copy()
            best_cutoff = cutoff

    print("\n" + "=" * 60)
    print("ЧАСТЬ 3: Влияние порядка фильтра K")
    print("=" * 60)

    test_cutoff = 45
    orders = [1, 2, 4, 6]
    order_results = []

    for order in orders:
        print(f"  Порядок K={order}, срез={test_cutoff}")
        mask = filter_bank.butterworth_highpass(test_cutoff, order)
        spectrum_filtered = spectrum_shifted * mask
        img_recovered = np.fft.ifft2(np.fft.ifftshift(spectrum_filtered))
        img_filtered = np.abs(img_recovered.real)
        img_norm = cv2.normalize(img_filtered, None, 0, 1, cv2.NORM_MINMAX)
        img_inv = 1 - img_norm
        order_results.append((order, img_inv))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Исходное изображение')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(order_results[0][1], cmap='gray')
    axes[0, 1].set_title(f'Порядок K={order_results[0][0]}')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(order_results[1][1], cmap='gray')
    axes[0, 2].set_title(f'Порядок K={order_results[1][0]}')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(order_results[2][1], cmap='gray')
    axes[1, 0].set_title(f'Порядок K={order_results[2][0]}')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(order_results[3][1], cmap='gray')
    axes[1, 1].set_title(f'Порядок K={order_results[3][0]}')
    axes[1, 1].axis('off')

    axes[1, 2].axis('off')

    plt.suptitle(f'Влияние порядка фильтра Баттерворта (срез={test_cutoff})', fontsize=14)
    plt.tight_layout()
    plt.savefig('05_butterworth_order_effect.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n" + "=" * 60)
    print("ЧАСТЬ 4: Применение бинаризации")
    print("=" * 60)

    print(f"Используем ФВЧ Баттерворта: срез={best_cutoff}, K=4")

    thresholds = [0.5, 0.65, 0.75, 0.85]
    binary_results = []

    for thr in thresholds:
        _, binary = cv2.threshold(best_butter_result, thr, 1.0, cv2.THRESH_BINARY)
        binary_results.append((thr, binary))
        print(f"  Порог {thr}: применен")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Исходное изображение')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(best_butter_result, cmap='gray')
    axes[0, 1].set_title('ФВЧ Баттерворта (без бинаризации)')
    axes[0, 1].axis('off')

    for i, (thr, binary) in enumerate(binary_results):
        row, col = (i + 2) // 3, (i + 2) % 3
        axes[row, col].imshow(binary, cmap='gray')
        axes[row, col].set_title(f'С бинаризацией (порог={thr})')
        axes[row, col].axis('off')

    plt.suptitle('Эффект бинаризации', fontsize=14)
    plt.tight_layout()
    plt.savefig('06_binarization_effect.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n" + "=" * 60)
    print("ЧАСТЬ 5: Сравнение с методом Кэнни")
    print("=" * 60)

    edges_canny1 = cv2.Canny(img_bgr, 50, 150)
    edges_canny2 = cv2.Canny(img_bgr, 30, 90)
    edges_canny3 = cv2.Canny(img_bgr, 70, 200)

    canny1_inv = 255 - edges_canny1
    canny2_inv = 255 - edges_canny2
    canny3_inv = 255 - edges_canny3

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Исходное изображение')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(best_butter_result, cmap='gray')
    axes[0, 1].set_title(f'ФВЧ Баттерворта\nсрез={best_cutoff}, K=4')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(binary_results[2][1], cmap='gray')
    axes[0, 2].set_title('ФВЧ + бинаризация\nпорог=0.75')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(canny1_inv, cmap='gray')
    axes[1, 0].set_title('Метод Кэнни\nпороги 50/150')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(canny2_inv, cmap='gray')
    axes[1, 1].set_title('Метод Кэнни\nпороги 30/90')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(canny3_inv, cmap='gray')
    axes[1, 2].set_title('Метод Кэнни\nпороги 70/200')
    axes[1, 2].axis('off')

    plt.suptitle('Сравнение методов выделения границ', fontsize=14)
    plt.tight_layout()
    plt.savefig('07_final_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
