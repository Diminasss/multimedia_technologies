"""
Лабораторная работа №5. Вариант 7.
Изучение методов управления контрастностью изображений.
Исследование влияния параметра clip_limit функции
exposure.equalize_adapthist() на результаты обработки.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from skimage import io, exposure
from skimage.color import rgb2lab, lab2rgb

IMAGE_PATH = r"input_image.jpg"

OUTPUT_DIR = "results"  # корневая папка для результатов

GAMMA = 0.1  # < 1 светлее, > 1 темнее
NBINS = 64  # для equalize_hist (степень двойки)
KERNEL = (16, 16)  # kernel_size для equalize_adapthist
CLIP_BASE = 0.02  # clip_limit для части 1

CLIP_VALUES = [0.005, 0.01, 0.05, 0.15, 0.50]  # значения для части 2


# Вспомогательные функции
def load_image(path: str):
    image = io.imread(path)
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    return image


def get_luminance(image):
    """Нормированная яркость L [0,1] + LAB-массив для обратного преобразования."""
    if image.ndim == 2:
        L = image.astype(np.float32) / 255.0 if image.max() > 1 else image.astype(np.float32)
        return L, None
    LAB = rgb2lab(image)
    L = (LAB[:, :, 0] / 100.0).astype(np.float32)
    return L, LAB


def rebuild_color(L_out, LAB):
    """Подставляет обработанную яркость обратно в цветное изображение."""
    if LAB is None:
        return L_out
    LAB_out = LAB.copy()
    LAB_out[:, :, 0] = L_out * 100.0
    return lab2rgb(LAB_out)


def save_card(img, L, title: str, path: str):
    """
    Одна карточка: слева фото в полном разрешении, справа гистограмма.
    """
    h, w = img.shape[:2]
    aspect = h / w

    card_w = 14
    img_w = card_w * 0.62
    hist_w = card_w * 0.38
    fig_h = max(img_w * aspect, 4)

    fig = plt.figure(figsize=(card_w, fig_h))
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)

    gs = fig.add_gridspec(1, 2, width_ratios=[img_w, hist_w], wspace=0.06)

    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(img, cmap='gray' if img.ndim == 2 else None)
    ax_img.axis('off')

    ax_hist = fig.add_subplot(gs[1])
    histogram, bin_edges = np.histogram(L, bins=256, range=(0.0, 1.0))
    ax_hist.bar(bin_edges[:-1], histogram, width=1 / 128,
                color='steelblue', edgecolor='none')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_xlabel("grayscale value", fontsize=10)
    ax_hist.set_ylabel("pixel count", fontsize=10)
    ax_hist.set_title("Гистограмма яркостей", fontsize=11)
    ax_hist.ticklabel_format(axis='y', style='sci', scilimits=(0, 4))
    ax_hist.grid(True, linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {path}")


# Основная программа
def main():
    part1_dir = os.path.join(OUTPUT_DIR, "part1_methods")
    part2_dir = os.path.join(OUTPUT_DIR, "part2_clip_limit")
    os.makedirs(part1_dir, exist_ok=True)
    os.makedirs(part2_dir, exist_ok=True)

    print(f"\nЗагружаем: {IMAGE_PATH}")
    image = load_image(IMAGE_PATH)
    L_orig, LAB = get_luminance(image)
    stem = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
    print(f"  Размер: {image.shape}, dtype: {image.dtype}")

    def p1(name):
        return os.path.join(part1_dir, name)

    def p2(name):
        return os.path.join(part2_dir, name)

    # ЧАСТЬ 1. Четыре метода контрастирования
    print("\n=== ЧАСТЬ 1: четыре метода ===")

    save_card(image, L_orig,
              "Исходное изображение",
              p1(f"{stem}_0_original.jpg"))

    # 1. rescale_intensity
    low, high = float(np.min(L_orig)), float(np.max(L_orig))
    L_rsc = exposure.rescale_intensity(
        L_orig, in_range=(low, high), out_range=(0., 1.)
    ).astype(np.float32)
    img_rsc = rebuild_color(L_rsc, LAB)
    save_card(img_rsc, L_rsc,
              f"rescale_intensity   [{low:.3f}, {high:.3f}] → [0, 1]",
              p1(f"{stem}_1_rescale.jpg"))
    print("  1) rescale_intensity — готово")

    # 2. adjust_gamma
    L_gam = exposure.adjust_gamma(L_rsc, gamma=GAMMA).astype(np.float32)
    img_gam = rebuild_color(L_gam, LAB)
    save_card(img_gam, L_gam,
              f"adjust_gamma   γ = {GAMMA}   (после rescale)",
              p1(f"{stem}_2_gamma.jpg"))
    print(f"  2) adjust_gamma (γ={GAMMA}) — готово")

    # 3. equalize_hist
    L_eq = exposure.equalize_hist(L_orig, nbins=NBINS).astype(np.float32)
    img_eq = rebuild_color(L_eq, LAB)
    save_card(img_eq, L_eq,
              f"equalize_hist   nbins = {NBINS}",
              p1(f"{stem}_3_equalize_hist.jpg"))
    print(f"  3) equalize_hist (nbins={NBINS}) — готово")

    # 4. equalize_adapthist
    L_adp = exposure.equalize_adapthist(
        L_orig, kernel_size=KERNEL, clip_limit=CLIP_BASE
    ).astype(np.float32)
    img_adp = rebuild_color(L_adp, LAB)
    save_card(img_adp, L_adp,
              f"equalize_adapthist   kernel={KERNEL},  clip_limit={CLIP_BASE}",
              p1(f"{stem}_4_adapthist.jpg"))
    print(f"  4) equalize_adapthist (C={CLIP_BASE}) — готово")

    # ЧАСТЬ 2. Исследование влияния clip_limit
    print("\n=== ЧАСТЬ 2: исследование clip_limit ===")

    save_card(image, L_orig,
              "Исходное (для сравнения)",
              p2(f"{stem}_0_original.jpg"))

    for C in CLIP_VALUES:
        L_c = exposure.equalize_adapthist(
            L_orig, kernel_size=KERNEL, clip_limit=C
        ).astype(np.float32)
        img_c = rebuild_color(L_c, LAB)
        c_str = str(C).replace(".", "p")
        save_card(img_c, L_c,
                  f"equalize_adapthist   kernel={KERNEL},  clip_limit = {C}",
                  p2(f"{stem}_C{c_str}.jpg"))
        print(f"  clip_limit = {C} — готово")

    print(f"\nГотово!")
    print(f"  Часть 1 → {os.path.abspath(part1_dir)}")
    print(f"  Часть 2 → {os.path.abspath(part2_dir)}")


if __name__ == "__main__":
    main()
