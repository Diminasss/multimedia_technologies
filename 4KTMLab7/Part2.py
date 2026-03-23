import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import restoration

def blur_mask_variant7(size):
    k = np.zeros((size, size), dtype=np.float32)
    center_col = size // 2
    for i in range(size):
         k[i, center_col] = i + 1
    k = k / np.sum(k)
    return k

input_image = cv2.imread('текст.png')
input_image_norm = input_image.astype(np.float32) / 255.0

true_mask_size = 15
PSF_true = blur_mask_variant7(true_mask_size)

blur_image = cv2.filter2D(input_image_norm, -1, PSF_true)

mask_size_small_err = 13
PSF_guess_small = blur_mask_variant7(mask_size_small_err)

mask_size_large_err = 25
PSF_guess_large = blur_mask_variant7(mask_size_large_err)

noise_var = 0.005
iterations = 15

def deblur_wiener(blur_img, psf, noise_var):
    deblurred = blur_img.copy()
    for i in range(3):
        deblurred[:, :, i] = restoration.wiener(blur_img[:, :, i], psf, noise_var)
    return np.clip(deblurred, 0., 1.)

def deblur_lucy(blur_img, psf, iterations):
    deblurred = blur_img.copy()
    for i in range(3):
        deblurred[:, :, i] = restoration.richardson_lucy(blur_img[:, :, i], psf, num_iter=iterations)
    return np.clip(deblurred, 0., 1.)

wiener_small_err = deblur_wiener(blur_image, PSF_guess_small, noise_var)
wiener_large_err = deblur_wiener(blur_image, PSF_guess_large, noise_var)

lucy_small_err = deblur_lucy(blur_image, PSF_guess_small, iterations)
lucy_large_err = deblur_lucy(blur_image, PSF_guess_large, iterations)

plt.figure(figsize=(18, 12))

plt.subplot(3, 3, 1)
plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
plt.title('Исходное')
plt.axis('off')

plt.subplot(3, 3, 2)
plt.imshow(cv2.cvtColor(blur_image, cv2.COLOR_BGR2RGB))
plt.title(f'Смазанное (истинная маска size={true_mask_size})')
plt.axis('off')

plt.subplot(3, 3, 4)
plt.imshow(cv2.cvtColor(wiener_small_err, cv2.COLOR_BGR2RGB))
plt.title(f'Винера (неточная size={mask_size_small_err})')
plt.axis('off')

plt.subplot(3, 3, 5)
plt.imshow(cv2.cvtColor(wiener_large_err, cv2.COLOR_BGR2RGB))
plt.title(f'Винера (неточная size={mask_size_large_err})')
plt.axis('off')

plt.subplot(3, 3, 7)
plt.imshow(cv2.cvtColor(lucy_small_err, cv2.COLOR_BGR2RGB))
plt.title(f'Люси-Ричардсона (неточная size={mask_size_small_err})')
plt.axis('off')

plt.subplot(3, 3, 8)
plt.imshow(cv2.cvtColor(lucy_large_err, cv2.COLOR_BGR2RGB))
plt.title(f'Люси-Ричардсона (неточная size={mask_size_large_err})')
plt.axis('off')

plt.tight_layout()
plt.show()
