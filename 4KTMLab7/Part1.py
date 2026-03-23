import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import restoration


def blur_mask_variant7(size):
    if size % 2 == 0:
        size += 1
    k = np.zeros((size, size), dtype=np.float32)
    center_col = size // 2
    for i in range(size):
        k[i, center_col] = i + 1
    k = k / np.sum(k)
    return k


input_image = cv2.imread('текст.png')
if input_image is None:
    print("Ошибка: Не удалось загрузить 'текст.png'. Убедитесь, что файл находится в папке со скриптом.")
    exit()

input_image_norm = input_image.astype(np.float32) / 255.0
mask_size = 15
PSF_blur = blur_mask_variant7(mask_size)
print("Маска искажений (Вариант 7):")
print(PSF_blur)
blur_image = cv2.filter2D(input_image_norm, -1, PSF_blur)
noise_var = 0.005
image_deblur_wiener = blur_image.copy()

for i in range(3):
    image_deblur_wiener[:, :, i] = restoration.wiener(blur_image[:, :, i], PSF_blur, noise_var)

image_deblur_wiener = np.clip(image_deblur_wiener, 0., 1.)
iterations = 15
image_deblur_lucy = blur_image.copy()

for i in range(3):
    image_deblur_lucy[:, :, i] = restoration.richardson_lucy(blur_image[:, :, i], PSF_blur, num_iter=iterations)

image_deblur_lucy = np.clip(image_deblur_lucy, 0., 1.)
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
plt.title('Исходное изображение')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(blur_image, cv2.COLOR_BGR2RGB))
plt.title(f'Смазанное (верт. смаз, вар.7, size={mask_size})')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(image_deblur_wiener, cv2.COLOR_BGR2RGB))
plt.title(f'Винера (точная маска, noise_var={noise_var})')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(image_deblur_lucy, cv2.COLOR_BGR2RGB))
plt.title(f'Люси-Ричардсона (точная маска, итер={iterations})')
plt.axis('off')

plt.tight_layout()
plt.show()
cv2.imwrite('blurred_variant7.jpg', (blur_image * 255).astype(np.uint8))
blurred_from_jpeg = cv2.imread('blurred_variant7.jpg').astype(np.float32) / 255.0

image_recovered_from_jpeg = blurred_from_jpeg.copy()
for i in range(3):
    image_recovered_from_jpeg[:, :, i] = restoration.wiener(blurred_from_jpeg[:, :, i], PSF_blur, noise_var)
image_recovered_from_jpeg = np.clip(image_recovered_from_jpeg, 0., 1.)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(blurred_from_jpeg, cv2.COLOR_BGR2RGB))
plt.title('Смазанное изображение из JPEG')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(image_recovered_from_jpeg, cv2.COLOR_BGR2RGB))
plt.title('Восстановление после JPEG сжатия')
plt.axis('off')
plt.tight_layout()
plt.show()
