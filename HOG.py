#URL: https://www.thepythoncode.com/article/hog-feature-extraction-in-python

from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
import matplotlib.pyplot as plt

img = imread('ast.jpg')
plt.axis("off")
plt.imshow(img)
print(img.shape)

resized_img = resize(img, (128*2, 64*2))
plt.axis("off")
plt.imshow(resized_img)
print(resized_img.shape)

fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                	cells_per_block=(2, 2), visualize=True, multichannel=True)

plt.axis("off")
plt.imshow(hog_image, cmap="gray")
