# %%
# Exercise 6 - Image Processing with OpenCV
import cv2

# %%
file_path = 'images/Lenna_(test_image).png'
# 1. Load image
lenna_original = cv2.imread(file_path)

# %%
# 2. Display image
cv2.imshow("Original Lenna Image", lenna_original)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# 3. Reading an image as grayscale image
lenna_grayscale = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

# Display image
cv2.imshow("Grayscale Lenna Image", lenna_grayscale)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# 4. Colour conversion
lenna_cvt_grayscale = cv2.cvtColor(lenna_original, cv2.COLOR_BGR2GRAY)
lenna_cvt_rgb = cv2.cvtColor(lenna_original, cv2.COLOR_BGR2RGB)

#Display the two images
cv2.imshow("Converted Grayscale", lenna_cvt_grayscale)
cv2.waitKey(0)
cv2.imshow("RGB Lenna", lenna_cvt_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# 5. Image resizing
# (A) Downsize
lenna_downsize = cv2.resize(lenna_original, (256, 256))
# (B) Upsize
lenna_upsize_1 = cv2.resize(lenna_downsize, (760, 760), interpolation=cv2.INTER_NEAREST)
lenna_upsize_2 = cv2.resize(lenna_downsize, (760, 760), interpolation=cv2.INTER_LINEAR)
lenna_upsize_3 = cv2.resize(lenna_downsize, (760, 760), interpolation=cv2.INTER_CUBIC)
cv2.imshow("Original Lenna", lenna_original)
cv2.waitKey(0)
cv2.imshow("Downsized Lenna", lenna_downsize)
cv2.waitKey(0)
cv2.imshow("Nearest Neighbour", lenna_upsize_1)
cv2.waitKey(0)
cv2.imshow("Bilinear", lenna_upsize_2)
cv2.waitKey(0)
cv2.imshow("Bicubic", lenna_upsize_3)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# 6. Save an image
import os

root_path = 'images/'
image_path = os.path.join(root_path, "Lenna downsized.png")
cv2.imwrite(image_path, lenna_downsize)
# %%
# 7. Image blurring
image_path = 'images/Lenna noisy.png'
lenna_noisy = cv2.imread(image_path)

# (A) Mean filter
lenna_meanblur = cv2.blur(lenna_noisy, (7,7))

# (B) Median filter
lenna_medianblur = cv2.medianBlur(lenna_noisy, 7)

# (C) Gaussian filter
lenna_gaussianblur = cv2.GaussianBlur(lenna_noisy, (7,7), 5)

cv2.imshow("Noisy Lenna", lenna_noisy)
cv2.waitKey(0)
cv2.imshow("Mean filter", lenna_meanblur)
cv2.waitKey(0)
cv2.imshow("Median filter", lenna_medianblur)
cv2.waitKey(0)
cv2.imshow("Gaussian filter", lenna_gaussianblur)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# 8. Unsharp masking
# Perform unsharp masking on Lenna grayscale image
# Step 1: Apply blurring
lenna_masking_meanblur = cv2.blur(lenna_grayscale, (5,5))

# Step 2: Use original image minus blurred image
lenna_detail = lenna_grayscale - lenna_masking_meanblur

# Step 3: Use the result from step 2 to add back to original image
lenna_unsharp = lenna_grayscale + lenna_detail

cv2.imshow("Lenna Grayscale", lenna_grayscale)
cv2.waitKey(0)
cv2.imshow("Lenna Grayscale Meanblur", lenna_masking_meanblur)
cv2.waitKey(0)
cv2.imshow("Lenna Detail", lenna_detail)
cv2.waitKey(0)
cv2.imshow("Lenna Unsharp Masking", lenna_unsharp)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# 8.1 Trying unsharp masking with color image
lenna_masking_meanblur = cv2.blur(lenna_original, (5,5))
lenna_detail = lenna_original - lenna_masking_meanblur
lenna_unsharp = lenna_original + lenna_detail

cv2.imshow("Lenna Original", lenna_original)
cv2.waitKey(0)
cv2.imshow("Lenna Original Meanblur", lenna_masking_meanblur)
cv2.waitKey(0)
cv2.imshow("Lenna Detail", lenna_detail)
cv2.waitKey(0)
cv2.imshow("Lenna Unsharp Masking", lenna_unsharp)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# 9. Edge detection
# (A) Sobel filter
sobel_x = cv2.Sobel(lenna_original, ddepth=-1, dx=1, dy=0)
sobel_y = cv2.Sobel(lenna_original, ddepth=-1, dx=0, dy=1)
sobel_xy = cv2.Sobel(lenna_original, ddepth=-1, dx=1, dy=1) # Don't do this
sobel_both = sobel_x + sobel_y

# Display the images
cv2.imshow("Original Lenna", lenna_original)
cv2.waitKey(0)
cv2.imshow("Sobel X", sobel_x)
cv2.waitKey(0)
cv2.imshow("Sobel Y", sobel_y)
cv2.waitKey(0)
cv2.imshow("Sobel both", sobel_both)
cv2.waitKey(0)
cv2.imshow("Sobel xy", sobel_xy)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# (B) Laplacian filter
laplacian = cv2.Laplacian(lenna_original, ddepth=-1, ksize=3)

cv2.imshow("Original Lenna", lenna_original)
cv2.waitKey(0)
cv2.imshow('Laplacian', laplacian)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# (C) Canny edge detection
canny = cv2.Canny(lenna_original, threshold1=50, threshold2=250)

cv2.imshow("Original Lenna", lenna_original)
cv2.waitKey(0)
cv2.imshow('Canny edge detection', canny)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# 10. Histogram equalization
import matplotlib.pyplot as plt
import numpy as np

img_path = 'images/bad contrast.png'
bad_contrast = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
bad_contrast_flatten = bad_contrast.flatten()

equalized = cv2.equalizeHist(bad_contrast)
equalized_flatten = equalized.flatten()

cv2.imshow('Bad contrast', bad_contrast)
cv2.waitKey(0)
cv2.imshow('Histogram Equalized', equalized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Plot the histogram
plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.hist(bad_contrast_flatten, 256, [0,255], color='r')
plt.subplot(1,2,2)
plt.hist(equalized_flatten, 256, [0,255], color='b')
plt.show()

# %%
