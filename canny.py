# Canny Edge Detection Algorithm
# Cassandra Hopkins

# Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import cv2
from scipy import ndimage

###############################################
# METHOD 1: CANNY IMPLEMENTATION FROM SCRATCH
###############################################

# Helper Functions
def visualize(imgs, m, n):
    """
        Visualize images with the matplotlib library
    """
    plt.figure(figsize=(20, 40))
    for i, img in enumerate(imgs):
        plt_idx = i+1
        plt.subplot(m, n, plt_idx)
        plt.imshow(img, cmap='gray')
    plt.show()

# FROM CHATGPT
# 1: Greyscale conversion
def greyscale(image):
    # If image is colour
    if image.shape[2] == 3:
    # Convert the RGB image to greyscale
        return np.dot(image[..., :3], [0.299, 0.587, 0.114])
    else:
    # If the image is already greyscale, use it directly
        return image

# 2: Noise reduction

# 2.1: Create Gaussian kernel
def gaussian_kernel2(kernel_size, sigma):
    # new kernel size equals floor division of the inputted kernel size
    center = int(kernel_size) // 2
    # creates a meshgrid with the dimensions of [kernel size, kernel size]
    # for size = 5, goes -2,-1,0,1,2 (last term is not inclusive)
    # standardized (about 0)
    # x, y = np.mgrid[-center:center+1, -center:center+1]
    #x, y = np.mgrid[0:kernel_size, 0:kernel_size]
    normal = 1 / (2.0 * np.pi * sigma**2)
    gauss = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    print ("sum: ", np.sum(gauss))
            
    return gauss

def gaussian_kernel(kernel_size, sigma):
    gauss = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    test = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    test2 = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    for x in range(kernel_size):
        for y in range(kernel_size):
            i = x - center 
            j = y - center
            test[x,y] = i
            test2[x,y] = j
            gauss[x,y] = np.exp(-(((i)**2 + (j)**2) / (2.0*sigma**2))) * (1 / (2.0 * np.pi * sigma**2))
    print(test)
    print(test2)
    print(gauss)
    return gauss 

# # 2.2: Perform convolution with greyscale image and gaussian kernel
def convolution(image, kernel):
    vert_strides = image.shape[0] -  kernel.shape[0] + 1
    hori_strides = image.shape[1]  - kernel.shape[1] + 1

    padding = kernel.shape[0] // 2
    padded_image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')
    result = np.zeros((image.shape[0], image.shape[1]), dtype=np.int32)

    for i in range(vert_strides):
        for j in range(hori_strides):
            image_region = padded_image[i:i+kernel.shape[1], j:j+kernel.shape[0]]
            result[i,j] = np.sum(np.multiply(image_region, kernel))
    return result
            

# 3: Gradient calculation

def gradient(image):
    # Sobel filters
    g_x = np.array(
          [[1,0,-1], 
           [2, 0, -2],
           [1, 0 ,-1]], np.int32)
    g_y = np.array(
          [[-1, -2, -1],
           [0, 0, 0],
           [1, 2, 1]], np.int32)
    # Gradient in x and y directions
    grad_x = convolution(image, g_x)
    grad_y = convolution(image, g_y)
    # Gradient magnitude
    G = np.sqrt((grad_x ** 2.0)+(grad_y ** 2.0))
    # Gradient direction
    theta = np.arctan(grad_y / grad_x)
    return G, theta
        
# 4: Non-maximum suppression

def NMS(I, direction):
    height, width = I.shape
    result = np.zeros((height, width), dtype=np.int32)
    # Convert from radians to positive degrees
    angle = (direction * (180/np.pi) + 180)
    
    for i in range(1, height-1):
         for j in range(1, width-1):
            left = 0 
            right = 0
            # Horizontal gradient
            if ((0 <= angle[i,j] < 22.5) or 
            (337.5 <= angle[i,j] <= 360) or 
            (157.5 <= angle[i,j] <= 202.5)):
                left = I[i,j-1]
                right = I[i,j+1]
            # Diagonal gradient
            elif((22.5 <= angle[i,j] < 67.5) or
            (202.5 <= angle[i,j] < 247.5)):
                left = I[i-1, j-1]
                right = I[i+1,j+1]
            # Vertical gradient
            elif((67.5 <= angle[i,j] < 112.5) or
            (247.5 <= angle[i,j]< 292.5)):
                left = I[i-1, j]
                right = I[i+1, j]
            # Anti-diagonal gradient
            elif((112.2 <= angle[i,j] < 157.5) or
            (292.5 <= angle[i,j]< 337.5)):
                left = I[i+1, j-1]
                right = I[i-1,j+1] 
                
            if (I[i,j] > left) and (I[i,j] > right):
                result[i,j] = 255
            else:
                result[i,j] = 0           
    return result           

# 5: Double Thresholding and hysteresis

def double_threshold(G, low, high):
    height, width = G.shape
    Z = np.zeros((height, width), dtype=np.int32) 
      
    for i in range(1,height-1):
        for j in range(1,width-1):
            # Strong edge
            if (G[i,j] >= high):
                Z[i,j] = 255
            if (G[i,j] <= low):
                Z[i,j] = 0
            if ((G[i,j] < high) and (G[i,j] > low)):
                Z[i,j] = hysteresis(G, high)
    return Z
            
    
def hysteresis(G, high):
    height, width = G.shape
       
    for i in range(1,height-1):
        for j in range(1,width-1):
            strong = G[i,j] >= high
            if ((G[i+1, j-1] == strong) or (G[i+1, j] == strong) or
                (G[i+1, j+1] == strong) or (G[i, j-1] == strong) or
                (G[i, j+1] == strong) or (G[i-1, j-1] == strong) or
                (G[i-1, j] == strong) or (G[i-1, j+1] == strong)):
                return 255
            else:
                return 0

# 3D array where each value is a uint8 in the range 0-255 representing the rgb colour scale
# image has three dimensions, height, width, channels, 
# where channels represent the color channels (e.g., Red, Green, Blue for an RGB image)
image = mpimg.imread("tree.jpeg")

#1: Greyscale   
gray_image = greyscale(image)

#2: Blur
image_blurred = convolution(gray_image, gaussian_kernel(5, 1))

#3: Gradient
G, theta = gradient(image_blurred)
image_NMS = NMS(G, theta)

#4 and 5: Thresholding and hysteresis
image_final = double_threshold(G, 50, 150)

plt.figure(1) 
plt.subplot(1,1,1)
plt.imshow(image_final, cmap='gray')
plt.title("Resulting Image - Implementation from Scratch")

plt.figure(2)
plt.subplot(2,2,1)
plt.imshow(image_blurred, cmap='gray')
plt.title("Blurred Greyscale Image (1)")

plt.subplot(2,2,2)
plt.imshow(G, cmap='gray')
plt.title("Gradient Magnitude (2)")

plt.subplot(2,2,3)
plt.imshow(theta, cmap='gray')
plt.title("Gradient Direction (3)")

plt.subplot(2,2,4)
plt.imshow(image_NMS, cmap='gray')
plt.title("NMS (4)")

plt.show()

  
###############################################
# METHOD 2: USING OPENCV
###############################################

# Step 1: Read the image
image3 = cv2.imread('tree.jpeg', cv2.IMREAD_GRAYSCALE)

# Step 2: Apply Gaussian blur to the image
blur = cv2.GaussianBlur(image3, (5, 5), 0)

sobelx = cv2.Sobel(src=blur, ddepth=cv2.CV_64F, dx= 1, dy=0, ksize=3)
sobely = cv2.Sobel(src=blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
sobelxy = cv2.Sobel(src=blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=3)

# Step 3: Apply Canny edge detection
edges = cv2.Canny(blur, 50, 150)

# img_list = [image2, edges]
# visualize(img_list, 2, 1)

# plt.subplot(1,1,1)
# s = np.sqrt(sobelx**2.0+sobely**2.0)
# plt.imshow(edges, cmap='gray')
# plt.title("Resulting Image - OpenCV")
# plt.show()