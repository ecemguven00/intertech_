import cv2
import numpy as np
from rembg import remove
import easygui
from PIL import Image
from skimage import io,filters,feature
import matplotlib.pyplot as plt
from skimage.filters import roberts, sobel, scharr, prewitt, farid

# Load the original image
img = cv2.imread('inputs/input0.jpg')
#print(type(img))

#Remove the background
output = remove(img)
cv2.imshow('removed image', output)
cv2.waitKey(0)
cv2.imwrite('background.jpg', output)
cv2.destroyAllWindows()
#print(type(output))

# Create boundaries for the blue color
lower_blue = [30, 20, 70]
upper_blue = [180, 255, 255]

# Create NumPy arrays from the boundaries
lower = np.array(lower_blue)
upper = np.array(upper_blue)

# Transform RGB image into HSV
hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)

# Calculate color variance for each pixel
color_variance = np.var(hsv, axis=2)

# Define a threshold for color variance
variance_threshold = 100 # Adjust this threshold based on your needs

# Create a binary mask based on color variance
variance_mask = np.where(color_variance > variance_threshold, 255, 0).astype(np.uint8)

# find the colors within the specified boundaries and apply
# the mask
mask = cv2.inRange(hsv, lower, upper)
output_1 = cv2.bitwise_and(output, output, mask=variance_mask)

# convert the image to grayscale format
img_gray = cv2.cvtColor(output_1, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscaled image', img_gray)
cv2.waitKey(0)
cv2.imwrite('image_gray4.jpg', img_gray)
cv2.destroyAllWindows()

# apply 3*3 gaussian blur
blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)
cv2.imshow('Blurred image', blurred)
cv2.waitKey(0)
cv2.imwrite('image_blurred4.jpg', blurred)
cv2.destroyAllWindows()

# canny edge detection
edge = cv2.Canny(blurred, threshold1=30, threshold2=200)
cv2.imshow('Edge image', edge)
cv2.waitKey(0)
cv2.imwrite('image_edge4.png', edge)
cv2.destroyAllWindows()

# apply threshold
ret,thresh = cv2.threshold(variance_mask,50,255,0)
# morphological

# Define kernel for morphological operations
kernel = np.ones((5,5),np.uint8)

invert = cv2.bitwise_not(thresh)

# erode the image
erosion = cv2.erode(invert, kernel,iterations=1)
cv2.imshow('Erosion image', erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()

# dilate the image 
dilation = cv2.dilate(erosion,kernel,iterations = 1)
cv2.imshow('Dilation image', dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()

# contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours, _ = cv2.findContours(variance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# find all the contour areas
def get_contour_areas(contours):
    all_areas= []
    for cnt in contours:
        area= cv2.contourArea(cnt)
        all_areas.append(area)

    return all_areas

# sort all contour areas and find the largest (expected to be the id card)
sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)
# print(len(sorted_contours))

"""for c in sorted_contours :
    # largest_item= sorted_contours[c]
    # cv2.drawContours(img, largest_item, -1, (255,0,0),2)
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draws a green rectangle
    cv2.waitKey(0)
    cv2.imshow('Largest Object', img)"""

largest_item= sorted_contours[0]
x, y, w, h = cv2.boundingRect(largest_item)
cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draws a green rectangle
# cv2.waitKey(0)
# cv2.imshow('Largest Object', img)

# Target aspect ratio and aspect ratio calculation
rectangle_height = h
rectangle_width = w

# Calculate the aspect ratio of the green rectangle
rectangle_aspect_ratio = rectangle_width / rectangle_height

# Define the expected aspect ratio (9:6)
expected_aspect_ratio = 9 / 6


# Check if the original aspect ratio is approximately equal to the target aspect ratio
if abs(rectangle_aspect_ratio - expected_aspect_ratio) < 0.1:  # Adjust tolerance as needed
    # Display the result
    print("Detected object has the expected aspect ratio.")
    cv2.imshow('Detected ID Card', img)
    
else:
    print("Image does not have the correct aspect ratio. Skipping processing.")
    
cv2.waitKey(0)
cv2.destroyAllWindows()