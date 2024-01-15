import cv2
import matplotlib.pyplot as plt 
import numpy as np


img = cv2.imread('shape.png')
cv2.waitKey(0)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#appplying canny

edges = cv2.Canny(img,100,200,3,L2gradient = True)
cv2.waitKey(0)

#finding contours
contours, hierarchy = cv2.findContours(edges,
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)



# Lists to store the lengths of inner contours (lines)
inner_contour_lengths = []
# Loop through the contours and hierarchy
for i, contour in enumerate(contours):
    # Get the parent index from the hierarchy
    parent_idx = hierarchy[0][i][3]
    
    if parent_idx != -1:
        # Inner contour, line
        perimeter = cv2.arcLength(contour, True)
        inner_contour_lengths.append((perimeter, i))  # Store length and index

# Sort the inner contour lengths in ascending order
inner_contour_lengths.sort()

# Assign numbers to the lines based on their lengths
line_numbers = {length_index[1]: i + 1 for i, length_index in enumerate(inner_contour_lengths)}

# Draw and label the lines for the four contours with lowest lengths
for length, index in inner_contour_lengths[:4]:  # Only the first four contours
    contour = contours[index]
    cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)  # Green color
    number = line_numbers[index]
    cv2.putText(img, str(number), tuple(contour[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Red color

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

cv2.imshow('Canny Edges After Contouring', edges)
cv2.waitKey(0)


print("Number of Contours found = " + str(len(contours)))

# Draw all contours
# -1 signifies drawing all contours
cv2.drawContours(img, contours, -1, (255, 0, 0), 3)

cv2.imshow('Contours', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# plt.figure()
# plt.title("Rectangle")
# plt.imshow(edges,cmap='gray')
# plt.show()
