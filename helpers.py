import numpy as np
import matplotlib.pyplot as plt
import cv2


def getUpperandLowerRowBoundary(frame, height, width, threshold = 150, plotImage = False):
    thresholded = np.all(frame > threshold, axis=-1).astype(np.uint8)

    # Convert to a binary image (0s and 1s)
    binary_image = (thresholded * 255).astype(np.uint8)  # To make it a proper binary image
    
    # Find the indices of nonzero pixels
    nonzero_indices = np.nonzero(binary_image)

    # Extract the row indices
    row_indices = nonzero_indices[0]

    if len(row_indices) > 1:
        bottommost_row_index = np.max(row_indices)  # Bottommost nonzero row index
        uppermost_row_index = np.min(row_indices)
    else:
        print("No nonzero pixels found.")
    
    if plotImage:
        plt.imshow(binary_image, cmap='gray')   
        plt.axhline(y=bottommost_row_index, color='red', linewidth=1.5, label="Bottommost Nonzero Row")
        plt.axhline(y=uppermost_row_index, color='red', linewidth=1.5, label="Uppermost Nonzero Row")
        plt.axis('off')
        plt.show()        

    return bottommost_row_index, uppermost_row_index



def find_side_lines(frame, output_path, bottommost_row_index, uppermost_row_index, threshold=100, min_line_length=50, max_line_gap=10, plotEdgeDetectionResult=False):
    """
    Detects and draws lines on an image using the Hough Transform.

    Parameters:
        image_path (str): Path to the input image.
        output_path (str): Path to save the output image with drawn lines.
        threshold (int): Threshold for the Hough Line Transform.
        min_line_length (int): Minimum length of a line. Shorter lines are rejected.
        max_line_gap (int): Maximum allowed gap between line segments to treat them as a single line.

    Returns:
        None
    """
    # Read the input image
    image = frame
    if image is None:
        raise ValueError("Image not found at the specified path.")

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform edge detection
    edges = cv2.Canny(gray, 20, 60, apertureSize=3)
    
    if plotEdgeDetectionResult:
        plt.imshow(edges, cmap='gray')
        plt.title('Edges Detected with Canny')
        plt.axis('off')
        plt.show()
        plt.imsave('edges.png', edges, cmap='gray')
    
    # Use the Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold, minLineLength = min_line_length, maxLineGap = max_line_gap)
    
    y_difference = lines[:, :, 3] - lines[:, :, 1]
    x_difference = (lines[:, :, 2] - lines[:, :, 0]) + 1e-10
    slopeArray = y_difference / x_difference
    
    lowerSlopeBound = 8
    upperSlopeBound = 30
    leftLines = slopeArray[(slopeArray >= lowerSlopeBound) & (slopeArray <= upperSlopeBound)]

    rightLines = slopeArray[(slopeArray <= -lowerSlopeBound) & (slopeArray >= -upperSlopeBound)]

    leftLineIndex = leftLines[int(np.median(leftLines.shape[0])) - 1]
    rightLineIndex = rightLines[int(np.median(rightLines.shape[0])) - 1]

    leftwhere = np.where(slopeArray == leftLineIndex)
    leftLine = lines[leftwhere[0][0]]

    rightwhere = np.where(slopeArray == rightLineIndex)
    rightLine = lines[rightwhere[0][0]]

    # Create a copy of the original image to draw lines
    output_image = image.copy()
    coordinates = list()
    if lines is not None:
        # for line in lines:
        for lineIndex, line in enumerate([leftLine, rightLine]):
            x1, y1, x2, y2 = line[0]
                        
            # Calculate the slope
            if x2 != x1:  # Avoid division by zero
                slope = (y2 - y1) / (x2 - x1)
            else:
                slope = float('inf')  # Vertical line

            if True:
                
                if lineIndex == 0: #this is for left line
                    delta_x = x2 - x1
                    delta_y = y2 - y1
                    
                    # Normalize direction vector
                    length = (delta_x**2 + delta_y**2)**0.5
                    unit_x = delta_x / length
                    unit_y = delta_y / length

                    extension_factor2 = (bottommost_row_index - y2) / unit_y
                    extension_factor1 = (y1 - uppermost_row_index) / unit_y
                    
                    new_x1 = int(x1 - unit_x * extension_factor1)
                    new_y1 = int(y1 - unit_y * extension_factor1)
                    new_x2 = int(x2 + unit_x * extension_factor2)
                    new_y2 = int(y2 + unit_y * extension_factor2)
                    
                    coordinates.append([new_x1, new_y1])
                    coordinates.append([new_x2, new_y2])
                    
                else:
                    delta_x = x2 - x1
                    delta_y = y1 - y2
                    
                    # Normalize direction vector
                    length = (delta_x**2 + delta_y**2)**0.5
                    unit_x = delta_x / length
                    unit_y = delta_y / length

                    extension_factor2 = (y2 - uppermost_row_index) / unit_y
                    extension_factor1 = (bottommost_row_index - y1) / unit_y
                    new_x1 = int(x1 - unit_x * extension_factor1)
                    new_y1 = int(y1 + unit_y * extension_factor1)
                    new_x2 = int(x2 + unit_x * extension_factor2)
                    new_y2 = int(y2 - unit_y * extension_factor2)
                    
                    coordinates.append([new_x1, new_y1])
                    coordinates.append([new_x2, new_y2])

            else:
                pass

    return coordinates


def saveSegmentedImage(frame, height, width, coordinates, imageName):

    # Define the size of the mask image
    image_size = (height, width)

    # Create a black mask
    mask = np.zeros(image_size, dtype=np.uint8)

    # Define the four coordinates of the region (x, y)
    maskCoordinates = np.array([coordinates[0], coordinates[3], coordinates[2], coordinates[1]]) #leftop, rightop, rightbottom, leftbottom
    cv2.fillPoly(mask, [maskCoordinates], 255)
    mask_expanded = np.expand_dims(mask, axis=-1)  # Shape: (3840, 2160, 1)

    masked_image = np.where(mask_expanded == 255, frame, 0)
    return masked_image



def remove_bottom_black(image):
    """
    Removes the bottom black portion of an image.

    :param image: Input image as a NumPy array (H, W, C).
    :return: Cropped image without the black bottom portion.
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Find the non-black pixels
    non_black_row_indices = np.where(gray.max(axis=1) > 0)[0]
    
    if len(non_black_row_indices) == 0:
        # If the entire image is black, return an empty array
        print("The image is completely black.")
        return None
    
    # Get the last row index with non-black pixels
    last_non_black_row = non_black_row_indices[-1]
    
    # Crop the image to remove the black bottom portion
    cropped_image = image[:last_non_black_row + 1, :]
    
    return cropped_image



def checkInputImage(frame, height, width):
    if height < width:
        half1sum = np.sum(frame[:, :int(width / 2)])
        half2sum = np.sum(frame[:, int(width / 2):])
        if half1sum > half2sum:
            framerotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            print('rotated 90 degrees clockwise')
        else:
            framerotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            print('rotated 90 degrees counterclockwise')

    else:
        half1sum = np.sum(frame[:int(height / 2), :])
        half2sum = np.sum(frame[int(height / 2):, :])
        if half1sum > half2sum:
            framerotated = frame
            print('NO rotation')
            pass
        else:
            framerotated = cv2.rotate(frame, cv2.ROTATE_180)
            print('rotated 180 degrees')

    return framerotated, framerotated.shape[0], framerotated.shape[1]