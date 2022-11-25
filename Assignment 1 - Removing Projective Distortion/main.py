""" 
James Carlo E. Sorsona
2019-01401
CoE 197 M-THY
Removing Projective Distortion on Images
"""

from tkinter import *
from PIL import Image, ImageTk
import numpy as np

# getting the image dimensions and resizing according to canvas
def get_image_dimensions(file_path):
    file = Image.open(file_path)
    width, height = file.size
    if height > 720:
        width = int(width * (720 / height))
        height = 720
    if width > 540:
        height = int(height * (540 / width))
        width = 540
    return width, height

# mark the image in the tkinter canvas then execute remove_distortion() function once 4 points are collected
def mark_coordinates(event):
    if len(source_coordinates) < 4:
        source_coordinates.append([event.x, event.y])
        source_canvas.create_oval(event.x-3, event.y-3, event.x+3, event.y+3, fill="red")
    if len(source_coordinates) == 4:
        remove_distortion()

# reorder the coordinates into upper and lower given the 4 points
def get_order(coordinates):
    upper = []
    lower = []
    for i in range(4):
        if coordinates[i][1] < 360:
            upper.append(coordinates[i])
        else:
            lower.append(coordinates[i])
    upper.sort()
    lower.sort()
    return upper + lower

# order the matrix [x -> y, y -> x]
def order_matrix(matrix):
    x, y, z = matrix.shape
    new_matrix = np.zeros((y,x,z))
    for i in range(x):
        new_matrix[:,i] = matrix[i]
    return new_matrix.astype(int)
# reorder back the matrix [y -> x, x -> y]
def reorder_matrix(matrix):
    x, y, z = matrix.shape
    image = np.zeros((y,x,z))
    for i in range(x):
        image[:,i] = matrix[i]
    return image.astype(int)

# removes the projective distortion 
def remove_distortion():
    # getting the coordinates
    ordered_source_coordinates = get_order(source_coordinates)
    # fitting the coordinates in the canvas
    destination_coordinates = [[0,0],[540,0],[0,720],[540,720]]

    # HOMOGRAPHY
    p_i = np.zeros((8,9))
    for i in range(4):
        src_x, src_y = ordered_source_coordinates[i]
        dest_x, dest_y = destination_coordinates[i]
        p_i[i*2,:] = [src_x, src_y, 1, 0, 0, 0, -dest_x*src_x, -dest_x*src_y, -dest_x]
        p_i[i*2+1,:] = [0, 0, 0, src_x, src_y, 1, -dest_y*src_x, -dest_y*src_y, -dest_y]
    [U,S,V] = np.linalg.svd(p_i)
    matrix = V[-1,:] / V[-1,-1]
    homography = np.reshape(matrix,(3,3))

    pil_image = Image.open(file_path)
    pil_image = pil_image.resize((source_width, source_height)).convert("RGB")

    pil_matrix = order_matrix(np.array(pil_image))
    corrected_image = np.zeros((540,720,3))

    for i in range(corrected_image.shape[0]):
        for j in range(corrected_image.shape[1]):
            matrix = np.dot(homography, [i,j,1])
            k, l, m = (matrix / matrix[2]).astype(int)
            if (k >= 0 and k < 540) and (l >= 0 and l < 720):
                corrected_image[k,l] = pil_matrix[i,j]
    corrected_image = reorder_matrix(corrected_image)

    pil_array = Image.fromarray(corrected_image.astype("uint8"))
    pil_corrected = ImageTk.PhotoImage(pil_array)
    destination_canvas.create_image(0,0, image=pil_corrected, anchor=NW)
    
    root.mainloop()

if __name__ == "__main__":
    # window initialization
    root = Tk()
    root.title("Perspective Correction")
    root.geometry("1080x720")

    # list for storing coordinates
    global source_coordinates 
    source_coordinates = []

    # canvas initialization
    source_canvas = Canvas(height=720,width=540, bg="white")
    source_canvas.grid(row=2,column=0)
    destination_canvas = Canvas(height=720, width=540, bg="white")
    destination_canvas.grid(row=2,column=1)

    # getting image file
    file_path = "./test-images/cs-study-nook.jpg"

    # getting the image dimensions
    source_width, source_height = get_image_dimensions(file_path)

    # displaying the image in the canvas and marking the coordinates
    pill_image = Image.open(file_path)
    pill_image = pill_image.resize((source_width, source_height))
    source_image = ImageTk.PhotoImage(pill_image)
    source_image_canvas = source_canvas.create_image(0,0,image = source_image, anchor=NW)
    source_canvas.bind('<Button-1>', mark_coordinates)

    # tkinter mainloop
    root.mainloop()