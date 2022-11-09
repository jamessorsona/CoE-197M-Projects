import tkinter as tk
from PIL import Image, ImageTk

root = tk.Tk()
root.title("CoE 197M - Removing Projective Distortion on Images")
root.geometry("900x600")

# insert an image
img = Image.open("./test-images/cs-study-nook.jpg")
img = img.resize((600, 400), Image.ANTIALIAS)
img = ImageTk.PhotoImage(img)
panel = tk.Label(root, image=img)
panel.pack(side="top", fill="both", expand="yes")



root.mainloop()