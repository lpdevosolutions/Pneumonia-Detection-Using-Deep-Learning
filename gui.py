import os
import numpy as np
import customtkinter as ctk
import tensorflow as tf
from tkinter import filedialog
from PIL import Image, ImageTk


model_path = "dense.h5"
model = tf.keras.models.load_model(model_path)


class_labels = ["Normal", "Pneumonia Detected"]

def predict_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if not file_path:
        return
    
    img = Image.open(file_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)[0][0]
    result_text.set(f"Prediction: {class_labels[int(prediction > 0.5)]}")
    
    img_tk = ImageTk.PhotoImage(img)
    img_label.configure(image=img_tk)
    img_label.image = img_tk


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.title("Pneumonia Detection")
root.geometry("600x700")
root.configure(bg="#1E1E1E")

frame = ctk.CTkFrame(root, corner_radius=20, fg_color="#292B2F")
frame.pack(pady=20, padx=20, fill="both", expand=True)

header_label = ctk.CTkLabel(frame, text="🔍 Pneumonia Detection", font=("Arial", 22, "bold"), text_color="#FFFFFF")
header_label.pack(pady=10)

img_label = ctk.CTkLabel(frame, text="", width=350, height=350, corner_radius=15, fg_color="#3A3F44")
img_label.pack(pady=10)

result_text = ctk.StringVar()
result_label = ctk.CTkLabel(frame, textvariable=result_text, font=("Arial", 18, "bold"), text_color="#00FF7F")
result_label.pack(pady=10)

upload_button = ctk.CTkButton(frame, text="📂 Select Image", command=predict_image, fg_color="#7289DA", hover_color="#5B6EAE", font=("Arial", 14, "bold"))
upload_button.pack(pady=10)
root.iconbitmap("icon.ico")
root.mainloop()