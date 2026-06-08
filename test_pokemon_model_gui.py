from PIL import Image, ImageTk, ImageGrab
import tkinter as tk
from tkinter import filedialog

from classifier import PokemonClassifier, load_class_names

# Load class names from saved file
class_names = load_class_names()

# GUI setup
class PokemonApp:
    def __init__(self, root, classifier):
        self.classifier = classifier
        self.root = root
        self.root.title("Pokémon Identifier")
        
        self.label = tk.Label(root, text="Select or paste an image to identify the Pokémon", font=("Arial", 14))
        self.label.pack(pady=20)

        self.image_label = tk.Label(root)
        self.image_label.pack()

        self.result_label = tk.Label(root, text="", font=("Arial", 14))
        self.result_label.pack(pady=10)

        self.button_frame = tk.Frame(root)
        self.button_frame.pack(pady=20)

        self.select_button = tk.Button(self.button_frame, text="Select Image", command=self.select_image, font=("Arial", 12))
        self.select_button.grid(row=0, column=0, padx=10)

        self.paste_button = tk.Button(self.button_frame, text="Paste Image", command=self.paste_image, font=("Arial", 12))
        self.paste_button.grid(row=0, column=1, padx=10)

    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Pokémon Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            img = Image.open(file_path)
            self.display_and_predict(img)

    def paste_image(self):
        # Try to get an image from the clipboard
        try:
            img = ImageGrab.grabclipboard()
            if isinstance(img, Image.Image):
                self.display_and_predict(img)
            else:
                self.result_label.config(text="No image in clipboard.")
        except Exception as e:
            self.result_label.config(text=f"Error: {e}")

    def display_and_predict(self, img):
        img.thumbnail((256, 256))
        img_tk = ImageTk.PhotoImage(img)
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk

        prediction, confidence = self.classifier.predict(img)[0]
        self.result_label.config(text=f"Prediction: {prediction} ({confidence:.2f}%)")

if __name__ == "__main__":
    model_path = 'models/best_sketch.pth'
    classifier = PokemonClassifier(model_path, class_names)

    root = tk.Tk()
    app = PokemonApp(root, classifier)
    root.mainloop()
