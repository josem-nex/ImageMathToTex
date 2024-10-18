import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from texteller.inference import process_image
from texteller.models.ocr_model.model.TexTeller import TexTeller

def select_image():
#* Aqui se recibe el path de la imagen
    file_path = filedialog.askopenfilename()

    if file_path:
        img= Image.open(file_path)
        img.thumbnail((200, 200))  # Resizing image to fit in window
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img

        print(img)

        res = process_image(file_path, mix = True, latex_rec_model=latex_rec_model, tokenizer=tokenizer)
        latex_code.set(res)
        code_display.config(state='normal')  # Enable editing in Text widget to insert content
        code_display.delete(1.0, tk.END)  # Clear the Text widget
        code_display.insert(tk.END, latex_code.get())  # Insert LaTeX code
        code_display.config(state='disabled')  # Disable editing after insertion

root = tk.Tk()
root.title("Image to LaTeX")

print('Loading model and tokenizer...')
latex_rec_model = TexTeller.from_pretrained()
tokenizer = TexTeller.get_tokenizer()
print('Model and tokenizer loaded.')

latex_code = tk.StringVar()

# Button to select image
btn = tk.Button(root, text="Select Image", command=select_image)
btn.pack(pady=20)

# Label to display selected image
panel = tk.Label(root)
panel.pack()

# Label to display LaTeX code

latex_label = tk.Label(root, text="Generated LaTeX Code:")
latex_label.pack()

code_display = tk.Text(root, height=15, width=50, font=("Arial", 10), wrap="word")
code_display.pack(pady=5, padx=10)
code_display.config(state='disabled') 
def copy_to_clipboard():
    root.clipboard_clear()
    root.clipboard_append(latex_code.get())

copy_button = tk.Button(root, text="Copy LaTeX Code", command=copy_to_clipboard, height=2, width=20)
copy_button.pack(pady=10)


root.mainloop()
