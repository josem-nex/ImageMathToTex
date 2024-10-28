import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import ttkbootstrap as ttk
from texteller.inference import process_image
from texteller.models.ocr_model.model.TexTeller import TexTeller

def select_image():

    file_path = filedialog.askopenfilename()

    if file_path:
        img= Image.open(file_path)
        img.thumbnail((200, 200))  # Resizing image to fit in window
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img
        
        loading_label.forget()
        loading_label.pack()
        show_loading_overlay()
        root.update()
        
        root.after(5000, None) 
        
        res = process_image(file_path, mix = True, latex_rec_model=latex_rec_model, tokenizer=tokenizer)
        latex_code.set(res)
        
        code_display.config(state='normal')  # Enable editing in Text widget to insert content
        code_display.delete(1.0, tk.END)  # Clear the Text widget
        code_display.insert(tk.END, latex_code.get())  # Insert LaTeX code
        code_display.config(state='disabled')
        # Disable editing after insertion
        loading_label.pack_forget()
        hide_loading_overlay()
    else:
        hide_loading_overlay()
       

def show_loading_overlay():
    loading_status.set(True)
    loading_overlay.place(relx=0.5, rely=0.3, anchor="center")
    loading_overlay.lift()  # Bring to the front
 
def hide_loading_overlay():
    loading_status.set(False)
    loading_overlay.place_forget()        

print('Loading model and tokenizer...')
latex_rec_model = TexTeller.from_pretrained()
tokenizer = TexTeller.get_tokenizer()
print('Model and tokenizer loaded.')

root = ttk.Window(themename="superhero")
ravenLogo = ttk.PhotoImage(file = 'logo.png',name='Logo') 
root.iconphoto(False,ravenLogo)
root.title("Image to LaTeX")

#vars like statuses
loading_status=tk.BooleanVar()
latex_code = tk.StringVar()

# Header Label
header = ttk.Label(root, text="Image to LaTeX Converter", font=("Helvetica", 16, "bold"))
header.pack(pady=10)


# Button to select image
btn = ttk.Button(root, text="Select Image", command=select_image, bootstyle="primary-outline", width=20)
btn.pack(pady=10)

# Label to display selected image
panel_frame = ttk.Frame(root, padding=10, bootstyle="secondary")
panel_frame.pack(pady=5, padx=10, fill='both')
panel = ttk.Label(panel_frame, text="No image selected", width=50, anchor="center")
panel.pack(padx=5, pady=5)

# Loading overlay
loading_overlay = ttk.Frame(root, style="secondary", padding=20)
loading_label = ttk.Label(loading_overlay, text="Loading...", font=("Arial", 14, "italic"))
loading_label.pack()

latex_label = ttk.Label(root, text="Generated LaTeX Code:", font=("Arial", 12))
latex_label.pack(pady=10)


code_frame = ttk.Frame(root, padding=10)
code_frame.pack(fill='both', expand=True, padx=5, pady=5)

# Text widget to display LaTeX code (read-only)
code_display = tk.Text(code_frame, font=("Arial", 10), wrap="word", borderwidth=2, relief="ridge",height=10)
code_display.pack(fill='both')
code_display.config(state='disabled')

def copy_to_clipboard():
    root.clipboard_clear()
    root.clipboard_append(latex_code.get())

copy_button = ttk.Button(root, text="Copy LaTeX Code", command=copy_to_clipboard, bootstyle="success", width=20)
copy_button.pack(pady=10)

root.mainloop()
