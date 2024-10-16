# Importamos las dependencias del proyecto
import cv2
import pytesseract

# Esta línea es importante para correr Tesseract en Windows
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Definimos una función que recibe la ruta de una imagen y devuelve el texto extraído
def extract_text_from_image(image):
    # Como OpenCV carga las imágenes en formato BGR, la convertimos a RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Extraemos el texto de la imagen utilizando Tesseract
    text = pytesseract.image_to_string(image)

    # Devolvemos el texto escaneado
    return text