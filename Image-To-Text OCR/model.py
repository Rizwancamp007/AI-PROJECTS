import pytesseract
import cv2
import numpy as np
from difflib import SequenceMatcher

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def preprocess_image(image_path):
    """
    Apply image preprocessing techniques to improve OCR accuracy
    """
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Image not found or invalid image format")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    kernel = np.ones((2, 2), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    processed = cv2.resize(processed, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    return processed


def extract_text_from_image(image_path):
    """
    Extract text using Tesseract OCR
    """
    processed_image = preprocess_image(image_path)

    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(processed_image, config=custom_config)

    return text.strip()


def character_error_rate(reference, extracted):
    """
    Calculate Character Error Rate (CER)
    """
    matcher = SequenceMatcher(None, reference, extracted)
    return (1 - matcher.ratio()) * 100


def word_error_rate(reference, extracted):
    """
    Calculate Word Error Rate (WER)
    """
    ref_words = reference.split()
    ext_words = extracted.split()

    matcher = SequenceMatcher(None, ref_words, ext_words)
    return (1 - matcher.ratio()) * 100
