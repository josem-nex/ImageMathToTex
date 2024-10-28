import re
import heapq
import cv2
import time
import numpy as np
import math
import pytesseract

from collections import Counter
from typing import List
from PIL import Image

from ..det_model.inference import predict as latex_det_predict
from ..det_model.Bbox import Bbox, draw_bboxes

from ..ocr_model.utils.inference import inference as latex_rec_predict
from ..ocr_model.utils.to_katex import to_katex, change_all

from texteller.ocr_for_text import extract_text_from_image
MAXV = 999999999

ERROR_Y = 5
ERROR_X = 5
MARGIN = 27


def remove_noise(image):
    # Dividimos la imagen en sus canales B, G, R
    b_channel, g_channel, r_channel = cv2.split(image)
    
    # Definimos el kernel para "sharpening"
    kernel = np.array([[0, -1, 0], [-1, 4.9, -1], [0, -1, 0]])

    # Aplicamos el filtro y denoising a cada canal por separado
    sharpen_b = cv2.filter2D(b_channel, -1, kernel)
    sharpen_g = cv2.filter2D(g_channel, -1, kernel)
    sharpen_r = cv2.filter2D(r_channel, -1, kernel)

    denoised_b = cv2.fastNlMeansDenoising(sharpen_b, None, h=30, templateWindowSize=7, searchWindowSize=21)
    denoised_g = cv2.fastNlMeansDenoising(sharpen_g, None, h=30, templateWindowSize=7, searchWindowSize=21)
    denoised_r = cv2.fastNlMeansDenoising(sharpen_r, None, h=30, templateWindowSize=7, searchWindowSize=21)

    # Combinamos los canales de nuevo
    denoised_image = cv2.merge([denoised_b, denoised_g, denoised_r])

    # Guardamos el resultado
    cv2.imwrite('imagen_denoised_color.jpg', denoised_image)

    return denoised_image

def mask_img(img, bboxes: List[Bbox], bg_color: np.ndarray) -> np.ndarray:
    mask_img = img.copy()
    for bbox in bboxes:
        mask_img[bbox.p.y:bbox.p.y + bbox.h, bbox.p.x:bbox.p.x + bbox.w] = bg_color
    return mask_img


def bbox_merge(sorted_bboxes: List[Bbox]) -> List[Bbox]:
    if (len(sorted_bboxes) == 0):
        return []
    bboxes = sorted_bboxes.copy()
    guard = Bbox(MAXV, bboxes[-1].p.y, -1, -1, label="guard")
    bboxes.append(guard)
    res = []
    prev = bboxes[0]
    for curr in bboxes:
        if prev.ur_point.x <= curr.p.x or not prev.same_row(curr):
            res.append(prev)
            prev = curr
        else:
            prev.w = max(prev.w, curr.ur_point.x - prev.p.x)
    return res


def split_conflict(ocr_bboxes: List[Bbox], latex_bboxes: List[Bbox]) -> List[Bbox]:
    if latex_bboxes == []:
        return ocr_bboxes
    if ocr_bboxes == [] or len(ocr_bboxes) == 1:
        return ocr_bboxes

    bboxes = sorted(ocr_bboxes + latex_bboxes)

    # log results
    for idx, bbox in enumerate(bboxes):
        bbox.content = str(idx)
        
    draw_bboxes(Image.fromarray(img), bboxes, name="before_split_confict.png")

    assert len(bboxes) > 1

    heapq.heapify(bboxes)
    res = []
    candidate = heapq.heappop(bboxes)
    curr = heapq.heappop(bboxes)
    idx = 0
    while (len(bboxes) > 0):
        idx += 1
        assert candidate.p.x <= curr.p.x or not candidate.same_row(curr)

        if candidate.ur_point.x <= curr.p.x or not candidate.same_row(curr):
            res.append(candidate)
            candidate = curr
            curr = heapq.heappop(bboxes)
        elif candidate.ur_point.x < curr.ur_point.x:
            assert not (candidate.label != "text" and curr.label != "text")
            if candidate.label == "text" and curr.label == "text":
                candidate.w = curr.ur_point.x - candidate.p.x
                curr = heapq.heappop(bboxes)
            elif candidate.label != curr.label:
                if candidate.label == "text":
                    candidate.w = curr.p.x - candidate.p.x
                    res.append(candidate)
                    candidate = curr
                    curr = heapq.heappop(bboxes)
                else:
                    curr.w = curr.ur_point.x - candidate.ur_point.x
                    curr.p.x = candidate.ur_point.x
                    heapq.heappush(bboxes, curr)
                    curr = heapq.heappop(bboxes)
                
        elif candidate.ur_point.x >= curr.ur_point.x:
            assert not (candidate.label != "text" and curr.label != "text")

            if candidate.label == "text":
                assert curr.label != "text"
                heapq.heappush(
                    bboxes,
                    Bbox(
                        curr.ur_point.x,
                        candidate.p.y,
                        candidate.h,
                        candidate.ur_point.x - curr.ur_point.x,
                        label="text",
                        confidence=candidate.confidence,
                        content=None
                    )
                )
                candidate.w = curr.p.x - candidate.p.x
                res.append(candidate)
                candidate = curr
                curr = heapq.heappop(bboxes)
            else:
                assert curr.label == "text"
                curr = heapq.heappop(bboxes)
        else:
            assert False
    res.append(candidate)
    res.append(curr)

    # log results
    for idx, bbox in enumerate(res):
        bbox.content = str(idx)
        bbox.ord = idx
    draw_bboxes(Image.fromarray(img), res, name="after_split_confict.png")

    return res


def slice_from_image(img: np.ndarray, ocr_bboxes: List[Bbox]) -> List[np.ndarray]:
    sliced_imgs = []
    for bbox in ocr_bboxes:
        x, y = int(bbox.p.x), int(bbox.p.y)
        w, h = int(bbox.w), int(bbox.h)
        sliced_img = img[y:y+h, x:x+w]
        sliced_imgs.append(sliced_img)
    return sliced_imgs


def union_bboxes(ocr_bboxes):
    new_ocr_bboxes = []

    if(len(ocr_bboxes)==0):
        return new_ocr_bboxes
    
    first_y = ocr_bboxes[0].p.y
    min_x = ocr_bboxes[0].p.x
    max_x = min_x+ocr_bboxes[0].w
    tag = ocr_bboxes[0].ord

    i = 1
    while i < len(ocr_bboxes):
        if (int(ocr_bboxes[i].content) == int(ocr_bboxes[i-1].content) + 1) and (abs(ocr_bboxes[i].p.x-min_x)<70) and (abs(ocr_bboxes[i].p.x+ocr_bboxes[i].w-max_x)<70):
            min_x = min(min_x,ocr_bboxes[i].p.x)
            max_x = max(max_x,ocr_bboxes[i].p.x+ocr_bboxes[i].w)
        else:
            
            bbox = Bbox(max(0,math.floor(min_x-ERROR_X)),max(0,math.floor(first_y-ERROR_Y)),math.ceil(ocr_bboxes[i-1].p.y+ocr_bboxes[i-1].h-max(0,math.floor(first_y-ERROR_Y))+ERROR_Y),math.ceil(max_x-math.floor(min_x-ERROR_X)+ERROR_X),'text')
            new_ocr_bboxes.append(bbox)
            first_y = int(ocr_bboxes[i].p.y)
            min_x = int(ocr_bboxes[i].p.x)
            max_x = int(min_x+ocr_bboxes[i].w)
            bbox.ord = tag
            tag = ocr_bboxes[i].ord
        i += 1
    bbox = Bbox(max(0,math.floor(min_x-ERROR_X)),max(0,math.floor(first_y-ERROR_Y)),math.ceil(ocr_bboxes[i-1].p.y+ocr_bboxes[i-1].h-max(0,math.floor(first_y-ERROR_Y))+ERROR_Y),math.ceil(max_x-math.floor(min_x-ERROR_X)+ERROR_X),'text')
    bbox.ord = tag
    new_ocr_bboxes.append(bbox)
    return new_ocr_bboxes


def mix_inference(
    img_path: str,
    infer_config,
    latex_det_model,

    lang_ocr_models,

    latex_rec_models,
    accelerator="cpu",
    num_beams=1
) -> str:
    '''
    Input a mixed image of formula text and output str (in markdown syntax)
    '''
    global img
    img = cv2.imread(img_path)
    width = img.shape[1]

    # Remove the noise from the image.
    img = remove_noise(img)
    corners = [tuple(img[0, 0]), tuple(img[0, -1]),
               tuple(img[-1, 0]), tuple(img[-1, -1])]
    bg_color = np.array(Counter(corners).most_common(1)[0][0])

    # Detect and merge mathematical formula bounding boxes using a detection model (latex_det_model).
    start_time = time.time()
    latex_bboxes = latex_det_predict(img_path, latex_det_model, infer_config)
    end_time = time.time()
    print(f"latex_det_model time: {end_time - start_time:.2f}s")
    latex_bboxes = sorted(latex_bboxes)
    # log results
    draw_bboxes(Image.fromarray(img), latex_bboxes, name="latex_bboxes(unmerged).png")
    latex_bboxes = bbox_merge(latex_bboxes)
    # log results
    draw_bboxes(Image.fromarray(img), latex_bboxes, name="latex_bboxes(merged).png")

    # Apply a background color to the regions determined by the provided bounding boxes.
    masked_img = mask_img(img, latex_bboxes, bg_color)

    # Detect and merge text bounding boxes using an OCR detection model (det_model).
    det_model, rec_model = lang_ocr_models
    start_time = time.time()
    det_prediction, _ = det_model(masked_img)
    end_time = time.time()
    print(f"ocr_det_model time: {end_time - start_time:.2f}s")
    ocr_bboxes = [
        Bbox(
            p[0][0], p[0][1], p[3][1]-p[0][1], p[1][0]-p[0][0],
            label="text",
            confidence=None,
            content=None
        )
        for p in det_prediction
    ]
    # log results
    draw_bboxes(Image.fromarray(img), ocr_bboxes, name="ocr_bboxes(unmerged).png")

    ocr_bboxes = sorted(ocr_bboxes)
    ocr_bboxes = bbox_merge(ocr_bboxes)
    # log results
    draw_bboxes(Image.fromarray(img), ocr_bboxes, name="ocr_bboxes(merged).png")

    # Resolve overlapping conflicts between text and formula bounding boxes.
    ocr_bboxes = split_conflict(ocr_bboxes, latex_bboxes)

    # Merge adjacent text bounding boxes.
    ocr_bboxes = list(filter(lambda x: x.label == "text", ocr_bboxes))
    ocr_bboxes = union_bboxes(ocr_bboxes)
    draw_bboxes(Image.fromarray(img), ocr_bboxes, name="ocr_bboxes(union).png")

    # Extract and recognize the text contained in each bounding box using a text recognition model (Tesseract).
    sliced_imgs: List[np.ndarray] = slice_from_image(img, ocr_bboxes)
    start_time = time.time()
    rec_predictions = [extract_text_from_image(image) for image in sliced_imgs]
    end_time = time.time()
    print(f"ocr_tesseract_model time: {end_time - start_time:.2f}s")

    for content, bbox in zip(rec_predictions, ocr_bboxes):
        bbox.content = content

    # Recognize and convert mathematical formulas using a formula recognition model (latex_rec_model).
    latex_imgs =[]
    for bbox in latex_bboxes:
        latex_imgs.append(img[bbox.p.y:bbox.p.y + bbox.h, bbox.p.x:bbox.p.x + bbox.w])
    start_time = time.time()
    latex_rec_res = latex_rec_predict(*latex_rec_models, latex_imgs, accelerator, num_beams, max_tokens=800)
    end_time = time.time()
    print(f"latex_rec_model time: {end_time - start_time:.2f}s")

    for bbox, content in zip(latex_bboxes, latex_rec_res):
        bbox.content = to_katex(content)
        if bbox.label == "embedding":
            bbox.content = " $" + bbox.content + "$ "
        elif bbox.label == "isolated":
            bbox.content = '\n\n' + r"\begin{align}" + bbox.content + r"\end{align}" + '\n\n'

    
    bboxes = sorted(ocr_bboxes+latex_bboxes,key=lambda x:x.ord)
    if bboxes == []:
        return ""

    # Return the result in Markdown format, with correctly formatted extracted text and mathematical formulas.
    md = ""
    prev = Bbox(bboxes[0].p.x, bboxes[0].p.y, -1, -1, label="guard")
    for curr in bboxes:
        
        if not prev.same_row(curr) and curr.label!="isolated" and prev.label!="isolated":

            if max(0,(curr.ul_point.y - prev.ll_point.y))>10:
                md += r'\vspace{0.25cm}'
                md += '\n\n'
            elif prev.ur_point.x/width < 0.7:
                md += '\n\n'
            else:
                md += " "
            
        # Add the formula number back to the isolated formula
        if (
            prev.label == "isolated"
            and curr.label == "text"
            and prev.same_row(curr)
        ):
            curr.content = curr.content.strip()
            if curr.content.startswith('(') and curr.content.endswith(')'):
                curr.content = curr.content[1:-1]

            if re.search(r'\\tag\{.*\}$', md[:-4]) is not None:
                # in case of multiple tag
                md = md[:-5] + f', {curr.content}' + '}' + md[-11:]
            else:
                md = md[:-13] + f'\\tag{{{curr.content}}}' + md[-13:]
            continue

        if curr.label == "embedding":
            # remove the bold effect from inline formulas
            curr.content = change_all(curr.content, r'\bm', r' ', r'{', r'}', r'', r' ')
            curr.content = change_all(curr.content, r'\boldsymbol', r' ', r'{', r'}', r'', r' ')
            curr.content = change_all(curr.content, r'\textit', r' ', r'{', r'}', r'', r' ')
            curr.content = change_all(curr.content, r'\textbf', r' ', r'{', r'}', r'', r' ')
            curr.content = change_all(curr.content, r'\textbf', r' ', r'{', r'}', r'', r' ')
            curr.content = change_all(curr.content, r'\mathbf', r' ', r'{', r'}', r'', r' ')

            # change split environment into aligned
            curr.content = curr.content.replace(r'\begin{split}', r'\begin{aligned}')
            curr.content = curr.content.replace(r'\end{split}', r'\end{aligned}')

            # remove extra spaces (keeping only one)
            curr.content = re.sub(r' +', ' ', curr.content)
            assert curr.content.startswith(' $') and curr.content.endswith('$ ')
            curr.content = ' $' + curr.content[2:-2].strip() + '$ '
        md += curr.content
        prev = curr
    
    return r'\documentclass[a4paper,11pt ]{article}\usepackage{amsmath}\usepackage{amssymb}\usepackage{geometry}\usepackage{setspace}\geometry{left=2.5 cm, right=2.5 cm, top=2.5 cm, bottom=2.5 cm}\begin{document}'+md.strip()+r'\end{document}'
