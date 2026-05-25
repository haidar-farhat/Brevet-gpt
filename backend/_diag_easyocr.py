import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch
import sentence_transformers
print("torch:", torch.__version__, "| sentence-transformers:", sentence_transformers.__version__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import gpu_ocr_books as gp
import fitz
import numpy as np
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

gp.register_fonts()
paras = [
    "المدنية مادة تعلّم الطلاب الحقوق والواجبات.",
    "في سنة 1943 نالت لبنان استقلالها وأصبحت دولة مستقلة.",
]
src = os.path.join(ROOT, "_eo.pdf"); png = os.path.join(ROOT, "_eo.png")
c = canvas.Canvas(src, pagesize=letter); c.setFont(gp.FONT_NAME, 24)
y = 720
for p in paras:
    c.drawRightString(560, y, gp.shape_rtl(p)); y -= 80
c.showPage(); c.save()
d = fitz.open(src); d[0].get_pixmap(dpi=200).save(png); d.close()

import easyocr
reader = easyocr.Reader(["ar"], gpu=False, verbose=False)
img = np.array(Image.open(png).convert("RGB"))
lines = reader.readtext(img, detail=0, paragraph=True)
print("EASYOCR result:")
for l in lines:
    print("  ", l)

os.remove(src); os.remove(png)
