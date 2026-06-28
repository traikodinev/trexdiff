# If needed, run once:
# %pip install ipywidgets ipycanvas pillow

import numpy as np
import time
import io
from PIL import Image, ImageDraw
from ipycanvas import Canvas, hold_canvas
from ipywidgets import VBox, HBox, Box, Button, HTML, FloatProgress, Image as WidgetImage, Layout
from IPython.display import display


# -----------------------------
# Settings
# -----------------------------

CANVAS_SIZE = 320
BRUSH_SIZE = 22

# Your OpenML MNIST preprocessing is:
# X = pixels / 255.0
# So the widget must output values in [0, 1].
PIXEL_SCALE = 255.0

# Keep at 1.0 for proper MNIST normalization.
# Increase slightly only if your drawn digits are consistently too faint.
INPUT_GAIN = 1.0

# MNIST-style conversion:
# crop digit, resize longest side to 20, paste into 28x28.
USE_BOUNDING_BOX = True
DIGIT_BOX_SIZE = 20

# Usually helps because MNIST digits are centered by mass.
USE_CENTER_OF_MASS = True

# Preview only. Does not affect model input.
PREVIEW_BRIGHTNESS = 1.8
PREVIEW_GAMMA = 0.75
PREVIEW_SIZE = 196

# Higher value = fewer updates while drawing, smoother UI.
PREDICT_INTERVAL = 0.10


# -----------------------------
# Python-side drawing buffer
# -----------------------------
# source_img always matches MNIST convention before normalization:
#   background = 0
#   digit      = 255

source_img = None
source_draw = None


def reset_source_image():
    global source_img, source_draw

    source_img = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
    source_draw = ImageDraw.Draw(source_img)


def draw_source_dot(x, y):
    r = BRUSH_SIZE / 2

    source_draw.ellipse(
        [x - r, y - r, x + r, y + r],
        fill=255
    )


def draw_source_line(x0, y0, x1, y1):
    source_draw.line(
        [x0, y0, x1, y1],
        fill=255,
        width=BRUSH_SIZE
    )

    # Rounded caps.
    draw_source_dot(x0, y0)
    draw_source_dot(x1, y1)


# -----------------------------
# UI
# -----------------------------

canvas = Canvas(
    width=CANVAS_SIZE,
    height=CANVAS_SIZE,
    sync_image_data=False
)

canvas.layout.border = "1px solid #aaa"
canvas.layout.width = f"{CANVAS_SIZE}px"
canvas.layout.height = f"{CANVAS_SIZE}px"

result = HTML(
    """
    <h2 style='margin-top: 0; width: 340px; white-space: nowrap; overflow: hidden;'>
      Draw a digit
    </h2>
    """,
    layout=Layout(width="340px", overflow="hidden")
)

preview_title = HTML("<b>28×28 model input</b>")

preview = WidgetImage(
    format="png",
    width=PREVIEW_SIZE - 2,
    height=PREVIEW_SIZE - 2,
    layout=Layout(
        width=f"{PREVIEW_SIZE - 2}px",
        height=f"{PREVIEW_SIZE - 2}px",
        overflow="hidden"
    )
)

preview_container = Box(
    [preview],
    layout=Layout(
        width=f"{PREVIEW_SIZE}px",
        height=f"{PREVIEW_SIZE}px",
        overflow="hidden",
        border="1px solid #ddd",
        align_items="center",
        justify_content="center"
    )
)

bars = [
    FloatProgress(
        value=0,
        min=0,
        max=1,
        description=str(i),
        readout_format=".2f",
        layout=Layout(width="280px")
    )
    for i in range(10)
]

clear_button = Button(description="Clear")


def reset_canvas():
    canvas.clear()

    canvas.fill_style = "black"
    canvas.fill_rect(0, 0, CANVAS_SIZE, CANVAS_SIZE)

    canvas.stroke_style = "white"
    canvas.fill_style = "white"
    canvas.line_width = BRUSH_SIZE
    canvas.line_cap = "round"
    canvas.line_join = "round"

    reset_source_image()


reset_canvas()


# -----------------------------
# Centering helpers
# -----------------------------

def shift_image_zero_padded(arr, shift_y, shift_x):
    """
    Shift a 2D array without wraparound.
    Empty areas are filled with zero.
    """

    out = np.zeros_like(arr)
    h, w = arr.shape

    src_y0 = max(0, -shift_y)
    src_y1 = min(h, h - shift_y)
    dst_y0 = max(0, shift_y)
    dst_y1 = min(h, h + shift_y)

    src_x0 = max(0, -shift_x)
    src_x1 = min(w, w - shift_x)
    dst_x0 = max(0, shift_x)
    dst_x1 = min(w, w + shift_x)

    if src_y1 > src_y0 and src_x1 > src_x0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = arr[src_y0:src_y1, src_x0:src_x1]

    return out


def center_by_mass(arr28):
    """
    Center the digit's center of mass near the middle of the 28x28 image.
    """

    mass = arr28.sum()

    if mass <= 0:
        return arr28

    ys, xs = np.indices(arr28.shape)

    cy = float((ys * arr28).sum() / mass)
    cx = float((xs * arr28).sum() / mass)

    target = 13.5

    shift_y = int(round(target - cy))
    shift_x = int(round(target - cx))

    return shift_image_zero_padded(arr28, shift_y, shift_x)


# -----------------------------
# Preview helper
# -----------------------------

def update_preview(arr28):
    """
    Show the actual 28x28 model input.
    Model input convention:
        0.0 = black background
        1.0 = white digit
    """

    img_arr = np.clip(arr28, 0.0, 1.0)

    # Preview brightness only. Does not affect inference.
    img_arr = np.clip(img_arr * PREVIEW_BRIGHTNESS, 0.0, 1.0)
    img_arr = img_arr ** PREVIEW_GAMMA

    img_arr = (img_arr * 255).astype(np.uint8)

    img = Image.fromarray(img_arr, mode="L")

    try:
        nearest = Image.Resampling.NEAREST
    except AttributeError:
        nearest = Image.NEAREST

    img = img.resize((PREVIEW_SIZE - 2, PREVIEW_SIZE - 2), nearest)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    preview.value = buf.getvalue()


# -----------------------------
# Convert drawing to MNIST input
# -----------------------------

def canvas_to_digit28(return_image=False):
    """
    Converts the Python-side drawing buffer into a normalized MNIST input.

    Returns:
        x shape: (1, 784)
        x values: 0.0 to 1.0
    """

    gray = np.array(source_img).astype(np.uint8)

    coords = np.argwhere(gray > 20)

    if coords.size == 0:
        arr28 = np.zeros((28, 28), dtype=np.float32)
        x = arr28.reshape(1, 784)

        if return_image:
            return x, arr28
        return x

    if USE_BOUNDING_BOX:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1

        crop = gray[y0:y1, x0:x1]

        img = Image.fromarray(crop)

        # Fit digit into a 20x20 box, like common MNIST preprocessing.
        scale = DIGIT_BOX_SIZE / max(img.size)
        new_size = tuple(max(1, int(round(s * scale))) for s in img.size)

        try:
            lanczos = Image.Resampling.LANCZOS
        except AttributeError:
            lanczos = Image.LANCZOS

        img = img.resize(new_size, lanczos)

        arr28_raw = np.zeros((28, 28), dtype=np.float32)

        x_offset = (28 - new_size[0]) // 2
        y_offset = (28 - new_size[1]) // 2

        arr28_raw[
            y_offset:y_offset + new_size[1],
            x_offset:x_offset + new_size[0]
        ] = np.array(img).astype(np.float32)

    else:
        # Less recommended: resize entire canvas directly.
        # This often makes the digit too small.
        img = Image.fromarray(gray)

        try:
            lanczos = Image.Resampling.LANCZOS
        except AttributeError:
            lanczos = Image.LANCZOS

        img = img.resize((28, 28), lanczos)
        arr28_raw = np.array(img).astype(np.float32)

    # Proper OpenML MNIST normalization:
    # raw pixels 0-255 -> floats 0-1.
    arr28 = arr28_raw / PIXEL_SCALE

    # Keep INPUT_GAIN at 1.0 for strict normalization.
    arr28 = np.clip(arr28 * INPUT_GAIN, 0.0, 1.0)

    if USE_CENTER_OF_MASS:
        arr28 = center_by_mass(arr28)

    x = arr28.reshape(1, 784)

    if return_image:
        return x, arr28

    return x


# -----------------------------
# Classification
# -----------------------------

def classify():
    x, arr28 = canvas_to_digit28(return_image=True)

    update_preview(arr28)

    probs = inference(x)[0]

    pred = int(np.argmax(probs))
    conf = float(probs[pred])

    result.value = f"""
    <h2 style='margin-top: 0; width: 340px; white-space: nowrap; overflow: hidden;'>
      Prediction:
      <span style='display: inline-block; width: 24px; text-align: center;'>{pred}</span>
      &nbsp; Confidence:
      <span style='display: inline-block; width: 72px; text-align: right; font-family: monospace;'>{conf:.1%}</span>
    </h2>
    """

    for i, bar in enumerate(bars):
        bar.value = float(probs[i])


# -----------------------------
# Drawing callbacks
# -----------------------------

drawing = False
last_x, last_y = None, None
last_pred_time = 0


def on_mouse_down(x, y):
    global drawing, last_x, last_y, last_pred_time

    x = int(round(x))
    y = int(round(y))

    drawing = True
    last_x, last_y = x, y

    draw_source_dot(x, y)

    with hold_canvas(canvas):
        canvas.fill_circle(x, y, BRUSH_SIZE / 2)

    classify()
    last_pred_time = time.time()


def on_mouse_move(x, y):
    global last_x, last_y, last_pred_time

    if not drawing:
        return

    x = int(round(x))
    y = int(round(y))

    draw_source_line(last_x, last_y, x, y)

    with hold_canvas(canvas):
        canvas.stroke_line(last_x, last_y, x, y)

    last_x, last_y = x, y

    now = time.time()
    if now - last_pred_time > PREDICT_INTERVAL:
        classify()
        last_pred_time = now


def on_mouse_up(x, y):
    global drawing, last_pred_time

    drawing = False
    classify()
    last_pred_time = time.time()


def clear(_=None):
    global drawing, last_x, last_y, last_pred_time

    drawing = False
    last_x, last_y = None, None
    last_pred_time = 0

    reset_canvas()

    empty = np.zeros((28, 28), dtype=np.float32)
    update_preview(empty)

    result.value = """
    <h2 style='margin-top: 0; width: 340px; white-space: nowrap; overflow: hidden;'>
      Draw a digit
    </h2>
    """

    for bar in bars:
        bar.value = 0


canvas.on_mouse_down(on_mouse_down)
canvas.on_mouse_move(on_mouse_move)
canvas.on_mouse_up(on_mouse_up)

clear_button.on_click(clear)

update_preview(np.zeros((28, 28), dtype=np.float32))


# -----------------------------
# Layout
# -----------------------------

preview_box = VBox(
    [
        result,
        preview_title,
        preview_container,
        clear_button
    ],
    layout=Layout(
        align_items="flex-start",
        margin="0 0 0 20px",
        width="360px",
        overflow="hidden"
    )
)

classes_title = HTML("<b>Class probabilities</b>")

classes_box = VBox(
    [
        classes_title,
        *bars
    ],
    layout=Layout(
        align_items="flex-start",
        margin="0 0 0 36px",
        padding="0",
        width="320px",
        overflow="hidden"
    )
)

right_panel = HBox([preview_box, classes_box], layout=Layout(align_items="flex-start", width="700px", overflow="hidden"))
display(HBox([canvas, right_panel],layout=Layout(align_items="flex-start", overflow="hidden")))