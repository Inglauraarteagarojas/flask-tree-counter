"""
CONTADOR DE ÁRBOLES — UMNG Cajicá
Flask + NumPy + SciPy | Detección de copas de árboles
Laura Mercedes Arteaga Rojas — UMNG — Mayo 2026
"""

import os
import uuid
import json
import base64
from io import BytesIO

from flask import Flask, render_template, request, jsonify, Response
import numpy as np
from PIL import Image, ImageDraw

# Permitir ortofotos grandes de dron.
# Esto corrige el error:
# Image size exceeds limit ... could be decompression bomb DOS attack.
Image.MAX_IMAGE_PIXELS = None

from scipy.ndimage import (
    uniform_filter,
    label,
    find_objects,
    binary_opening,
    binary_erosion,
    binary_dilation,
    gaussian_filter,
    maximum_filter,
    distance_transform_edt,
)

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app.config["UPLOAD_FOLDER"] = os.path.join(BASE_DIR, "static", "uploads")
app.config["RESULTS_FOLDER"] = os.path.join(BASE_DIR, "static", "results")
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024

ALLOWED = {"png", "jpg", "jpeg", "tif", "tiff", "bmp"}

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["RESULTS_FOLDER"], exist_ok=True)


def allowed_file(filename):
    """Valida extensiones permitidas."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED


def detect_trees(path, params=None):
    """
    Detecta copas de árboles usando:
    - Color verde aproximado.
    - Índice ExG.
    - Textura local.
    - Separación de copas con transformada de distancia.
    """

    p = dict(
        max_dim=1600,
        hue_min=35,
        hue_max=170,
        sat_min=8,
        val_max=72,
        exg_min=3,
        texture_thr=3,
        erosion_iter=1,
        peak_spacing=14,
        min_radius=3,
        gauss_sigma=2,
    )

    if params:
        for key, value in params.items():
            if key in p:
                p[key] = type(p[key])(value)

    img = Image.open(path).convert("RGB")
    original_width, original_height = img.size
    scale = 1.0

    # Reducir imagen grande para que Render no se quede sin memoria.
    if max(original_width, original_height) > p["max_dim"]:
        scale = p["max_dim"] / max(original_width, original_height)
        img = img.resize(
            (int(original_width * scale), int(original_height * scale)),
            Image.LANCZOS,
        )

    arr = np.array(img).astype(float)
    height, width = arr.shape[:2]

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    # Índice Excess Green.
    exg = 2 * g - r - b

    max_rgb = np.maximum(np.maximum(r, g), b)
    min_rgb = np.minimum(np.minimum(r, g), b)
    delta = max_rgb - min_rgb

    with np.errstate(invalid="ignore", divide="ignore"):
        sat = np.where(max_rgb > 0, (delta / max_rgb) * 100, 0)

    val = (max_rgb / 255) * 100
    hue = np.zeros_like(r)

    mask_r = (max_rgb == r) & (delta > 0)
    mask_g = (max_rgb == g) & (delta > 0) & ~mask_r
    mask_b = (max_rgb == b) & (delta > 0) & ~mask_r & ~mask_g

    with np.errstate(invalid="ignore", divide="ignore"):
        hue[mask_r] = 60 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
        hue[mask_g] = 60 * ((b[mask_g] - r[mask_g]) / delta[mask_g] + 2)
        hue[mask_b] = 60 * ((r[mask_b] - g[mask_b]) / delta[mask_b] + 4)

    hue[hue < 0] += 360

    # Textura local sobre canal verde.
    green_mean = uniform_filter(g, size=7)
    green_std = np.sqrt(
        np.maximum(uniform_filter(g * g, size=7) - green_mean**2, 0)
    )

    # Máscara candidata de vegetación.
    candidate_mask = (
        (hue >= p["hue_min"])
        & (hue <= p["hue_max"])
        & (sat >= p["sat_min"])
        & (val >= 5)
        & (val <= p["val_max"])
        & (exg >= p["exg_min"])
        & (g > r * 0.8)
        & (g > b * 0.95)
        & (green_std > p["texture_thr"])
    )

    # Filtro para reducir falsos positivos de césped o suelo verdoso homogéneo.
    grass_mask = (val > 48) & (green_std < 3.5) & (sat < 35)
    candidate_mask = candidate_mask & ~grass_mask

    clean_mask = binary_erosion(
        candidate_mask,
        structure=np.ones((3, 3)),
        iterations=p["erosion_iter"],
    )

    clean_mask = binary_opening(
        clean_mask,
        structure=np.ones((3, 3)),
        iterations=1,
    )

    clean_mask = binary_dilation(
        clean_mask,
        structure=np.ones((2, 2)),
        iterations=1,
    )

    # Separación de copas.
    dist = distance_transform_edt(clean_mask)
    dist_smooth = gaussian_filter(dist, sigma=p["gauss_sigma"])

    local_max = maximum_filter(dist_smooth, size=p["peak_spacing"])
    peaks = (dist_smooth == local_max) & (
        dist_smooth >= max(2.0, p["min_radius"] * 0.45)
    )

    peak_labels, _ = label(peaks)
    peak_slices = find_objects(peak_labels)

    trees = []

    for idx, peak_slice in enumerate(peak_slices):
        if peak_slice is None:
            continue

        peak_mask = peak_labels[peak_slice] == (idx + 1)
        peak_y, peak_x = np.where(peak_mask)

        if len(peak_y) == 0 or len(peak_x) == 0:
            continue

        center_y = int(peak_slice[0].start + peak_y.mean())
        center_x = int(peak_slice[1].start + peak_x.mean())

        radii = []

        for angle_index in range(16):
            angle = 2 * np.pi * angle_index / 16
            border_distance = 0

            for distance in range(2, 55):
                px = int(round(center_x + np.cos(angle) * distance))
                py = int(round(center_y + np.sin(angle) * distance))

                if 0 <= px < width and 0 <= py < height and candidate_mask[py, px]:
                    border_distance = distance
                elif distance > border_distance + 3:
                    break

            radii.append(border_distance)

        if not radii:
            continue

        median_radius = float(np.median(radii))

        if median_radius < p["min_radius"]:
            continue

        box_radius = median_radius * 1.20

        x0 = max(0, int(center_x - box_radius))
        y0 = max(0, int(center_y - box_radius))
        x1 = min(width - 1, int(center_x + box_radius))
        y1 = min(height - 1, int(center_y + box_radius))

        box_width = x1 - x0
        box_height = y1 - y0

        if box_width < 5 or box_height < 5:
            continue

        mask_inside = candidate_mask[y0:y1, x0:x1]
        fill_ratio = mask_inside.sum() / max(1, box_width * box_height)

        if fill_ratio < 0.05:
            continue

        mean_exg = (
            float(exg[y0:y1, x0:x1][mask_inside].mean())
            if mask_inside.sum() > 0
            else 0
        )

        mean_sat = (
            float(sat[y0:y1, x0:x1][mask_inside].mean())
            if mask_inside.sum() > 0
            else 0
        )

        if mean_exg > 12 and mean_sat > 16:
            health = "Healthy"
        elif mean_exg > 4:
            health = "Moderate"
        else:
            health = "Dry"

        trees.append(
            dict(
                cx=center_x,
                cy=center_y,
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                radius=round(median_radius, 1),
                exg=round(mean_exg, 1),
                health=health,
                source="auto",
            )
        )

    # Eliminar detecciones demasiado solapadas.
    sorted_trees = sorted(trees, key=lambda tree: tree["radius"], reverse=True)
    keep = []

    for tree in sorted_trees:
        valid = True

        for selected in keep:
            ix0 = max(tree["x0"], selected["x0"])
            iy0 = max(tree["y0"], selected["y0"])
            ix1 = min(tree["x1"], selected["x1"])
            iy1 = min(tree["y1"], selected["y1"])

            if ix0 < ix1 and iy0 < iy1:
                intersection = (ix1 - ix0) * (iy1 - iy0)
                area_tree = (tree["x1"] - tree["x0"]) * (tree["y1"] - tree["y0"])
                area_selected = (selected["x1"] - selected["x0"]) * (
                    selected["y1"] - selected["y0"]
                )
                union = area_tree + area_selected - intersection

                if union > 0:
                    iou = intersection / union
                    if iou > 0.2:
                        valid = False
                        break

        if valid:
            keep.append(tree)

    trees = keep
    trees.sort(key=lambda tree: (tree["cy"] // 35, tree["cx"]))

    for index, tree in enumerate(trees):
        tree["id"] = index + 1
        tree["label"] = f"a{index + 1}"

    result_image = draw_detections(img, trees)

    health_count = {}

    for tree in trees:
        health_count[tree["health"]] = health_count.get(tree["health"], 0) + 1

    stats = dict(
        original_size=f"{original_width}x{original_height}",
        processed_size=f"{width}x{height}",
        scale=round(scale * 100, 1),
        total=len(trees),
        healthy=health_count.get("Healthy", 0),
        moderate=health_count.get("Moderate", 0),
        dry=health_count.get("Dry", 0),
        params=p,
    )

    return trees, result_image, stats


def draw_detections(img, trees):
    """Dibuja cajas, puntos centrales y etiquetas sobre la imagen."""
    result = img.copy()
    draw = ImageDraw.Draw(result, "RGBA")

    colors = {
        "Healthy": (102, 255, 51),
        "Moderate": (255, 215, 0),
        "Dry": (255, 51, 51),
    }

    for tree in trees:
        color = colors.get(tree["health"], colors["Healthy"])

        x0 = tree["x0"]
        y0 = tree["y0"]
        x1 = tree["x1"]
        y1 = tree["y1"]

        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)

        handle_size = 5
        handle_points = [
            (x0, y0),
            (x1, y0),
            (x0, y1),
            (x1, y1),
            ((x0 + x1) // 2, y0),
            ((x0 + x1) // 2, y1),
            (x0, (y0 + y1) // 2),
            (x1, (y0 + y1) // 2),
        ]

        for handle_x, handle_y in handle_points:
            draw.rectangle(
                [
                    handle_x - handle_size // 2,
                    handle_y - handle_size // 2,
                    handle_x + handle_size // 2,
                    handle_y + handle_size // 2,
                ],
                fill=(255, 255, 255),
                outline=color,
            )

        cross_size = min(8, max(3, (x1 - x0) // 3))

        draw.line(
            [
                (tree["cx"] - cross_size, tree["cy"]),
                (tree["cx"] + cross_size, tree["cy"]),
            ],
            fill=(255, 34, 34),
            width=1,
        )

        draw.line(
            [
                (tree["cx"], tree["cy"] - cross_size),
                (tree["cx"], tree["cy"] + cross_size),
            ],
            fill=(255, 34, 34),
            width=1,
        )

        text = f"{tree['label']} {tree['health']}"
        bbox = draw.textbbox((0, 0), text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        text_color = (0, 0, 0) if tree["health"] != "Dry" else (255, 255, 255)

        label_y0 = max(0, y0 - text_height - 6)
        label_y1 = max(text_height + 4, y0 - 1)

        draw.rectangle(
            [x0, label_y0, x0 + text_width + 8, label_y1],
            fill=color + (220,),
        )

        draw.text(
            (x0 + 4, label_y0 + 2),
            text,
            fill=text_color,
        )

    return result


def image_to_base64(image):
    """Convierte una imagen PIL a base64 para mostrarla en el navegador."""
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Recibe la imagen, ejecuta detección y retorna resultados al frontend."""
    if "file" not in request.files:
        return jsonify(error="No se recibió archivo"), 400

    file = request.files["file"]

    if not file.filename or not allowed_file(file.filename):
        return jsonify(error="Formato no válido"), 400

    extension = file.filename.rsplit(".", 1)[1].lower()
    file_id = str(uuid.uuid4())[:8]

    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        f"{file_id}.{extension}",
    )

    file.save(file_path)

    params = {}

    for key in [
        "peak_spacing",
        "val_max",
        "texture_thr",
        "exg_min",
        "erosion_iter",
        "min_radius",
        "gauss_sigma",
    ]:
        value = request.form.get(key)

        if value:
            params[key] = float(value) if "." in value else int(value)

    try:
        trees, result_image, stats = detect_trees(file_path, params)
    except Exception as error:
        return jsonify(error=str(error)), 500

    result_path = os.path.join(
        app.config["RESULTS_FOLDER"],
        f"{file_id}_result.jpg",
    )

    json_path = os.path.join(
        app.config["RESULTS_FOLDER"],
        f"{file_id}_trees.json",
    )

    result_image.save(result_path, quality=90)

    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(
            dict(trees=trees, stats=stats),
            json_file,
            ensure_ascii=False,
            indent=2,
        )

    # Imagen original reducida para previsualización.
    original_image = Image.open(file_path).convert("RGB")

    if max(original_image.size) > 1800:
        resize_scale = 1800 / max(original_image.size)
        original_image = original_image.resize(
            (
                int(original_image.width * resize_scale),
                int(original_image.height * resize_scale),
            ),
            Image.LANCZOS,
        )

    return jsonify(
        ok=True,
        fid=file_id,
        trees=trees,
        stats=stats,
        orig_b64=image_to_base64(original_image),
        result_b64=image_to_base64(result_image),
    )


@app.route("/csv/<file_id>")
def download_csv(file_id):
    """Descarga los resultados como CSV."""
    json_path = os.path.join(
        app.config["RESULTS_FOLDER"],
        f"{file_id}_trees.json",
    )

    if not os.path.exists(json_path):
        return "Not found", 404

    with open(json_path, encoding="utf-8") as json_file:
        data = json.load(json_file)

    rows = ["ID,Label,Source,CentroX,CentroY,Radio,BboxW,BboxH,ExG,Health"]

    for tree in data["trees"]:
        rows.append(
            f"{tree['id']},"
            f"{tree['label']},"
            f"{tree.get('source', 'auto')},"
            f"{tree['cx']},"
            f"{tree['cy']},"
            f"{tree['radius']},"
            f"{tree['x1'] - tree['x0']},"
            f"{tree['y1'] - tree['y0']},"
            f"{tree['exg']},"
            f"{tree['health']}"
        )

    return Response(
        "\n".join(rows),
        mimetype="text/csv",
        headers={
            "Content-Disposition": f"attachment;filename=arboles_{file_id}.csv"
        },
    )


@app.route("/health")
def health():
    """Ruta simple para comprobar que la app está viva en Render."""
    return jsonify(status="ok")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug_mode, host="0.0.0.0", port=port)
