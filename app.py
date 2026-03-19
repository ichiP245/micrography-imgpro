import os
import zipfile
import io
import json
from typing import Dict, Any, Tuple

import matplotlib
matplotlib.use('Agg')

import cv2
cv2.setUseOptimized(True)
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

# Existing pipeline modules
from getmeresults import getMeResults
import getmepores as gmp
import getmeflashes as gmfl
import getmefibers as gmf
from common import getSegmentationFigure

# ---------- Helpers ----------

def fig_to_img(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = buf.reshape((h, w, 4))[:, :, :3]
    plt.close(fig)
    return img

def png_bytes(img: np.ndarray) -> bytes:
    if img is None:
        return b""
    ok, buf = cv2.imencode(".png", to_uint8(img))
    return buf.tobytes() if ok else b""

def png_bytes_bgr(img: np.ndarray) -> bytes:
    if img is None:
        return b""
    img = cv2.cvtColor(to_uint8(img), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes() if ok else b""

def name_only_from_source(source: str) -> str:
    base = os.path.basename(source) if source else "image"
    return os.path.splitext(base)[0] or "image"

def to_uint8(img: np.ndarray) -> np.ndarray:
    if img is None:
        return img
    if img.dtype == np.bool_:
        return img.astype(np.uint8) * 255
    if np.issubdtype(img.dtype, np.floating):
        return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.clip(img, 0, 255).astype(np.uint8)

def decode_uploaded_gray(upload) -> np.ndarray:
    upload.seek(0)
    data = np.frombuffer(upload.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not decode {upload.name}")
    return img

def to_rgb_for_display(gray_or_bgr: np.ndarray) -> np.ndarray:
    if gray_or_bgr.ndim == 2:
        return cv2.cvtColor(gray_or_bgr, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(gray_or_bgr, cv2.COLOR_BGR2RGB)

def normalize_mask_for_display(mask: np.ndarray) -> np.ndarray:
    if mask is None:
        return mask
    if mask.ndim == 3:
        return to_rgb_for_display(mask)
    m = mask.astype(np.float32)
    mn, mx = float(np.min(m)), float(np.max(m))
    if mx <= mn:
        return np.zeros_like(mask, dtype=np.uint8)
    return ((m - mn) / (mx - mn) * 255.0).clip(0, 255).astype(np.uint8)

def make_preview_image(img: np.ndarray, downscale_size: int = 1000) -> np.ndarray:
    h, w = img.shape[:2]
    side = min(int(downscale_size), h, w)

    cy, cx = h // 2, w // 2
    half = side // 2

    y1 = max(0, cy - half)
    x1 = max(0, cx - half)
    y2 = y1 + side
    x2 = x1 + side

    if y2 > h:
        y2 = h
        y1 = h - side
    if x2 > w:
        x2 = w
        x1 = w - side

    return img[y1:y2, x1:x2].copy()

def as_odd(n: int) -> int:
    return n if (n % 2 == 1) else n + 1

def get_default_params() -> Dict[str, Any]:
    return {
        "otsu_classes": 5,
        "otsu_range": (0, 4),
        "first_kernel_size": (5, 5),
        "second_kernel_size": (3, 3),
        "gamma": 1.0,
        "gain": 1.0,
        "cont_mult": 2.5,
    }

def build_parameters_ui(p: Dict[str, Any], key_suffix: str) -> Dict[str, Any]:
    st.subheader("Parameters")

    with st.expander("Advanced Settings", expanded=False):
        gamma = st.slider("gamma (fibers)", 0.1, 5.0, float(p.get("gamma", 1.0)), 0.1, key=f"gam_{key_suffix}")
        gain = st.slider("gamma gain (fibers)", 0.1, 5.0, float(p.get("gain", 1.0)), 0.1, key=f"gain_{key_suffix}")
        o_classes = st.slider("Multi-Otsu Classes", 2, 10, p.get("otsu_classes", 5), key=f"ots_c_{key_suffix}")
        curr_range = p.get("otsu_range", (0, 4))
        safe_range = (min(curr_range[0], o_classes - 1), min(curr_range[1], o_classes - 1))
        o_range = st.slider("Class Range", 0, o_classes - 1, safe_range, key=f"ots_r_{key_suffix}")

    return {
        "otsu_classes": int(o_classes),
        "otsu_range": o_range,
        "first_kernel_size": p.get("first_kernel_size", (5, 5)),
        "second_kernel_size": p.get("second_kernel_size", (3, 3)),
        "gamma": gamma,
        "gain": gain,
        "cont_mult": p.get("cont_mult", 2.5),
    }

def run_pipeline(base_img_gray: np.ndarray, parameters: Dict[str, Any]):
    outputs: Dict[str, Any] = {}
    try:
        stats, segmentation, coloring = getMeResults(base_img_gray, parameters)
        fig, ax = plt.subplots(figsize=(10, 6))
        getSegmentationFigure(segmentation, stats, "out", ax=ax)
        outputs["results"] = fig_to_img(fig)
        outputs["stats"] = stats

        _, _, _, fiber_steps = gmf.getMeFibersGammaOtsuWatershed(
            base_img_gray,
            gamma=parameters.get("gamma", 1.0),
            gain=parameters.get("gain", 1.0),
            otsu_classes=parameters["otsu_classes"],
            otsu_range=parameters["otsu_range"],
            return_steps=True,
        )
        outputs["fiber_steps"] = fiber_steps
    except Exception as e:
        st.error(f"Pipeline error: {e}")
    return outputs

def serialize_stats_for_export(stats, fname, params):
    if not stats:
        return "stats.csv", b""

    row = {}
    row["file"] = fname

    for k, v in params.items():
        row[f"param_{k}"] = v

    for k, v in stats.items():
        row[k] = v

    df = pd.DataFrame([row])
    return "stats.csv", df.to_csv(index=False).encode("utf-8")

def get_exportable_items(outputs: Dict[str, Any], name_only: str, export_result: bool, export_data: bool) -> list[Tuple[str, bytes]]:
    items = []
    if not outputs:
        return items

    if export_result and "results" in outputs:
        items.append((f"{name_only}_results.png", png_bytes_bgr(outputs["results"])))

    if export_data and "stats" in outputs:
        stats_fname, stats_bytes = serialize_stats_for_export(outputs["stats"], name_only, {})
        items.append((f"{name_only}_{stats_fname}", stats_bytes))

    return items

# ---------- UI Execution ----------
st.set_page_config(page_title="Batch Image Processor", layout="wide")

if "img_data" not in st.session_state:
    st.session_state.img_data = {}

with st.sidebar:
    st.header("Upload")
    uploaded_files = st.file_uploader("Select images", type=["png", "jpg", "tif"], accept_multiple_files=True)

    if uploaded_files is not None:
        current_names = [f.name for f in uploaded_files]
        st.session_state.img_data = {
            k: v for k, v in st.session_state.img_data.items()
            if k in current_names
        }

    for k in st.session_state.img_data:
        st.session_state.img_data[k]["preview_image"] = make_preview_image(
            st.session_state.img_data[k]["image"],
            st.session_state.get("downscale_size", 1000)
        )

    if uploaded_files:
        for f in uploaded_files:
            if f.name not in st.session_state.img_data:
                img = decode_uploaded_gray(f)
                st.session_state.img_data[f.name] = {
                    "image": img,
                    "preview_image": make_preview_image(
                        img,
                        st.session_state.get("downscale_size", 1000)
                    ),
                    "params": get_default_params(),
                    "export_result": True,
                    "export_data": False,
                    "outputs": None
                }

    if st.session_state.img_data:
        st.divider()
        st.header("Selection & Tuning")
        active_file = st.selectbox("Pick image to edit:", list(st.session_state.img_data.keys()))

        data = st.session_state.img_data[active_file]

        st.caption("Processing Mode: All Results")

        st.subheader("Export Options")
        data["export_result"] = st.checkbox("Result", value=data.get("export_result", True), key=f"export_result_{active_file}")
        data["export_data"] = st.checkbox("Data (stats)", value=data.get("export_data", False), key=f"export_data_{active_file}")

        data["params"] = build_parameters_ui(data["params"], active_file)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Preview"):
                data["outputs"] = run_pipeline(data["preview_image"], data["params"])
        with col2:
            if st.button("Apply to All"):
                for k in st.session_state.img_data:
                    st.session_state.img_data[k]["params"] = data["params"].copy()
                    st.session_state.img_data[k]["export_result"] = data["export_result"]
                    st.session_state.img_data[k]["export_data"] = data["export_data"]
                st.success("Applied to all!")

        st.slider("Downscale size", 100, 2000, 1000, 50, key="downscale_size")

        st.divider()
        st.header("Batch Export")
        if st.button("🚀 Process & Download All (ZIP)", type="primary"):
            zip_buffer = io.BytesIO()
            prog = st.progress(0)
            status = st.empty()

            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                for idx, (fname, d) in enumerate(st.session_state.img_data.items()):
                    status.text(f"Processing {fname}...")
                    out = run_pipeline(d["image"], d["params"])
                    name_only = name_only_from_source(fname)

                    if d.get("export_result", True) and out and "results" in out:
                        zip_file.writestr(
                            f"{name_only}_result.png",
                            png_bytes_bgr(out["results"])
                        )

                    if d.get("export_data", False) and out and "stats" in out:
                        stats_fname, stats_bytes = serialize_stats_for_export(out["stats"], name_only, d["params"])
                        zip_file.writestr(
                            f"{name_only}_{stats_fname}",
                            stats_bytes
                        )

                    prog.progress((idx + 1) / len(st.session_state.img_data))

            status.text("Done!")
            st.download_button(
                "Download ZIP",
                zip_buffer.getvalue(),
                "batch_results.zip",
                "application/zip"
            )

# ---------- Main Panel ----------
if st.session_state.img_data and 'active_file' in locals():
    data = st.session_state.img_data[active_file]
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader(f"Input: {active_file}")
        st.image(to_rgb_for_display(data["image"]), width="stretch")

    with col_r:
        st.subheader("Output Preview")
        if data["outputs"]:
            if isinstance(data["outputs"].get("results"), np.ndarray):
                st.image(data["outputs"]["results"], caption="results", width="stretch")
            if "fiber_steps" in data["outputs"]:
                st.subheader("Fiber Filter Steps")
                for step_name, step_img in data["outputs"]["fiber_steps"]:
                    st.image(
                        normalize_mask_for_display(step_img),
                        caption=step_name,
                        width="stretch"
                    )
        else:
            st.info("Adjust parameters and click 'Preview'.")
else:
    st.info("Please upload images in the sidebar.")
