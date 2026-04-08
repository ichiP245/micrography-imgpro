import io
import os
import zipfile
from typing import Any, Callable, Dict, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from common import getSegmentationFigure
import getmefibers as gmf
from getmeresults import getMeResults, getMeResultsSimple

matplotlib.use("Agg")
cv2.setUseOptimized(True)


ModeConfig = Dict[str, Any]


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


def get_simple_default_params() -> Dict[str, Any]:
    return {
        "otsu_classes": 5,
        "otsu_range": (3, 4),
        "first_kernel_size": (5, 5),
        "second_kernel_size": (3, 3),
        "gamma": 1.0,
        "cont_mult": 2.5,
        "ws_ths_factor": 0.025,
        "ws_gl_vecinity": 15,
    }


def get_pro_default_params() -> Dict[str, Any]:
    return {
        "otsu_classes": 5,
        "otsu_range": (3, 4),
        "first_kernel_size": (5, 5),
        "second_kernel_size": (3, 3),
        "bh_ks": (7, 7),
        "bhm_iter": 4,
        "bhm_mult": 60,
        "cont_mult": 2.5,
        "ws_ths_factor": 0.025,
        "ws_gl_vecinity": 15,
    }


def build_simple_parameters_ui(p: Dict[str, Any], key_suffix: str) -> Dict[str, Any]:
    st.subheader("Parameters")

    with st.expander("Advanced Settings", expanded=False):
        gamma = st.slider("gamma (fibers)", 0.1, 5.0, float(p.get("gamma", 1.0)), 0.1, key=f"gam_{key_suffix}")
        o_classes = st.slider("Multi-Otsu Classes", 2, 10, p.get("otsu_classes", 5), key=f"ots_c_{key_suffix}")
        curr_range = p.get("otsu_range", (0, 4))
        safe_range = (min(curr_range[0], o_classes - 1), min(curr_range[1], o_classes - 1))
        o_range = st.slider("Class Range", 0, o_classes - 1, safe_range, key=f"ots_r_{key_suffix}")
        ws_ths_factor = st.slider(
            "ws_ths_factor",
            0.0001,
            0.2,
            float(p.get("ws_ths_factor", 0.025)),
            0.0005,
            format="%.4f",
            key=f"wsf_{key_suffix}",
        )
        ws_gl_vecinity = st.slider("ws_gl_vecinity", 1, 200, p.get("ws_gl_vecinity", 15), 1, key=f"wsv_{key_suffix}")

    return {
        "otsu_classes": int(o_classes),
        "otsu_range": o_range,
        "first_kernel_size": p.get("first_kernel_size", (5, 5)),
        "second_kernel_size": p.get("second_kernel_size", (3, 3)),
        "gamma": gamma,
        "cont_mult": p.get("cont_mult", 2.5),
        "ws_ths_factor": ws_ths_factor,
        "ws_gl_vecinity": ws_gl_vecinity,
    }


def build_pro_parameters_ui(p: Dict[str, Any], key_suffix: str) -> Dict[str, Any]:
    st.subheader("Parameters")

    with st.expander("Advanced Settings", expanded=False):
        st.caption(" Black Hat / Fibers Enhanced")
        bh = st.slider("bh_ks (odd)", 1, 61, p.get("bh_ks", (7, 7))[0], 2, key=f"bh_{key_suffix}")
        bhm_iter = st.slider("bhm_iter", 1, 20, p.get("bhm_iter", 4), 1, key=f"bmi_{key_suffix}")
        bhm_mult = st.slider("bhm_mult", 1, 300, p.get("bhm_mult", 60), 1, key=f"bmm_{key_suffix}")
        
        st.caption("Multi-Otsu Selection")
        o_classes = st.slider("Multi-Otsu Classes", 2, 10, p.get("otsu_classes", 5), key=f"ots_c_{key_suffix}")
        curr_range = p.get("otsu_range", (0, 4))
        safe_range = (min(curr_range[0], o_classes - 1), min(curr_range[1], o_classes - 1))
        o_range = st.slider("Class Range", 0, o_classes - 1, safe_range, key=f"ots_r_{key_suffix}")

        st.caption("Contours / Flood Mask")
        cont_mult = st.slider("cont_mult (fibers)", 0.1, 10.0, float(p.get("cont_mult", 2.5)), 0.1, key=f"cmf_{key_suffix}")

        st.caption("Watershed")
        ws_ths_factor = st.slider(
            "ws_ths_factor",
            0.0001,
            0.2,
            float(p.get("ws_ths_factor", 0.025)),
            0.0005,
            format="%.4f",
            key=f"wsf_{key_suffix}",
        )
        ws_gl_vecinity = st.slider("ws_gl_vecinity", 1, 200, p.get("ws_gl_vecinity", 15), 1, key=f"wsv_{key_suffix}")

    return {
        "otsu_classes": int(o_classes),
        "otsu_range": o_range,
        "first_kernel_size": p.get("first_kernel_size", (5, 5)),
        "second_kernel_size": p.get("second_kernel_size", (3, 3)),
        "bh_ks": (as_odd(bh), as_odd(bh)),
        "bhm_iter": bhm_iter,
        "bhm_mult": bhm_mult,
        "cont_mult": cont_mult,
        "ws_ths_factor": ws_ths_factor,
        "ws_gl_vecinity": ws_gl_vecinity,
    }


def run_simple_pipeline(base_img_gray: np.ndarray, parameters: Dict[str, Any]) -> Dict[str, Any]:
    outputs: Dict[str, Any] = {}
    try:
        stats, segmentation, coloring = getMeResultsSimple(base_img_gray, parameters)
        fig, ax = plt.subplots(figsize=(10, 6))
        getSegmentationFigure(segmentation, stats, "out", ax=ax)
        outputs["results"] = fig_to_img(fig)
        outputs["coloring"] = coloring
        outputs["stats"] = stats

        _, _, _, fiber_steps = gmf.getMeFibersGammaOtsuWatershed(
            base_img_gray,
            gamma=parameters.get("gamma", 1.0),
            ws_ths_factor=parameters.get("ws_ths_factor", 0.025),
            ws_gl_vecinity=parameters.get("ws_gl_vecinity", 15),
            otsu_classes=parameters["otsu_classes"],
            otsu_range=parameters["otsu_range"],
            return_steps=True,
        )
        outputs["debug_images"] = dict(fiber_steps)
    except Exception as e:
        st.error(f"Pipeline error: {e}")
    return outputs


def run_pro_pipeline(base_img_gray: np.ndarray, parameters: Dict[str, Any]) -> Dict[str, Any]:
    outputs: Dict[str, Any] = {}
    try:
        stats, segmentation, coloring, debug_images = getMeResults(base_img_gray, parameters, return_debug=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        getSegmentationFigure(segmentation, stats, "out", ax=ax)
        outputs["results"] = fig_to_img(fig)
        outputs["coloring"] = coloring
        outputs["stats"] = stats
        outputs["debug_images"] = debug_images
    except Exception as e:
        st.error(f"Pipeline error: {e}")
    return outputs


def serialize_stats_for_export(stats, fname, params):
    if not stats:
        return "stats.csv", b""

    row = {"file": fname}
    for k, v in params.items():
        row[f"param_{k}"] = v
    for k, v in stats.items():
        row[k] = v

    df = pd.DataFrame([row])
    return "stats.csv", df.to_csv(index=False).encode("utf-8")


def render_output_preview(outputs: Dict[str, Any], show_debug: bool):
    if not outputs:
        st.info("Adjust parameters and click 'Preview'.")
        return

    for k, v in outputs.items():
        if k in {"coloring", "debug_images", "stats"}:
            continue
        if isinstance(v, np.ndarray):
            st.image(
                normalize_mask_for_display(v) if "mask" in k or "binary" in k else v,
                caption=k,
                width="stretch",
            )

    if show_debug:
        debug_images = outputs.get("debug_images", {})
        if debug_images:
            st.subheader("Debug Images")
            _, debug_col, _ = st.columns([1, 2, 1])
            with debug_col:
                for k, v in debug_images.items():
                    if isinstance(v, np.ndarray):
                        st.image(
                            normalize_mask_for_display(v) if "mask" in k or "binary" in k else v,
                            caption=k,
                            width="stretch",
                        )
        else:
            st.info("No debug images available for this preview.")


def render_mode(config: ModeConfig):
    img_data_key = config["img_data_key"]
    downscale_key = config["downscale_key"]
    build_parameters_ui: Callable[[Dict[str, Any], str], Dict[str, Any]] = config["build_parameters_ui"]
    get_default_params: Callable[[], Dict[str, Any]] = config["get_default_params"]
    run_pipeline: Callable[[np.ndarray, Dict[str, Any]], Dict[str, Any]] = config["run_pipeline"]

    if img_data_key not in st.session_state:
        st.session_state[img_data_key] = {}

    with st.sidebar:
        st.header("Upload")
        uploaded_files = st.file_uploader(
            "Select images",
            type=["png", "jpg", "tif"],
            accept_multiple_files=True,
            key=f"uploader_{img_data_key}",
        )

        if uploaded_files is not None:
            current_names = [f.name for f in uploaded_files]
            st.session_state[img_data_key] = {
                k: v for k, v in st.session_state[img_data_key].items()
                if k in current_names
            }

        for k in st.session_state[img_data_key]:
            st.session_state[img_data_key][k]["preview_image"] = make_preview_image(
                st.session_state[img_data_key][k]["image"],
                st.session_state.get(downscale_key, 1000),
            )

        if uploaded_files:
            for f in uploaded_files:
                if f.name not in st.session_state[img_data_key]:
                    img = decode_uploaded_gray(f)
                    st.session_state[img_data_key][f.name] = {
                        "image": img,
                        "preview_image": make_preview_image(img, st.session_state.get(downscale_key, 1000)),
                        "params": get_default_params(),
                        "export_result": True,
                        "export_coloring": False,
                        "export_data": False,
                        "outputs": None,
                    }

        if st.session_state[img_data_key]:
            st.divider()
            st.header("Selection & Tuning")
            active_file = st.selectbox(
                "Pick image to edit:",
                list(st.session_state[img_data_key].keys()),
                key=f"active_{img_data_key}",
            )
            data = st.session_state[img_data_key][active_file]

            st.caption(f"Processing Mode: {config['label']}")

            st.subheader("Export Options")
            data["export_result"] = st.checkbox(
                "Result",
                value=data.get("export_result", True),
                key=f"export_result_{img_data_key}_{active_file}",
            )
            data["export_coloring"] = st.checkbox(
                "Coloring",
                value=data.get("export_coloring", False),
                key=f"export_coloring_{img_data_key}_{active_file}",
            )
            data["export_data"] = st.checkbox(
                "Data (stats)",
                value=data.get("export_data", False),
                key=f"export_data_{img_data_key}_{active_file}",
            )

            data["params"] = build_parameters_ui(data["params"], f"{img_data_key}_{active_file}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Preview", key=f"preview_{img_data_key}"):
                    data["outputs"] = run_pipeline(data["preview_image"], data["params"])
            with col2:
                if st.button("Apply to All", key=f"apply_all_{img_data_key}"):
                    for k in st.session_state[img_data_key]:
                        st.session_state[img_data_key][k]["params"] = data["params"].copy()
                        st.session_state[img_data_key][k]["export_result"] = data["export_result"]
                        st.session_state[img_data_key][k]["export_coloring"] = data["export_coloring"]
                        st.session_state[img_data_key][k]["export_data"] = data["export_data"]
                    st.success("Applied to all!")

            st.slider("Downscale size", 100, 2000, 1000, 50, key=downscale_key)

            st.divider()
            st.header("Batch Export")
            if st.button("Process & Download All (ZIP)", type="primary", key=f"download_{img_data_key}"):
                zip_buffer = io.BytesIO()
                prog = st.progress(0)
                status = st.empty()

                with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                    total_files = len(st.session_state[img_data_key])
                    for idx, (fname, item_data) in enumerate(st.session_state[img_data_key].items(), start=1):
                        status.text(f"Processing {fname}...")
                        out = run_pipeline(item_data["image"], item_data["params"])
                        name_only = name_only_from_source(fname)

                        if item_data.get("export_result", True) and out and "results" in out:
                            zip_file.writestr(f"{name_only}_result.png", png_bytes_bgr(out["results"]))

                        if item_data.get("export_coloring", False) and out and "coloring" in out:
                            zip_file.writestr(f"{name_only}_coloring.png", png_bytes(out["coloring"]))

                        if item_data.get("export_data", False) and out and "stats" in out:
                            stats_fname, stats_bytes = serialize_stats_for_export(out["stats"], name_only, item_data["params"])
                            zip_file.writestr(f"{name_only}_{stats_fname}", stats_bytes)

                        prog.progress(idx / total_files)

                status.text("Done!")
                st.download_button(
                    "Download ZIP",
                    zip_buffer.getvalue(),
                    "batch_results.zip",
                    "application/zip",
                    key=f"download_button_{img_data_key}",
                )

    if st.session_state[img_data_key]:
        active_file = st.session_state.get(f"active_{img_data_key}")
        if active_file and active_file in st.session_state[img_data_key]:
            data = st.session_state[img_data_key][active_file]
            col_l, col_r = st.columns(2)

            with col_l:
                st.subheader(f"Input: {active_file}")
                st.image(to_rgb_for_display(data["image"]), width="stretch")
                show_debug = st.checkbox("Show debug images", value=False, key=f"show_debug_{img_data_key}_{active_file}")

            with col_r:
                st.subheader("Output Preview")
                render_output_preview(data["outputs"], show_debug)
            return

    st.info("Please upload images in the sidebar.")
    st.markdown(
        """
        The app has two workflows in the sidebar:

        - `Simple`: a lighter workflow with fewer controls, centered on gamma correction,
          Multi-Otsu selection, and watershed refinement.
        - `Pro`: a more detailed workflow with extra preprocessing controls for black-hat
          enhancement and contour filtering.

        In both modes, the app lets you preview the result on a cropped region and then
        batch-process the full images.

        I suggest trying the Simple workflow first, and only moving to Pro if the analysis
        turns out really bad. You can absolutely get the hang of Pro and learn it properly,
        but it has a steeper learning curve.

        [Documentation](https://docs.google.com/document/d/1poxKbw4yWZf-ew6A376iOljxoanXr6GMMZizSAl_muk/edit?usp=drive_link)
        | [Tutorial Video](https://youtu.be/Gdlq5muXD2s)
        """
    )


st.set_page_config(page_title="Micrography Image Processor", layout="wide")
st.title("Micrography Image Processor")

with st.sidebar:
    st.header("Mode")
    selected_mode = st.radio("Choose workflow", ["Simple", "Pro"], key="selected_mode")

if selected_mode == "Simple":
    render_mode(
        {
            "label": "Simple",
            "img_data_key": "img_data_simple",
            "downscale_key": "downscale_size_simple",
            "get_default_params": get_simple_default_params,
            "build_parameters_ui": build_simple_parameters_ui,
            "run_pipeline": run_simple_pipeline,
        }
    )
else:
    render_mode(
        {
            "label": "Pro",
            "img_data_key": "img_data_pro",
            "downscale_key": "downscale_size_pro",
            "get_default_params": get_pro_default_params,
            "build_parameters_ui": build_pro_parameters_ui,
            "run_pipeline": run_pro_pipeline,
        }
    )
