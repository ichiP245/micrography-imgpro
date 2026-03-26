# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all


streamlit_datas, streamlit_binaries, streamlit_hiddenimports = collect_all("streamlit")
matplotlib_datas, matplotlib_binaries, matplotlib_hiddenimports = collect_all("matplotlib")
altair_datas, altair_binaries, altair_hiddenimports = collect_all("altair")
pydeck_datas, pydeck_binaries, pydeck_hiddenimports = collect_all("pydeck")
skimage_datas, skimage_binaries, skimage_hiddenimports = collect_all("skimage")

datas = [
    ("app.py", "."),
    ("common.py", "."),
    ("getmefibers.py", "."),
    ("getmeflashes.py", "."),
    ("getmepores.py", "."),
    ("getmeresults.py", "."),
] + streamlit_datas + matplotlib_datas + altair_datas + pydeck_datas + skimage_datas

binaries = (
    streamlit_binaries
    + matplotlib_binaries
    + altair_binaries
    + pydeck_binaries
    + skimage_binaries
)

hiddenimports = (
    streamlit_hiddenimports
    + matplotlib_hiddenimports
    + altair_hiddenimports
    + pydeck_hiddenimports
    + skimage_hiddenimports
)


a = Analysis(
    ["run_app.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="micrography-imgpro",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
