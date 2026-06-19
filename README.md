# micrography-imgpro
This is a powerful **Python application** engineered to calculate the precise structural proportions between **fiber, resin, and pores** in composite materials. 

By utilizing advanced deterministic image processing filters, the tool automates material analysis without the need for manual guesswork.

| Micrography | Processed Image |
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/6b79272a-55cc-48ea-8be9-601715b4262d" width="100%" alt="First Image"> | <img src="https://github.com/user-attachments/assets/e2b2d808-e3a7-4932-89cb-83b4ab9db153" width="100%" alt="Second Image"> |

The app has two workflows in the sidebar:

- `Simple`: a lighter workflow with fewer controls, centered on gamma correction, Multi-Otsu selection, and watershed refinement.
- `Pro`: a more detailed workflow with extra preprocessing controls for black-hat enhancement and contour filtering.

In both modes, the app lets you preview the result on a cropped region and then batch-process the full images.

I suggest trying to use the simple version first and only if the analisis turns out really bad use the pro version, you can get the hang of it and learn it properly but it's a steep learning curve. 

<img width="1867" height="872" alt="image" src="https://github.com/user-attachments/assets/475882e4-4a07-4ed9-ad48-4f8525ca55f2" />


YouTube tutorial/showcase: https://youtu.be/Gdlq5muXD2s


## What is included

- `app.py`: Single Streamlit web UI with a sidebar mode switch for Simple and Pro workflows.
- `controller.py`: command-line batch runner.
- `getmeresults.py`, `getmefibers.py`, `getmeflashes.py`, `getmepores.py`: image processing pipeline modules.
- `run_app.py`: launcher used for PyInstaller builds.
- `micrography-imgpro.spec`: PyInstaller spec file for building the executable.

## Requirements

- Python 3.10 or newer is recommended.
- Install the packages listed in [`requirements.txt`](requirements.txt).

## Install

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the app

Start the Streamlit UI with:

```powershell
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Use the CLI

Run the batch processor from the terminal:

```powershell
python .\controller.py
```

Optional flags:

- `-f`: run fibers only
- `-fl`: run flashes only
- `-p`: run pores only

Example:

```powershell
python .\controller.py -f -p
```

Running without flags processes the full combined result pipeline and writes output files to `processed_results`.

## Build a single executable

This project includes [`run_app.py`](run_app.py) and [`micrography-imgpro.spec`](micrography-imgpro.spec) for packaging the Streamlit app with PyInstaller.

Already compiled in:
```
https://drive.google.com/drive/folders/19tp-wSOcunXumASR2pq6P5RpKWenxGwO?usp=drive_link
```

Install PyInstaller:

```powershell
pip install pyinstaller
```

`requirements.txt` installs Python dependencies for the app. PyInstaller still needs the spec file to bundle the Streamlit launcher, app source files, and runtime assets into an executable.

Build the executable from the project root:

```powershell
pyinstaller --clean micrography-imgpro.spec
```

After the build completes, the executable will be created at:

```text
dist\micrography-imgpro.exe
```

Run it from a terminal the first time so any startup errors stay visible:

```powershell
.\dist\micrography-imgpro.exe
```

## Notes

- The packaged app still launches a local Streamlit server and opens in the browser.
- Do not commit `dist/` or `build/` outputs to GitHub unless you are using Git LFS.
