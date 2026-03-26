import os
import sys
from streamlit.web.cli import main

if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

APP_PATH = os.path.join(BASE_DIR, "app.py")

if __name__ == "__main__":
    sys.argv = [
        "streamlit",
        "run",
        APP_PATH,
        "--global.developmentMode=false",
        "--server.headless=false",
    ]
    raise SystemExit(main())
