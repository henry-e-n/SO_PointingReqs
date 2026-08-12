from pathlib import Path

paths = []
try:
    REPO_DIR = Path(__file__).resolve().parent
    HOME_DIR = REPO_DIR.home()
    SMARTIES_CLONE = HOME_DIR / "smarties"
    SMARTIES_DIR = SMARTIES_CLONE / "smarties"
    
    paths = [REPO_DIR, HOME_DIR, SMARTIES_CLONE, SMARTIES_DIR]

except Exception as e:
    print(f"An error occurred: {e}")