from pathlib import Path
import runpy

app_file = Path(__file__).parent / "app" / "app.py"
runpy.run_path(str(app_file), run_name="__main__")
