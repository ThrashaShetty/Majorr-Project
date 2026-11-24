"""uvicorn entry point: python -B -m uvicorn backend.extensions.serve:app --host 127.0.0.1 --port 8000
Importing this module builds the app and starts loading SAM in a background thread."""
from backend.extensions.app import create_ext_app

app = create_ext_app()
