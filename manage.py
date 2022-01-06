import os
import sys
import uvicorn

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CURR_DIR}/garbage_server")

from garbage_server.main import app

if __name__ == "__main__":
    if sys.argv[1] == "runserver":
        uvicorn.run(app, host="0.0.0.0", port=80, log_level="info")
