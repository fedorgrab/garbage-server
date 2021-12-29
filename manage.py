import sys
import uvicorn
from garbage_server.main import app


if __name__ == "__main__":
    if sys.argv[1] == "runserver":
        uvicorn.run(app, host="0.0.0.0", port=80, log_level="info")
        
