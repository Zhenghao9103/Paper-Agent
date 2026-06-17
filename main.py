import os
import threading
import time
import webbrowser

import uvicorn


def open_browser() -> None:
    if os.environ.get("PAPER_AGENT_OPEN_BROWSER", "1") == "0":
        return
    time.sleep(1)
    webbrowser.open("http://localhost:8000")


if __name__ == "__main__":
    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=False)
