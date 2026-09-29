import atexit
import os
import socket
import threading
import time
import webbrowser

import uvicorn

from backend.app.core.config import get_settings


def port_is_in_use(host: str, port: int) -> bool:
    """Return whether another local service already owns the listen port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.settimeout(0.25)
        return client.connect_ex((host, port)) == 0


def open_browser() -> None:
    if os.environ.get("PAPER_AGENT_OPEN_BROWSER", "1") == "0":
        return
    time.sleep(1)
    webbrowser.open("http://localhost:8000")


def start_managed_router():
    """Start the local llama.cpp Router when ROUTER_MANAGED is enabled."""

    from pathlib import Path

    from backend.app.services.router_server import ManagedRouterServer, RouterServerError

    settings = get_settings()
    if settings.router_provider != "local" or not settings.router_managed:
        return None
    try:
        managed = ManagedRouterServer.from_settings(
            log_dir=Path("logs") / "router-server"
        )
        base_url = managed.start()
    except (RouterServerError, OSError) as exc:
        print(f"本地 Router 启动失败: {exc}")
        raise SystemExit(2) from exc
    atexit.register(managed.stop)
    print(f"本地 Router 已就绪: {base_url} ({settings.resolved_router_model})")
    return managed


if __name__ == "__main__":
    if port_is_in_use("127.0.0.1", 8000):
        print("PaperMind 已在 http://127.0.0.1:8000 运行，请勿重复启动。")
        raise SystemExit(0)
    managed_router = start_managed_router()
    try:
        threading.Thread(target=open_browser, daemon=True).start()
        uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=False)
    finally:
        if managed_router is not None:
            managed_router.stop()
