"""Desktop launcher that runs Streamlit inside a native pywebview window."""

import multiprocessing
import os
import socket
import subprocess
import sys
import time
import urllib.request

from streamlit.web import cli as stcli

SERVER_MODE_ENV = "GP_SERVER_MODE"
PORT_ENV = "GP_STREAMLIT_PORT"


def get_app_path() -> str:
    if getattr(sys, "frozen", False):
        return os.path.join(sys._MEIPASS, "AkShareDemo.py")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "gp", "AkShareDemo.py")


def _port_is_free(port: int) -> bool:
    for family, address in [
        (socket.AF_INET, ("0.0.0.0", port)),
        (socket.AF_INET6, ("::", port)),
    ]:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(address)
            except OSError:
                return False
    return True


def find_free_port(start=8501, attempts=100) -> str:
    for port in range(start, start + attempts):
        if _port_is_free(port):
            return str(port)
    return str(start)


def build_streamlit_argv(port: str, headless: bool = True):
    return [
        "streamlit",
        "run",
        get_app_path(),
        "--server.address",
        "127.0.0.1",
        "--server.headless",
        "true" if headless else "false",
        "--server.port",
        port,
        "--global.developmentMode",
        "false",
        "--browser.gatherUsageStats",
        "false",
    ]


def build_child_command(port: str):
    if getattr(sys, "frozen", False):
        return [sys.executable]
    return [sys.executable, os.path.abspath(__file__)]


def run_server(port: str) -> int:
    sys.argv = build_streamlit_argv(port, headless=True)
    return stcli.main()


def wait_for_server(url: str, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def run_webview(port: str) -> int:
    import webview

    env = os.environ.copy()
    env[SERVER_MODE_ENV] = "1"
    env[PORT_ENV] = port

    popen_kwargs = {}
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    child = subprocess.Popen(build_child_command(port), env=env, **popen_kwargs)
    url = f"http://127.0.0.1:{port}"

    try:
        if not wait_for_server(url):
            raise RuntimeError("Streamlit 服务启动超时")
        webview.create_window(
            "股票技术分析",
            url,
            width=1280,
            height=860,
            min_size=(900, 600),
        )
        webview.start()
    finally:
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()

    return 0


def main() -> int:
    port = os.environ.get(PORT_ENV) or find_free_port()
    if os.environ.get(SERVER_MODE_ENV) == "1":
        return run_server(port)
    return run_webview(port)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
