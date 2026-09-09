import socket

from main import port_is_in_use


def test_port_check_detects_an_existing_server() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]

        assert port_is_in_use("127.0.0.1", port) is True


def test_port_check_accepts_an_unused_port() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

    assert port_is_in_use("127.0.0.1", port) is False
