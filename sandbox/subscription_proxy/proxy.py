"""Small CONNECT-only egress proxy for subscription CLI provider traffic."""

from __future__ import annotations

import select
import socket
import socketserver


ALLOWED_SUFFIXES = (
    "anthropic.com",
    "claude.ai",
    "openai.com",
    "chatgpt.com",
)
MAX_HEADER_BYTES = 64 * 1024


def allowed_host(host: str) -> bool:
    normalized = host.rstrip(".").lower()
    return any(
        normalized == suffix or normalized.endswith("." + suffix)
        for suffix in ALLOWED_SUFFIXES
    )


class ConnectHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        self.request.settimeout(15)
        header = b""
        while b"\r\n\r\n" not in header and len(header) < MAX_HEADER_BYTES:
            chunk = self.request.recv(4096)
            if not chunk:
                return
            header += chunk
        try:
            first = header.split(b"\r\n", 1)[0].decode("ascii")
            method, authority, _ = first.split(" ", 2)
            host, separator, port_text = authority.rpartition(":")
            port = int(port_text) if separator else 443
        except (UnicodeDecodeError, ValueError):
            self.request.sendall(b"HTTP/1.1 400 Bad Request\r\n\r\n")
            return
        if method != "CONNECT" or port != 443 or not allowed_host(host):
            self.request.sendall(b"HTTP/1.1 403 Forbidden\r\n\r\n")
            return
        try:
            upstream = socket.create_connection((host, port), timeout=15)
        except OSError:
            self.request.sendall(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            return
        with upstream:
            self.request.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            sockets = (self.request, upstream)
            while True:
                readable, _, exceptional = select.select(sockets, (), sockets, 60)
                if exceptional or not readable:
                    return
                for source in readable:
                    try:
                        data = source.recv(65536)
                    except OSError:
                        return
                    if not data:
                        return
                    target = upstream if source is self.request else self.request
                    try:
                        target.sendall(data)
                    except OSError:
                        return


class Proxy(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    with Proxy(("0.0.0.0", 3128), ConnectHandler) as server:
        server.serve_forever()
