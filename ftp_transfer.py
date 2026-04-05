from __future__ import annotations

import hashlib
import logging
import socket
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from ftplib import FTP, all_errors
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import ThreadedFTPServer


ProgressCallback = Callable[[str, str, int, Optional[int], float], None]


@dataclass(slots=True)
class FTPConfig:
    """Single place for editable FTP settings."""

    host: str = "127.0.0.1"
    port: int = 2121
    username: str = "ftpuser"
    password: str = "ftp_password"
    root_dir: Path = Path("./ftp_root")

    passive_mode: bool = True

    timeout: float = 60.0
    connect_timeout: float = 20.0
    read_timeout: float = 120.0

    chunk_size: int = 4 * 1024 * 1024

    max_concurrent_workers: int = 4
    max_concurrent_client_downloads: int = 4

    retry_count: int = 3
    retry_delay_seconds: float = 3.0

    socket_send_buffer_size: int = 1024 * 1024
    socket_recv_buffer_size: int = 1024 * 1024

    temp_file_suffix: str = ".part"

    checksum_enabled: bool = False
    logging_enabled: bool = True
    log_level: int = logging.INFO

    server_max_cons: int = 64
    server_max_cons_per_ip: int = 16
    server_banner: str = "Internal FTP service ready"
    server_use_sendfile: bool = True


@dataclass(slots=True)
class TransferResult:
    success: bool
    operation: str
    local_path: Path
    remote_path: str
    host: str
    port: int
    bytes_transferred: int
    elapsed_seconds: float
    average_mbps: float
    attempts: int
    error_message: Optional[str] = None
    local_sha256: Optional[str] = None
    remote_sha256: Optional[str] = None


@dataclass(slots=True)
class RemoteFTP:
    host: str
    port: int = 21
    username: str = "anonymous"
    password: str = "anonymous@"


def _configure_logger(enabled: bool, level: int) -> logging.Logger:
    logger = logging.getLogger("ftp_transfer")
    if enabled:
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(threadName)s | %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(level)
    else:
        logger.handlers.clear()
        logger.setLevel(logging.CRITICAL + 1)
    logger.propagate = False
    return logger


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def compute_sha256(file_path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_temp_path(target_path: Path, suffix: str) -> Path:
    return target_path.with_name(f"{target_path.name}{suffix}")


def safe_replace(src: Path, dst: Path) -> None:
    ensure_directory(dst.parent)
    src.replace(dst)


def mbps(bytes_transferred: int, elapsed_seconds: float) -> float:
    if elapsed_seconds <= 0:
        return 0.0
    return (bytes_transferred / (1024 * 1024)) / elapsed_seconds


@contextmanager
def ftp_connection(config: FTPConfig, endpoint: Optional[RemoteFTP] = None) -> Iterable[FTP]:
    endpoint = endpoint or RemoteFTP(
        host=config.host,
        port=config.port,
        username=config.username,
        password=config.password,
    )

    ftp = FTP(timeout=config.timeout)
    ftp.connect(host=endpoint.host, port=endpoint.port, timeout=config.connect_timeout)
    ftp.login(user=endpoint.username, passwd=endpoint.password)
    ftp.set_pasv(config.passive_mode)

    if ftp.sock is not None:
        ftp.sock.settimeout(config.read_timeout)
        ftp.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, config.socket_send_buffer_size)
        ftp.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, config.socket_recv_buffer_size)

    try:
        yield ftp
    finally:
        try:
            ftp.quit()
        except all_errors:
            try:
                ftp.close()
            except all_errors:
                pass


class FTPServerManager:
    """Manages a local pyftpdlib FTP server lifecycle."""

    def __init__(self, config: FTPConfig):
        self.config = config
        self.logger = _configure_logger(config.logging_enabled, config.log_level)
        self._server: Optional[ThreadedFTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start_server(self) -> None:
        with self._lock:
            if self.is_running():
                return

            ensure_directory(self.config.root_dir)

            authorizer = DummyAuthorizer()
            authorizer.add_user(
                self.config.username,
                self.config.password,
                str(self.config.root_dir),
                perm="elradfmwMT",
            )

            handler = FTPHandler
            handler.authorizer = authorizer
            handler.banner = self.config.server_banner
            handler.passive_ports = None
            handler.use_sendfile = self.config.server_use_sendfile

            server = ThreadedFTPServer((self.config.host, self.config.port), handler)
            server.max_cons = self.config.server_max_cons
            server.max_cons_per_ip = self.config.server_max_cons_per_ip

            self._server = server
            self._thread = threading.Thread(
                target=server.serve_forever,
                kwargs={"timeout": 0.5, "blocking": True, "handle_exit": False},
                name="FTPServerThread",
                daemon=True,
            )
            self._thread.start()
            self.logger.info("FTP server started on %s:%s", self.config.host, self.config.port)

    def stop_server(self) -> None:
        with self._lock:
            if not self._server:
                return
            try:
                self._server.close_all()
            finally:
                if self._thread:
                    self._thread.join(timeout=5)
                self.logger.info("FTP server stopped")
                self._server = None
                self._thread = None


class FTPClientManager:
    """Handles uploads/downloads with retries and concurrent transfer helpers."""

    def __init__(self, config: FTPConfig):
        self.config = config
        self.logger = _configure_logger(config.logging_enabled, config.log_level)

    def remote_exists(self, remote_path: str, endpoint: Optional[RemoteFTP] = None) -> bool:
        try:
            with ftp_connection(self.config, endpoint) as ftp:
                ftp.size(remote_path)
            return True
        except all_errors:
            return False

    def remote_size(self, remote_path: str, endpoint: Optional[RemoteFTP] = None) -> Optional[int]:
        try:
            with ftp_connection(self.config, endpoint) as ftp:
                return ftp.size(remote_path)
        except all_errors:
            return None

    def list_remote_directory(self, remote_dir: str = ".", endpoint: Optional[RemoteFTP] = None) -> list[str]:
        with ftp_connection(self.config, endpoint) as ftp:
            return ftp.nlst(remote_dir)

    def upload_file(
        self,
        local_path: Path,
        remote_path: str,
        endpoint: Optional[RemoteFTP] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> TransferResult:
        local_path = local_path.resolve()
        return self._transfer_with_retry(
            operation="upload",
            local_path=local_path,
            remote_path=remote_path,
            endpoint=endpoint,
            progress_callback=progress_callback,
        )

    def download_file(
        self,
        remote_path: str,
        local_path: Path,
        endpoint: Optional[RemoteFTP] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> TransferResult:
        local_path = local_path.resolve()
        return self._transfer_with_retry(
            operation="download",
            local_path=local_path,
            remote_path=remote_path,
            endpoint=endpoint,
            progress_callback=progress_callback,
        )

    def upload_files_concurrently(
        self,
        items: Sequence[tuple[Path, str, Optional[RemoteFTP]]],
        progress_callback: Optional[ProgressCallback] = None,
        max_workers: Optional[int] = None,
    ) -> list[TransferResult]:
        workers = max_workers or self.config.max_concurrent_workers
        results: list[TransferResult] = []

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ftp-upload") as pool:
            futures = {
                pool.submit(self.upload_file, local, remote, endpoint, progress_callback): (local, remote)
                for local, remote, endpoint in items
            }
            for future in as_completed(futures):
                results.append(future.result())

        return results

    def download_files_concurrently(
        self,
        items: Sequence[tuple[str, Path, Optional[RemoteFTP]]],
        progress_callback: Optional[ProgressCallback] = None,
        max_workers: Optional[int] = None,
    ) -> list[TransferResult]:
        workers = max_workers or self.config.max_concurrent_client_downloads
        results: list[TransferResult] = []

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ftp-download") as pool:
            futures = {
                pool.submit(self.download_file, remote, local, endpoint, progress_callback): (remote, local)
                for remote, local, endpoint in items
            }
            for future in as_completed(futures):
                results.append(future.result())

        return results

    def benchmark_download(
        self,
        remote_path: str,
        local_path: Path,
        endpoint: Optional[RemoteFTP] = None,
        test_chunk_sizes: Sequence[int] = (1024 * 1024, 4 * 1024 * 1024, 8 * 1024 * 1024),
    ) -> list[TransferResult]:
        baseline = self.config.chunk_size
        results: list[TransferResult] = []

        for chunk in test_chunk_sizes:
            self.config.chunk_size = chunk
            result = self.download_file(remote_path, local_path.with_suffix(f".chunk{chunk}"), endpoint)
            results.append(result)

        self.config.chunk_size = baseline
        return results

    def _transfer_with_retry(
        self,
        operation: str,
        local_path: Path,
        remote_path: str,
        endpoint: Optional[RemoteFTP],
        progress_callback: Optional[ProgressCallback],
    ) -> TransferResult:
        attempts = 0
        last_error: Optional[str] = None

        while attempts < self.config.retry_count:
            attempts += 1
            try:
                if operation == "upload":
                    return self._upload_once(local_path, remote_path, endpoint, progress_callback, attempts)
                return self._download_once(local_path, remote_path, endpoint, progress_callback, attempts)
            except (OSError, all_errors) as exc:
                last_error = str(exc)
                self.logger.warning(
                    "%s failed on attempt %s/%s for %s: %s",
                    operation,
                    attempts,
                    self.config.retry_count,
                    remote_path,
                    last_error,
                )
                if attempts < self.config.retry_count:
                    time.sleep(self.config.retry_delay_seconds)

        endpoint = endpoint or RemoteFTP(self.config.host, self.config.port, self.config.username, self.config.password)
        return TransferResult(
            success=False,
            operation=operation,
            local_path=local_path,
            remote_path=remote_path,
            host=endpoint.host,
            port=endpoint.port,
            bytes_transferred=0,
            elapsed_seconds=0,
            average_mbps=0,
            attempts=attempts,
            error_message=last_error,
        )

    def _upload_once(
        self,
        local_path: Path,
        remote_path: str,
        endpoint: Optional[RemoteFTP],
        progress_callback: Optional[ProgressCallback],
        attempts: int,
    ) -> TransferResult:
        if not local_path.exists():
            raise FileNotFoundError(f"Local file does not exist: {local_path}")

        endpoint = endpoint or RemoteFTP(self.config.host, self.config.port, self.config.username, self.config.password)
        total_size = local_path.stat().st_size
        transferred = 0
        started = time.perf_counter()

        with ftp_connection(self.config, endpoint) as ftp, local_path.open("rb") as stream:
            data_sock = ftp.transfercmd(f"STOR {remote_path}")
            data_sock.settimeout(self.config.read_timeout)
            data_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self.config.socket_send_buffer_size)
            data_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.config.socket_recv_buffer_size)

            try:
                while chunk := stream.read(self.config.chunk_size):
                    data_sock.sendall(chunk)
                    transferred += len(chunk)
                    if progress_callback:
                        elapsed = time.perf_counter() - started
                        progress_callback("upload", remote_path, transferred, total_size, mbps(transferred, elapsed))
            finally:
                data_sock.close()
                ftp.voidresp()

        elapsed = time.perf_counter() - started
        local_sha = compute_sha256(local_path, self.config.chunk_size) if self.config.checksum_enabled else None

        return TransferResult(
            success=True,
            operation="upload",
            local_path=local_path,
            remote_path=remote_path,
            host=endpoint.host,
            port=endpoint.port,
            bytes_transferred=transferred,
            elapsed_seconds=elapsed,
            average_mbps=mbps(transferred, elapsed),
            attempts=attempts,
            local_sha256=local_sha,
        )

    def _download_once(
        self,
        local_path: Path,
        remote_path: str,
        endpoint: Optional[RemoteFTP],
        progress_callback: Optional[ProgressCallback],
        attempts: int,
    ) -> TransferResult:
        endpoint = endpoint or RemoteFTP(self.config.host, self.config.port, self.config.username, self.config.password)
        ensure_directory(local_path.parent)

        tmp_path = build_temp_path(local_path, self.config.temp_file_suffix)
        if tmp_path.exists():
            tmp_path.unlink()

        transferred = 0
        started = time.perf_counter()

        with ftp_connection(self.config, endpoint) as ftp:
            remote_size = ftp.size(remote_path)
            data_sock = ftp.transfercmd(f"RETR {remote_path}")
            data_sock.settimeout(self.config.read_timeout)
            data_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self.config.socket_send_buffer_size)
            data_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.config.socket_recv_buffer_size)

            try:
                with tmp_path.open("wb") as stream:
                    while chunk := data_sock.recv(self.config.chunk_size):
                        stream.write(chunk)
                        transferred += len(chunk)
                        if progress_callback:
                            elapsed = time.perf_counter() - started
                            progress_callback("download", remote_path, transferred, remote_size, mbps(transferred, elapsed))
            except Exception:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
                raise
            finally:
                data_sock.close()
                ftp.voidresp()

        safe_replace(tmp_path, local_path)

        elapsed = time.perf_counter() - started
        local_sha = compute_sha256(local_path, self.config.chunk_size) if self.config.checksum_enabled else None

        return TransferResult(
            success=True,
            operation="download",
            local_path=local_path,
            remote_path=remote_path,
            host=endpoint.host,
            port=endpoint.port,
            bytes_transferred=transferred,
            elapsed_seconds=elapsed,
            average_mbps=mbps(transferred, elapsed),
            attempts=attempts,
            local_sha256=local_sha,
        )


def default_progress_callback(
    operation: str,
    remote_path: str,
    bytes_done: int,
    total_bytes: Optional[int],
    speed_mbps: float,
) -> None:
    if total_bytes and total_bytes > 0:
        pct = (bytes_done / total_bytes) * 100
        logging.getLogger("ftp_transfer").info(
            "%s %s: %.2f%% (%d/%d bytes) speed=%.2f MB/s",
            operation,
            remote_path,
            pct,
            bytes_done,
            total_bytes,
            speed_mbps,
        )
    else:
        logging.getLogger("ftp_transfer").info(
            "%s %s: %d bytes speed=%.2f MB/s",
            operation,
            remote_path,
            bytes_done,
            speed_mbps,
        )


def example_simultaneous_server_client() -> None:
    """Example: start local server and download from another FTP server in parallel."""
    config = FTPConfig(root_dir=Path("C:/ftp_root"), chunk_size=4 * 1024 * 1024)

    server = FTPServerManager(config)
    client = FTPClientManager(config)

    server.start_server()
    try:
        remote_d = RemoteFTP(host="192.168.1.40", port=21, username="user_d", password="secret")
        result = client.download_file(
            remote_path="/exports/large.iso",
            local_path=Path("C:/ftp_root/incoming/large.iso"),
            endpoint=remote_d,
            progress_callback=default_progress_callback,
        )
        print(result)
    finally:
        server.stop_server()


if __name__ == "__main__":
    example_simultaneous_server_client()
