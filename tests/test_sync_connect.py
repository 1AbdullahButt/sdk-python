"""The synchronous backend shares one connect budget across a host's addresses.

A fake clock, resolver and connector stand in for the network, so these tests open no
sockets and never sleep: an address that never answers uses exactly the time it is
given, and a refused address fails at once.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any

import httpcore2
import pytest

from proofline import _sync_http

# Documentation addresses (RFC 3849, RFC 5737): nothing here reaches a network.
SILENT = ("2001:db8::1", "2001:db8::2", "2001:db8::3", "2001:db8::4")
REFUSED = "2001:db8::dead"
HEALTHY = ("192.0.2.1", "192.0.2.2")


class Connected(httpcore2.NetworkStream):
    def __init__(self, address: str) -> None:
        self.address = address

    def get_extra_info(self, info: str) -> Any:
        return self.address if info == "server_addr" else None


class Network:
    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, addresses: Iterable[str], lookup_seconds: float = 0
    ) -> None:
        self.now = 100.0
        self.attempts: list[tuple[str, float | None]] = []
        resolved = list(addresses)

        def resolve(host: str, port: int, timeout: float | None) -> list[str]:
            self.now += lookup_seconds
            return resolved

        def connect(
            backend: httpcore2.SyncBackend,
            host: str,
            port: int,
            timeout: float | None = None,
            local_address: str | None = None,
            socket_options: Any = None,
        ) -> httpcore2.NetworkStream:
            self.attempts.append((host, timeout))
            if host in SILENT:
                assert timeout is not None, "an unbounded attempt at a silent address never ends"
                self.now += timeout
                raise httpcore2.ConnectTimeout("timed out")
            if host == REFUSED:
                raise httpcore2.ConnectError("refused")
            return Connected(host)

        monkeypatch.setattr(_sync_http, "monotonic", lambda: self.now)
        monkeypatch.setattr(_sync_http, "resolve", resolve)
        monkeypatch.setattr(httpcore2.SyncBackend, "connect_tcp", connect)

    @contextmanager
    def deadline(self, seconds: float | None) -> Iterator[None]:
        token = _sync_http.IO_DEADLINE.set(None if seconds is None else self.now + seconds)
        try:
            yield
        finally:
            _sync_http.IO_DEADLINE.reset(token)

    def connect(self, timeout: float | None) -> httpcore2.NetworkStream:
        return _sync_http.NetworkBackend().connect_tcp("dual.proofline.invalid", 443, timeout)


def test_an_address_that_never_answers_leaves_time_for_the_next(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    network = Network(monkeypatch, [SILENT[0], HEALTHY[0]])
    with network.deadline(30):
        stream = network.connect(30)
    assert stream.get_extra_info("server_addr") == HEALTHY[0]
    assert network.attempts == [(SILENT[0], 15.0), (HEALTHY[0], 15.0)]


def test_two_silent_ipv6_addresses_still_reach_ipv4(monkeypatch: pytest.MonkeyPatch) -> None:
    # proofline.sh publishes two AAAA and two A records; a broken IPv6 route drops both.
    network = Network(monkeypatch, [SILENT[0], SILENT[1], HEALTHY[0], HEALTHY[1]])
    with network.deadline(30):
        stream = network.connect(30)
    assert stream.get_extra_info("server_addr") == HEALTHY[0]
    assert network.attempts == [(SILENT[0], 7.5), (SILENT[1], 7.5), (HEALTHY[0], 7.5)]
    assert network.now == 115.0


def test_addresses_that_never_answer_share_the_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    network = Network(monkeypatch, SILENT[:3])
    with network.deadline(30), pytest.raises(httpcore2.ConnectTimeout):
        network.connect(30)
    assert network.attempts == [(SILENT[0], 10.0), (SILENT[1], 10.0), (SILENT[2], 10.0)]
    assert network.now == 130.0


def test_without_a_request_deadline_the_connect_timeout_bounds_all_addresses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ProoflineClient always sets a request deadline; the bare transport does not, so
    # the shares must come from the time actually left, not from the full timeout.
    network = Network(monkeypatch, SILENT)
    with network.deadline(None), pytest.raises(httpcore2.ConnectTimeout):
        network.connect(30)
    shares = [timeout for _, timeout in network.attempts]
    assert shares == [7.5, 7.5, 7.5, 7.5]
    assert sum(share or 0 for share in shares) <= 30
    assert network.now - 100.0 <= 30


def test_time_between_clock_reads_does_not_move_the_request_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The clock moves on right after its first reading (here by 1 s, in practice by
    # microseconds). The attempt must still end at the request deadline, not after it.
    network = Network(monkeypatch, [SILENT[0]])
    readings = 0

    def monotonic() -> float:
        nonlocal readings
        readings += 1
        now = network.now
        if readings == 1:
            network.now += 1
        return now

    monkeypatch.setattr(_sync_http, "monotonic", monotonic)
    with network.deadline(30), pytest.raises(httpcore2.ConnectTimeout):
        network.connect(60)
    assert network.attempts == [(SILENT[0], 29.0)]
    assert network.now == 130.0


def test_the_lookup_and_the_attempts_share_one_connect_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    network = Network(monkeypatch, [SILENT[0], HEALTHY[0]], lookup_seconds=6)
    with network.deadline(None):
        stream = network.connect(30)
    assert stream.get_extra_info("server_addr") == HEALTHY[0]
    assert network.attempts == [(SILENT[0], 12.0), (HEALTHY[0], 12.0)]
    assert network.now - 100.0 <= 30


def test_a_lookup_that_uses_the_whole_timeout_leaves_no_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    network = Network(monkeypatch, [HEALTHY[0]], lookup_seconds=30)
    with network.deadline(None), pytest.raises(httpcore2.ConnectTimeout):
        network.connect(30)
    assert network.attempts == []


def test_a_refused_address_passes_its_time_to_the_next(monkeypatch: pytest.MonkeyPatch) -> None:
    network = Network(monkeypatch, [REFUSED, HEALTHY[0]])
    with network.deadline(30):
        stream = network.connect(30)
    assert stream.get_extra_info("server_addr") == HEALTHY[0]
    assert network.attempts == [(REFUSED, 15.0), (HEALTHY[0], 30.0)]


@pytest.mark.parametrize(
    ("timeout", "deadline", "expected"),
    [(30, 30, 30.0), (30, 12, 12.0), (5, 30, 5.0), (30, None, 30.0)],
)
def test_a_single_address_keeps_the_whole_budget(
    monkeypatch: pytest.MonkeyPatch, timeout: float, deadline: float | None, expected: float
) -> None:
    network = Network(monkeypatch, [HEALTHY[0]])
    with network.deadline(deadline):
        network.connect(timeout)
    assert network.attempts == [(HEALTHY[0], expected)]


def test_no_timeout_and_no_deadline_stay_unbounded(monkeypatch: pytest.MonkeyPatch) -> None:
    network = Network(monkeypatch, [REFUSED, HEALTHY[0]])
    with network.deadline(None):
        stream = network.connect(None)
    assert stream.get_extra_info("server_addr") == HEALTHY[0]
    assert network.attempts == [(REFUSED, None), (HEALTHY[0], None)]
