"""Block outbound networking inside the guest process except to the local key broker.

SIMULATION: this is an in-process guard (patched socket calls), not a kernel/hypervisor
firewall. On a real Secure Execution guest you would give the guest no network route except
to the broker (e.g. a dedicated virtio interface + nftables policy baked into the SE image).
"""
import socket

_ALLOWED = set()
_orig_connect = socket.socket.connect
_orig_getaddrinfo = socket.getaddrinfo


class EgressBlocked(PermissionError):
    pass


def _check(address):
    if isinstance(address, tuple) and len(address) >= 2:
        if (address[0], int(address[1])) in _ALLOWED:
            return
    raise EgressBlocked(f"guest egress blocked: {address!r}")


def _connect(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        _check(address)
    return _orig_connect(self, address)


def _getaddrinfo(host, port, *a, **k):
    if host not in {h for h, _ in _ALLOWED}:
        raise EgressBlocked(f"guest DNS/egress blocked: {host!r}")
    return _orig_getaddrinfo(host, port, *a, **k)


def install(allow):
    for h, p in allow:
        _ALLOWED.add((h, int(p)))
        try:   # a service name (e.g. in a container network) is also allowed by its resolved IPs
            for info in _orig_getaddrinfo(h, int(p), proto=socket.IPPROTO_TCP):
                _ALLOWED.add((info[4][0], int(p)))
        except OSError:
            pass
    socket.socket.connect = _connect
    socket.getaddrinfo = _getaddrinfo   # also covers socket.create_connection (same module globals)
