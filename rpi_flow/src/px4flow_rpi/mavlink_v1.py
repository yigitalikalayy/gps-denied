from __future__ import annotations

import struct
import time
from dataclasses import dataclass


def _x25_crc(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        tmp = b ^ (crc & 0xFF)
        tmp = (tmp ^ (tmp << 4)) & 0xFF
        crc = ((crc >> 8) ^ (tmp << 8) ^ (tmp << 3) ^ (tmp >> 4)) & 0xFFFF
    return crc


def mavlink1_pack(msgid: int, payload: bytes, sysid: int, compid: int, seq: int, crc_extra: int) -> bytes:
    if not (0 <= msgid <= 255):
        raise ValueError("msgid out of range")
    if len(payload) > 255:
        raise ValueError("payload too long")

    stx = 0xFE
    header = struct.pack("<BBBBB", len(payload), seq & 0xFF, sysid & 0xFF, compid & 0xFF, msgid & 0xFF)
    crc_input = header + payload + bytes([crc_extra & 0xFF])
    crc = _x25_crc(crc_input)
    return bytes([stx]) + header + payload + struct.pack("<H", crc)


@dataclass
class MavlinkFrame:
    msgid: int
    sysid: int
    compid: int
    payload: bytes
    seq: int
    timestamp: float


class Mavlink1Parser:
    def __init__(self) -> None:
        self._state = 0
        self._len = 0
        self._buf = bytearray()
        self._needed = 0

    def feed(self, data: bytes) -> list[MavlinkFrame]:
        out: list[MavlinkFrame] = []
        for b in data:
            if self._state == 0:
                if b == 0xFE:
                    self._buf = bytearray([b])
                    self._state = 1
            elif self._state == 1:
                self._len = b
                self._buf.append(b)
                self._needed = 5 + self._len + 2  # hdr (seq..msgid) + payload + checksum
                self._state = 2
            elif self._state == 2:
                self._buf.append(b)
                if len(self._buf) == 1 + 1 + self._needed:
                    frame = self._try_decode(bytes(self._buf))
                    if frame is not None:
                        out.append(frame)
                    self._state = 0
        return out

    @staticmethod
    def _try_decode(raw: bytes) -> MavlinkFrame | None:
        if len(raw) < 8 or raw[0] != 0xFE:
            return None
        payload_len = raw[1]
        if len(raw) != 1 + 1 + 5 + payload_len + 2:
            return None
        length, seq, sysid, compid, msgid = struct.unpack_from("<BBBBB", raw, 1)
        _ = length
        payload = raw[1 + 1 + 5 : 1 + 1 + 5 + payload_len]
        return MavlinkFrame(
            msgid=msgid,
            sysid=sysid,
            compid=compid,
            payload=payload,
            seq=seq,
            timestamp=time.time(),
        )

