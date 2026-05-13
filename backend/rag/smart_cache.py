import hashlib
import time
from threading import Lock


class SmartCache:

    def __init__(self, max_size=500, ttl=3600):

        self.cache = {}
        self.timestamps = {}

        self.max_size = max_size
        self.ttl = ttl

        # thread-safe (مهم في production API)
        self.lock = Lock()

    # =========================
    # normalize input (fast path)
    # =========================
    def _normalize(self, text: str):
        if not text:
            return ""
        return " ".join(text.strip().lower().split())

    # =========================
    # cache key (fast + stable)
    # =========================
    def _key(self, text: str):
        return hashlib.md5(self._normalize(text).encode()).hexdigest()

    # =========================
    # cleanup expired (lightweight)
    # =========================
    def _cleanup(self):

        now = time.time()

        expired_keys = [
            k for k, t in self.timestamps.items()
            if now - t > self.ttl
        ]

        for k in expired_keys:
            self.cache.pop(k, None)
            self.timestamps.pop(k, None)

    # =========================
    # GET
    # =========================
    def get(self, text: str):

        key = self._key(text)

        with self.lock:

            self._cleanup()

            value = self.cache.get(key)

            # refresh access time (LRU-like behavior)
            if value is not None:
                self.timestamps[key] = time.time()

            return value

    # =========================
    # SET (LRU + TTL safe)
    # =========================
    def set(self, text: str, value: str):

        key = self._key(text)

        with self.lock:

            self._cleanup()

            # 🔥 enforce memory limit
            if len(self.cache) >= self.max_size:

                # remove oldest entry (LRU-style)
                oldest_key = min(self.timestamps, key=self.timestamps.get)

                self.cache.pop(oldest_key, None)
                self.timestamps.pop(oldest_key, None)

            self.cache[key] = value
            self.timestamps[key] = time.time()