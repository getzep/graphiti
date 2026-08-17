from __future__ import annotations

from dataclasses import dataclass
from secrets import token_bytes

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

AES_256_KEY_BYTES = 32
AES_GCM_NONCE_BYTES = 12


class DecryptionError(ValueError):
    """Raised when encrypted connection material cannot be authenticated."""


@dataclass(frozen=True, slots=True)
class EncryptedValue:
    nonce: bytes
    ciphertext: bytes


class AesGcmCipher:
    """Small AES-256-GCM envelope used for OAuth credentials and PKCE verifiers."""

    def __init__(self, key: bytes):
        if not isinstance(key, bytes) or len(key) != AES_256_KEY_BYTES:
            raise ValueError('AES-256-GCM key must contain exactly 32 bytes')
        self._cipher = AESGCM(key)

    def encrypt_text(self, value: str, *, associated_data: bytes) -> EncryptedValue:
        if not isinstance(value, str) or not value:
            raise ValueError('encrypted text must be a non-empty string')
        if not isinstance(associated_data, bytes) or not associated_data:
            raise ValueError('associated_data must be non-empty bytes')
        nonce = token_bytes(AES_GCM_NONCE_BYTES)
        ciphertext = self._cipher.encrypt(nonce, value.encode('utf-8'), associated_data)
        return EncryptedValue(nonce=nonce, ciphertext=ciphertext)

    def decrypt_text(
        self,
        nonce: bytes,
        ciphertext: bytes,
        *,
        associated_data: bytes,
    ) -> str:
        if len(nonce) != AES_GCM_NONCE_BYTES:
            raise DecryptionError('encrypted value has an invalid nonce')
        try:
            plaintext = self._cipher.decrypt(nonce, ciphertext, associated_data)
            return plaintext.decode('utf-8')
        except (InvalidTag, UnicodeDecodeError) as exc:
            raise DecryptionError('encrypted value authentication failed') from exc
