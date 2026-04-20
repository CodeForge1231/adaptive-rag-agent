from cryptography.fernet import Fernet

from .base import EncryptionService


class FernetEncryptionService(EncryptionService):
    """
    Symmetric encryption service implementation using the Fernet specification.

    Attributes
    ----------
    _cipher : cryptography.fernet.Fernet
        The internal Fernet instance used for cryptographic operations.
    """

    def __init__(self, key: str):
        # Fernet expects a base64-encoded key
        self._cipher = Fernet(key.encode())

    def encrypt(self, data: bytes) -> bytes:
        """
        Encrypt data using Fernet symmetric encryption.

        Parameters
        ----------
        data : bytes
            The raw plaintext bytes to be protected.

        Returns
        -------
        bytes
            The secure, authenticated ciphertext.
        """
        return self._cipher.encrypt(data)

    def decrypt(self, data: bytes) -> bytes:
        """
        Decrypt data using Fernet symmetric encryption.

        Parameters
        ----------
        data : bytes
            The Fernet-encrypted ciphertext.

        Returns
        -------
        bytes
            The original plaintext bytes.
        """
        return self._cipher.decrypt(data)
