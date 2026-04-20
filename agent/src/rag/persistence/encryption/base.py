from abc import ABC, abstractmethod


class EncryptionService(ABC):
    """
    Abstract encryption service interface.
    Defines a contract for symmetric encryption and decryption.
    """

    @abstractmethod
    def encrypt(self, data: bytes) -> bytes:
        """
        Encrypt raw bytes into a secure payload.

        Parameters
        ----------
        data : bytes
            The plaintext data to be encrypted.

        Returns
        -------
        bytes
            The resulting ciphertext or encrypted payload.
        """
        pass

    @abstractmethod
    def decrypt(self, data: bytes) -> bytes:
        """
        Decrypt an encrypted payload back into raw bytes.

        Parameters
        ----------
        data : bytes
            The encrypted ciphertext to be decrypted.

        Returns
        -------
        bytes
            The recovered plaintext data.
        """
        pass
