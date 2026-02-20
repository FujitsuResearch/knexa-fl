#!/usr/bin/env python3
"""
Cryptographic utilities for KNEXA-FL P2P secure communication
Implements ECDH key exchange, AES-GCM encryption, and digital signatures
"""

import os
import base64
import hashlib
import time
from typing import Tuple, Optional, Dict, Any
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature
import logging

logger = logging.getLogger(__name__)

class KnexaCrypto:
    """
    Cryptographic handler for KNEXA-FL P2P secure communication
    Provides ECDH key exchange, AES-GCM encryption, and digital signatures
    """
    
    def __init__(self, peer_id: str):
        self.peer_id = peer_id
        self.backend = default_backend()
        
        # Generate ECDH key pair (P-256 curve for security and performance)
        self.private_key = ec.generate_private_key(ec.SECP256R1(), self.backend)
        self.public_key = self.private_key.public_key()
        
        # Session keys cache (peer_id -> session_data)
        self.session_keys: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Initialized KnexaCrypto for peer {peer_id}")
    
    def get_public_key_bytes(self) -> bytes:
        """Get public key in compressed format for transmission"""
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.X962,
            format=serialization.PublicFormat.UncompressedPoint
        )
    
    def get_public_key_b64(self) -> str:
        """Get base64-encoded public key for gRPC transmission"""
        return base64.b64encode(self.get_public_key_bytes()).decode('utf-8')
    
    def establish_shared_secret(self, peer_public_key_b64: str, peer_id: str) -> bool:
        """
        Establish shared secret using ECDH key exchange
        
        Args:
            peer_public_key_b64: Base64-encoded peer public key
            peer_id: Peer identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Decode peer public key
            peer_public_key_bytes = base64.b64decode(peer_public_key_b64)
            peer_public_key = ec.EllipticCurvePublicKey.from_encoded_point(
                ec.SECP256R1(), peer_public_key_bytes
            )
            
            # Perform ECDH key exchange
            shared_key = self.private_key.exchange(ec.ECDH(), peer_public_key)
            
            # Derive session keys using HKDF
            hkdf = HKDF(
                algorithm=hashes.SHA256(),
                length=32,  # 256-bit key for AES-GCM
                salt=None,
                info=f"KNEXA-FL-{self.peer_id}-{peer_id}".encode(),
                backend=self.backend
            )
            
            session_key = hkdf.derive(shared_key)
            
            # Store session data
            self.session_keys[peer_id] = {
                'key': session_key,
                'peer_public_key': peer_public_key,
                'established_at': time.time(),
                'expires_at': time.time() + 3600  # 1 hour expiry for forward secrecy
            }
            
            logger.info(f"Established shared secret with peer {peer_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to establish shared secret with {peer_id}: {e}")
            return False
    
    def encrypt_message(self, message: bytes, peer_id: str) -> Tuple[bytes, bytes]:
        """
        Encrypt message using AES-GCM with established session key
        
        Args:
            message: Message to encrypt
            peer_id: Target peer identifier
            
        Returns:
            Tuple of (encrypted_message, nonce)
        """
        if peer_id not in self.session_keys:
            raise ValueError(f"No session key established with peer {peer_id}")
        
        session_data = self.session_keys[peer_id]
        
        # Check if session is expired
        if time.time() > session_data['expires_at']:
            raise ValueError(f"Session with peer {peer_id} has expired")
        
        # Generate random nonce
        nonce = os.urandom(12)  # 96-bit nonce for AES-GCM
        
        # Encrypt message
        aesgcm = AESGCM(session_data['key'])
        encrypted_message = aesgcm.encrypt(nonce, message, None)
        
        return encrypted_message, nonce
    
    def decrypt_message(self, encrypted_message: bytes, nonce: bytes, peer_id: str) -> bytes:
        """
        Decrypt message using AES-GCM with established session key
        
        Args:
            encrypted_message: Encrypted message
            nonce: Nonce used for encryption
            peer_id: Source peer identifier
            
        Returns:
            Decrypted message
        """
        if peer_id not in self.session_keys:
            raise ValueError(f"No session key established with peer {peer_id}")
        
        session_data = self.session_keys[peer_id]
        
        # Check if session is expired
        if time.time() > session_data['expires_at']:
            raise ValueError(f"Session with peer {peer_id} has expired")
        
        # Decrypt message
        aesgcm = AESGCM(session_data['key'])
        decrypted_message = aesgcm.decrypt(nonce, encrypted_message, None)
        
        return decrypted_message
    
    def sign_message(self, message: bytes) -> bytes:
        """
        Sign message using ECDSA for authentication
        
        Args:
            message: Message to sign
            
        Returns:
            Digital signature
        """
        signature = self.private_key.sign(
            message,
            ec.ECDSA(hashes.SHA256())
        )
        return signature
    
    def verify_signature(self, message: bytes, signature: bytes, peer_id: str) -> bool:
        """
        Verify digital signature using peer's public key
        
        Args:
            message: Original message
            signature: Digital signature
            peer_id: Peer identifier
            
        Returns:
            True if signature is valid, False otherwise
        """
        if peer_id not in self.session_keys:
            return False
        
        try:
            peer_public_key = self.session_keys[peer_id]['peer_public_key']
            peer_public_key.verify(
                signature,
                message,
                ec.ECDSA(hashes.SHA256())
            )
            return True
        except InvalidSignature:
            return False
        except Exception as e:
            logger.error(f"Signature verification failed for peer {peer_id}: {e}")
            return False
    
    def create_secure_nonce(self) -> str:
        """Generate cryptographically secure nonce"""
        return base64.b64encode(os.urandom(32)).decode('utf-8')
    
    def rotate_session_key(self, peer_id: str) -> bool:
        """
        Rotate session key for forward secrecy
        
        Args:
            peer_id: Peer identifier
            
        Returns:
            True if successful, False otherwise
        """
        if peer_id not in self.session_keys:
            return False
        
        try:
            # Generate new ephemeral key pair
            new_private_key = ec.generate_private_key(ec.SECP256R1(), self.backend)
            peer_public_key = self.session_keys[peer_id]['peer_public_key']
            
            # Derive new shared secret
            shared_key = new_private_key.exchange(ec.ECDH(), peer_public_key)
            
            # Derive new session key
            hkdf = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=None,
                info=f"KNEXA-FL-ROTATE-{self.peer_id}-{peer_id}-{int(time.time())}".encode(),
                backend=self.backend
            )
            
            new_session_key = hkdf.derive(shared_key)
            
            # Update session data
            self.session_keys[peer_id]['key'] = new_session_key
            self.session_keys[peer_id]['established_at'] = time.time()
            self.session_keys[peer_id]['expires_at'] = time.time() + 3600
            
            logger.info(f"Rotated session key for peer {peer_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rotate session key for peer {peer_id}: {e}")
            return False
    
    def cleanup_expired_sessions(self):
        """Remove expired session keys"""
        current_time = time.time()
        expired_peers = [
            peer_id for peer_id, session_data in self.session_keys.items()
            if current_time > session_data['expires_at']
        ]
        
        for peer_id in expired_peers:
            del self.session_keys[peer_id]
            logger.info(f"Cleaned up expired session for peer {peer_id}")
    
    def get_session_info(self, peer_id: str) -> Optional[Dict[str, Any]]:
        """Get session information for a peer"""
        if peer_id not in self.session_keys:
            return None
        
        session_data = self.session_keys[peer_id]
        return {
            'peer_id': peer_id,
            'established_at': session_data['established_at'],
            'expires_at': session_data['expires_at'],
            'is_expired': time.time() > session_data['expires_at'],
            'time_remaining': max(0, session_data['expires_at'] - time.time())
        }
    
    def get_all_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all active sessions"""
        return {
            peer_id: self.get_session_info(peer_id)
            for peer_id in self.session_keys.keys()
        }


class PrivacyUtils:
    """
    Privacy utilities for differential privacy and SIER monitoring
    """
    
    @staticmethod
    def add_dp_noise(data: bytes, epsilon: float = 1.0, delta: float = 1e-5) -> bytes:
        """
        Add differential privacy noise to data
        
        Args:
            data: Original data
            epsilon: Privacy parameter
            delta: Privacy parameter
            
        Returns:
            Noisy data
        """
        # Convert bytes to numpy array for noise addition
        import numpy as np
        
        # Simple noise addition (in practice, use proper DP mechanisms)
        noise_scale = 1.0 / epsilon
        data_array = np.frombuffer(data, dtype=np.uint8)
        noise = np.random.laplace(0, noise_scale, data_array.shape)
        
        # Add noise and clip to valid byte range
        noisy_data = np.clip(data_array + noise, 0, 255).astype(np.uint8)
        
        return noisy_data.tobytes()
    
    @staticmethod
    def compute_sier_score(data: bytes, patterns: list) -> float:
        """
        Compute Sensitive Information Exposure Rate (SIER)
        
        Args:
            data: Data to analyze
            patterns: List of sensitive patterns
            
        Returns:
            SIER score (0.0 to 1.0)
        """
        import re
        
        text = data.decode('utf-8', errors='ignore')
        total_tokens = len(text.split())
        
        if total_tokens == 0:
            return 0.0
        
        sensitive_tokens = 0
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            sensitive_tokens += len(matches)
        
        return min(1.0, sensitive_tokens / total_tokens)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create two peers
    peer1 = KnexaCrypto("peer1")
    peer2 = KnexaCrypto("peer2")
    
    # Establish secure channel
    peer1_pubkey = peer1.get_public_key_b64()
    peer2_pubkey = peer2.get_public_key_b64()
    
    print(f"Peer1 public key: {peer1_pubkey}")
    print(f"Peer2 public key: {peer2_pubkey}")
    
    # Establish shared secrets
    peer1.establish_shared_secret(peer2_pubkey, "peer2")
    peer2.establish_shared_secret(peer1_pubkey, "peer1")
    
    # Test encryption/decryption
    message = b"Hello, KNEXA-FL P2P!"
    encrypted, nonce = peer1.encrypt_message(message, "peer2")
    decrypted = peer2.decrypt_message(encrypted, nonce, "peer1")
    
    print(f"Original: {message}")
    print(f"Encrypted: {encrypted}")
    print(f"Decrypted: {decrypted}")
    print(f"Match: {message == decrypted}")
    
    # Test signatures
    signature = peer1.sign_message(message)
    valid = peer2.verify_signature(message, signature, "peer1")
    print(f"Signature valid: {valid}")