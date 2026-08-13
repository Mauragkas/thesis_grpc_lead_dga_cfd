//! SRP: maps an address string to a u64 ring identifier.
//! OCP: new hashing strategies implement `AddressHasher` without
//! touching the ring state or networking code.

use sha2::{Digest, Sha256};

/// ISP: a single capability — hashing an address to a ring ID.
pub trait AddressHasher: Send + Sync {
    fn hash(&self, address: &str) -> u64;
}

/// SHA-256 of the address, first 8 bytes as big-endian u64.
/// Uniformly distributes nodes across the 64-bit ring space.
#[derive(Debug, Clone, Copy, Default)]
pub struct Sha256Hasher;

impl AddressHasher for Sha256Hasher {
    fn hash(&self, address: &str) -> u64 {
        let mut hasher = Sha256::new();
        hasher.update(address.as_bytes());
        let digest = hasher.finalize();
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&digest[..8]);
        u64::from_be_bytes(bytes)
    }
}
