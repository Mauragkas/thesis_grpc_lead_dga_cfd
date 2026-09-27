use std::sync::atomic::{AtomicUsize, Ordering};

pub const DEFAULT_NUM_CURVES: usize = 3;

static NUM_CURVES: AtomicUsize = AtomicUsize::new(DEFAULT_NUM_CURVES);

/// Set the number of Hilbert multi-probe curves used for feature normalization.
pub fn set_num_curves(n: usize) {
    if n > 0 {
        NUM_CURVES.store(n, Ordering::Relaxed);
    }
}

/// Get the current number of curves.
pub fn get_num_curves() -> usize {
    NUM_CURVES.load(Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// Feature extraction
// ---------------------------------------------------------------------------
pub fn feature(key: &str) -> f64 {
    if let Some((hex_part, _payload)) = key.split_once('|') {
        if !hex_part.is_empty() && hex_part.chars().all(|c| c.is_ascii_hexdigit()) {
            let num_curves = get_num_curves();
            let first_char = hex_part.chars().next().unwrap();

            // Multi-probe Hilbert key check: starts with curve index c < num_curves
            if let Some(c) = first_char.to_digit(16).map(|v| v as usize) {
                if c < num_curves && hex_part.len() > 1 {
                    let hilbert_hex = &hex_part[1..];
                    let len = hilbert_hex.len().min(16);
                    let prefix = &hilbert_hex[..len];
                    if let Ok(val) = u64::from_str_radix(prefix, 16) {
                        let max_for_len = if len < 16 {
                            (1u64 << (4 * len)) - 1
                        } else {
                            u64::MAX
                        };
                        let h_frac = if max_for_len > 0 {
                            val as f64 / max_for_len as f64
                        } else {
                            0.0
                        };
                        let normalized = (c as f64 + h_frac) / (num_curves as f64);
                        return normalized.clamp(0.0, 1.0);
                    }
                }
            }

            // Fallback for hex-prefix keys that do not conform to multi-probe curve index
            let len = hex_part.len().min(16);
            let prefix = &hex_part[..len];
            if let Ok(val) = u64::from_str_radix(prefix, 16) {
                let max_for_len = if len < 16 {
                    (1u64 << (4 * len)) - 1
                } else {
                    u64::MAX
                };
                if max_for_len > 0 {
                    return (val as f64 / max_for_len as f64).clamp(0.0, 1.0);
                }
            }
        }
    } else if let Some(hex_part) = key.split('|').next() {
        // Handle plain hex string keys without '|' delimiter
        if !hex_part.is_empty() && hex_part.chars().all(|c| c.is_ascii_hexdigit()) {
            let len = hex_part.len().min(16);
            let prefix = &hex_part[..len];
            if let Ok(val) = u64::from_str_radix(prefix, 16) {
                let max_for_len = if len < 16 {
                    (1u64 << (4 * len)) - 1
                } else {
                    u64::MAX
                };
                if max_for_len > 0 {
                    return (val as f64 / max_for_len as f64).clamp(0.0, 1.0);
                }
            }
        }
    }

    // Fallback for non-Hilbert keys
    let h = crate::ring::peer_hash(key);
    ((h as f64) / (u64::MAX as f64)).clamp(0.0, 1.0)
}
