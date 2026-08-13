// ---------------------------------------------------------------------------
// Feature extraction
// ---------------------------------------------------------------------------
pub fn feature(key: &str) -> f64 {
    if let Some(hex_part) = key.split('|').next() {
        if !hex_part.is_empty() && hex_part.chars().all(|c| c.is_ascii_hexdigit()) {
            // Normalize to 64-bit regardless of actual hex length.
            // Treat the hex string as a fixed-point fraction: value / 16^len,
            // then scale to u64 range. This makes short prefixes fill the
            // entire [0,1] range proportionally.
            let len = hex_part.len().min(16);
            let prefix = &hex_part[..len];
            if let Ok(val) = u64::from_str_radix(prefix, 16) {
                // Maximum possible value for this length: 16^len - 1
                let max_for_len = if len < 16 {
                    (1u64 << (4 * len)) - 1
                } else {
                    u64::MAX
                };
                if max_for_len > 0 {
                    return val as f64 / max_for_len as f64;
                }
            }
        }
    }
    // Fallback for non-Hilbert keys
    let h = crate::ring::peer_hash(key);
    (h as f64) / (u64::MAX as f64)
}
