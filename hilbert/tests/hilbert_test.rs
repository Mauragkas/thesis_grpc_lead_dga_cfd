//! Unit tests for hilbert crate.

use hilbert_rs::{GeneKeyGenerator, HilbertKeyGenerator, NUM_CURVES};

fn leading_u64(key: &str) -> u64 {
    let hex = key.split('|').next().unwrap();
    let prefix = &hex[..hex.len().min(16)];
    u64::from_str_radix(prefix, 16).unwrap()
}

fn hex_dist(a: &str, b: &str) -> u128 {
    (leading_u64(a) as i128 - leading_u64(b) as i128).unsigned_abs()
}

fn euclid(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[test]
fn keys_are_deterministic() {
    let kg = HilbertKeyGenerator::new(10);
    let genes = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.25];
    assert_eq!(kg.keys_for(&genes), kg.keys_for(&genes));
}

#[test]
fn produces_one_key_per_curve_with_distinct_curve_ids() {
    let kg = HilbertKeyGenerator::new(10);
    let genes = [0.5; 10];
    let keys = kg.keys_for(&genes);
    assert_eq!(keys.len(), NUM_CURVES);
    let mut curves: Vec<char> = keys.iter().map(|k| k.chars().next().unwrap()).collect();
    curves.sort();
    curves.dedup();
    assert_eq!(curves.len(), NUM_CURVES);
}

#[test]
fn key_roundtrips_through_parse() {
    let kg = HilbertKeyGenerator::new(10);
    let genes = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.25];
    for key in kg.keys_for(&genes) {
        assert_eq!(kg.parse_key(&key), Some(genes.clone()));
    }
}

#[test]
fn multi_probe_recovers_near_neighbor() {
    let kg = HilbertKeyGenerator::new(10);
    let a = vec![0.5; 10];
    let b = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.51];
    let far = vec![0.0; 10];

    // Sanity: b is genuinely closer to a than far is.
    assert!(euclid(&a, &b) < euclid(&a, &far));

    let keys_a = kg.keys_for(&a);
    let keys_b = kg.keys_for(&b);
    let keys_far = kg.keys_for(&far);

    // Multi-probe guarantee: the closest curve for the near neighbor is
    // closer than the closest curve for the far point.
    let min_ab = (0..NUM_CURVES)
        .map(|i| hex_dist(&keys_a[i], &keys_b[i]))
        .min()
        .unwrap();
    let min_af = (0..NUM_CURVES)
        .map(|i| hex_dist(&keys_a[i], &keys_far[i]))
        .min()
        .unwrap();
    assert!(
        min_ab < min_af,
        "multi-probe failed to recover near neighbor"
    );
}
