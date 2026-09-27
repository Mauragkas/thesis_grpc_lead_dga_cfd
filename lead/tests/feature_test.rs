use lead_node::rmi::feature;

#[test]
fn test_fallback_for_non_hex_keys() {
    let f1 = feature("plain_string_key_1");
    let f2 = feature("plain_string_key_2");
    assert!((0.0..=1.0).contains(&f1));
    assert!((0.0..=1.0).contains(&f2));
    assert_ne!(f1, f2);
}

#[test]
fn test_multi_probe_hilbert_keys_span_full_range() {
    // Curve 0: min and max Hilbert index
    let c0_min = "0000000000000000000000000000000000000000|{\"data\":1}";
    let c0_max = "0fffffffffffffffffffffffffffffffffffffff|{\"data\":1}";
    let f_c0_min = feature(c0_min);
    let f_c0_max = feature(c0_max);

    // Curve 0 should span [0.0, 1/3]
    assert!((f_c0_min - 0.0).abs() < 1e-4, "c0_min should be near 0.0, got {f_c0_min}");
    assert!((f_c0_max - 1.0 / 3.0).abs() < 1e-4, "c0_max should be near 0.3333, got {f_c0_max}");

    // Curve 1: min and max Hilbert index
    let c1_min = "1000000000000000000000000000000000000000|{\"data\":1}";
    let c1_max = "1fffffffffffffffffffffffffffffffffffffff|{\"data\":1}";
    let f_c1_min = feature(c1_min);
    let f_c1_max = feature(c1_max);

    // Curve 1 should span [1/3, 2/3]
    assert!((f_c1_min - 1.0 / 3.0).abs() < 1e-4, "c1_min should be near 0.3333, got {f_c1_min}");
    assert!((f_c1_max - 2.0 / 3.0).abs() < 1e-4, "c1_max should be near 0.6666, got {f_c1_max}");

    // Curve 2: min and max Hilbert index
    let c2_min = "2000000000000000000000000000000000000000|{\"data\":1}";
    let c2_max = "2fffffffffffffffffffffffffffffffffffffff|{\"data\":1}";
    let f_c2_min = feature(c2_min);
    let f_c2_max = feature(c2_max);

    // Curve 2 should span [2/3, 1.0]
    assert!((f_c2_min - 2.0 / 3.0).abs() < 1e-4, "c2_min should be near 0.6666, got {f_c2_min}");
    assert!((f_c2_max - 1.0).abs() < 1e-4, "c2_max should be near 1.0, got {f_c2_max}");
}

#[test]
fn test_multi_probe_monotonicity() {
    let keys = vec![
        "0100000000000000000000000000000000000000|{}",
        "0800000000000000000000000000000000000000|{}",
        "1200000000000000000000000000000000000000|{}",
        "1900000000000000000000000000000000000000|{}",
        "2400000000000000000000000000000000000000|{}",
        "2e00000000000000000000000000000000000000|{}",
    ];

    let features: Vec<f64> = keys.iter().map(|k| feature(k)).collect();
    for window in features.windows(2) {
        assert!(
            window[0] < window[1],
            "Monotonicity violated: {} should be < {}",
            window[0],
            window[1]
        );
    }
}
