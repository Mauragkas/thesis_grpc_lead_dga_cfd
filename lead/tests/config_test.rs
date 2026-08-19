use lead_node::config::{
    Config, DEFAULT_DRIFT_THRESHOLD, DEFAULT_FIX_FINGERS_INTERVAL_SECS,
    DEFAULT_FRM_GRACE_PERIOD_SECS, DEFAULT_FRM_QUORUM_THRESHOLD, DEFAULT_GRPC_BIND,
    DEFAULT_HEARTBEAT_INTERVAL_SECS, DEFAULT_HTTP_BIND, DEFAULT_JOIN_RETRY_COUNT,
    DEFAULT_JOIN_RETRY_DELAY_SECS, DEFAULT_MIN_KEYS_FOR_DRIFT, DEFAULT_PID_ADJUST_INTERVAL,
    DEFAULT_PID_CENTERING_STEP, DEFAULT_PID_MID_THRESHOLD, DEFAULT_PID_MIN_SAMPLES,
    DEFAULT_PID_SCALE_STEP, DEFAULT_PID_TARGET_RATIO, DEFAULT_PID_UPPER_THRESHOLD,
    DEFAULT_PRUNE_ERROR_RATE, DEFAULT_PRUNE_INACTIVE_SECS, DEFAULT_RANGE_OVERSCAN_MULTIPLIER,
    DEFAULT_RETRAIN_INTERVAL_SECS, DEFAULT_SELF_URI, DEFAULT_STABILIZE_INTERVAL_SECS,
    DEFAULT_SUCCESSOR_LIST_LEN, DEFAULT_VIRTUAL_NODE_COUNT,
};

#[test]
fn default_config_has_correct_values() {
    let cfg = Config::default();

    assert_eq!(cfg.http_bind, DEFAULT_HTTP_BIND);
    assert_eq!(cfg.grpc_bind, DEFAULT_GRPC_BIND);
    assert_eq!(cfg.self_uri, DEFAULT_SELF_URI);
    assert_eq!(cfg.join_uri, None);
    assert_eq!(cfg.virtual_node_count, DEFAULT_VIRTUAL_NODE_COUNT);
    assert_eq!(cfg.successor_list_len, DEFAULT_SUCCESSOR_LIST_LEN);
    assert_eq!(cfg.range_overscan_multiplier, DEFAULT_RANGE_OVERSCAN_MULTIPLIER);
    assert!((cfg.frm_quorum_threshold - DEFAULT_FRM_QUORUM_THRESHOLD).abs() < 1e-9);
    assert!((cfg.drift_threshold - DEFAULT_DRIFT_THRESHOLD).abs() < 1e-9);
    assert_eq!(cfg.min_keys_for_drift, DEFAULT_MIN_KEYS_FOR_DRIFT);
    assert_eq!(cfg.frm_grace_period_secs, DEFAULT_FRM_GRACE_PERIOD_SECS);
    assert_eq!(cfg.pid_adjust_interval, DEFAULT_PID_ADJUST_INTERVAL);
    assert!((cfg.pid_target_ratio - DEFAULT_PID_TARGET_RATIO).abs() < 1e-9);
    assert!((cfg.pid_scale_step - DEFAULT_PID_SCALE_STEP).abs() < 1e-9);
    assert!((cfg.pid_centering_step - DEFAULT_PID_CENTERING_STEP).abs() < 1e-9);
    assert!((cfg.pid_upper_threshold - DEFAULT_PID_UPPER_THRESHOLD).abs() < 1e-9);
    assert!((cfg.pid_mid_threshold - DEFAULT_PID_MID_THRESHOLD).abs() < 1e-9);
    assert_eq!(cfg.pid_min_samples, DEFAULT_PID_MIN_SAMPLES);
    assert!((cfg.prune_error_rate - DEFAULT_PRUNE_ERROR_RATE).abs() < 1e-9);
    assert_eq!(cfg.prune_inactive_secs, DEFAULT_PRUNE_INACTIVE_SECS);
    assert_eq!(cfg.stabilize_interval_secs, DEFAULT_STABILIZE_INTERVAL_SECS);
    assert_eq!(cfg.fix_fingers_interval_secs, DEFAULT_FIX_FINGERS_INTERVAL_SECS);
    assert_eq!(cfg.heartbeat_interval_secs, DEFAULT_HEARTBEAT_INTERVAL_SECS);
    assert_eq!(cfg.maybe_retrain_interval_secs, DEFAULT_RETRAIN_INTERVAL_SECS);
    assert_eq!(cfg.join_retry_count, DEFAULT_JOIN_RETRY_COUNT);
    assert_eq!(cfg.join_retry_delay_secs, DEFAULT_JOIN_RETRY_DELAY_SECS);
}

#[test]
fn config_from_vars_overrides_fields() {
    let vars = vec![
        ("HTTP_BIND", "0.0.0.0:9090"),
        ("GRPC_BIND", "0.0.0.0:60060"),
        ("SELF_URI", "http://node-custom:60060"),
        ("JOIN_URI", "http://node-bootstrap:60060"),
        ("VIRTUAL_NODE_COUNT", "50"),
        ("LEAD_SUCCESSOR_LIST_LEN", "8"),
        ("LEAD_RANGE_OVERSCAN_MULTIPLIER", "5"),
        ("LEAD_FRM_QUORUM_THRESHOLD", "0.85"),
        ("LEAD_DRIFT_THRESHOLD", "0.25"),
        ("LEAD_MIN_KEYS_FOR_DRIFT", "100"),
        ("LEAD_FRM_GRACE_PERIOD_SECS", "30"),
        ("LEAD_PID_ADJUST_INTERVAL", "200"),
        ("LEAD_PID_TARGET_RATIO", "0.98"),
        ("LEAD_PID_SCALE_STEP", "0.08"),
        ("LEAD_PID_CENTERING_STEP", "0.02"),
        ("LEAD_PID_UPPER_THRESHOLD", "0.07"),
        ("LEAD_PID_MID_THRESHOLD", "0.03"),
        ("LEAD_PID_MIN_SAMPLES", "40"),
        ("LEAD_PRUNE_ERROR_RATE", "0.45"),
        ("LEAD_PRUNE_INACTIVE_SECS", "240"),
        ("LEAD_STABILIZE_INTERVAL_SECS", "4"),
        ("LEAD_FIX_FINGERS_INTERVAL_SECS", "12"),
        ("LEAD_CHECK_PREDECESSOR_INTERVAL_SECS", "20"),
        ("LEAD_HEARTBEAT_INTERVAL_SECS", "15"),
        ("LEAD_RETRAIN_INTERVAL_SECS", "10"),
        ("LEAD_JOIN_RETRY_COUNT", "50"),
        ("LEAD_JOIN_RETRY_DELAY_SECS", "2"),
    ];

    let cfg = Config::from_vars(vars);

    assert_eq!(cfg.http_bind, "0.0.0.0:9090");
    assert_eq!(cfg.grpc_bind, "0.0.0.0:60060");
    assert_eq!(cfg.self_uri, "http://node-custom:60060");
    assert_eq!(cfg.join_uri, Some("http://node-bootstrap:60060".to_string()));
    assert_eq!(cfg.virtual_node_count, 50);
    assert_eq!(cfg.successor_list_len, 8);
    assert_eq!(cfg.range_overscan_multiplier, 5);
    assert!((cfg.frm_quorum_threshold - 0.85).abs() < 1e-9);
    assert!((cfg.drift_threshold - 0.25).abs() < 1e-9);
    assert_eq!(cfg.min_keys_for_drift, 100);
    assert_eq!(cfg.frm_grace_period_secs, 30);
    assert_eq!(cfg.pid_adjust_interval, 200);
    assert!((cfg.pid_target_ratio - 0.98).abs() < 1e-9);
    assert!((cfg.pid_scale_step - 0.08).abs() < 1e-9);
    assert!((cfg.pid_centering_step - 0.02).abs() < 1e-9);
    assert!((cfg.pid_upper_threshold - 0.07).abs() < 1e-9);
    assert!((cfg.pid_mid_threshold - 0.03).abs() < 1e-9);
    assert_eq!(cfg.pid_min_samples, 40);
    assert!((cfg.prune_error_rate - 0.45).abs() < 1e-9);
    assert_eq!(cfg.prune_inactive_secs, 240);
    assert_eq!(cfg.stabilize_interval_secs, 4);
    assert_eq!(cfg.fix_fingers_interval_secs, 12);
    assert_eq!(cfg.check_predecessor_interval_secs, 20);
    assert_eq!(cfg.heartbeat_interval_secs, 15);
    assert_eq!(cfg.maybe_retrain_interval_secs, 10);
    assert_eq!(cfg.join_retry_count, 50);
    assert_eq!(cfg.join_retry_delay_secs, 2);
}

#[test]
fn config_from_vars_ignores_invalid_values_and_falls_back_to_defaults() {
    let vars = vec![
        ("VIRTUAL_NODE_COUNT", "not-a-number"),
        ("LEAD_FRM_QUORUM_THRESHOLD", "invalid-float"),
        ("LEAD_STABILIZE_INTERVAL_SECS", "abc"),
        ("JOIN_URI", "   "),
    ];

    let cfg = Config::from_vars(vars);

    assert_eq!(cfg.virtual_node_count, DEFAULT_VIRTUAL_NODE_COUNT);
    assert!((cfg.frm_quorum_threshold - DEFAULT_FRM_QUORUM_THRESHOLD).abs() < 1e-9);
    assert_eq!(cfg.stabilize_interval_secs, DEFAULT_STABILIZE_INTERVAL_SECS);
    assert_eq!(cfg.join_uri, None);
}

#[test]
fn pid_tuner_helper_matches_config() {
    let cfg = Config {
        pid_target_ratio: 0.90,
        pid_scale_step: 0.10,
        pid_centering_step: 0.02,
        pid_upper_threshold: 0.08,
        pid_mid_threshold: 0.04,
        pid_min_samples: 50,
        ..Default::default()
    };

    let tuner = cfg.pid_tuner();
    assert!((tuner.target_ratio - 0.90).abs() < 1e-9);
    assert!((tuner.scale_step - 0.10).abs() < 1e-9);
    assert!((tuner.centering_step - 0.02).abs() < 1e-9);
    assert!((tuner.upper_threshold - 0.08).abs() < 1e-9);
    assert!((tuner.mid_threshold - 0.04).abs() < 1e-9);
    assert_eq!(tuner.min_samples, 50);
}
