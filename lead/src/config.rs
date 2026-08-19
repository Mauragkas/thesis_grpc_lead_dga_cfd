use std::collections::HashMap;
use tracing::warn;

use crate::lead::learning::PidTuner;

/// Default configuration constants
pub const DEFAULT_HTTP_BIND: &str = "0.0.0.0:8080";
pub const DEFAULT_GRPC_BIND: &str = "0.0.0.0:50051";
pub const DEFAULT_SELF_URI: &str = "http://127.0.0.1:50051";
pub const DEFAULT_VIRTUAL_NODE_COUNT: usize = 10;
pub const DEFAULT_SUCCESSOR_LIST_LEN: usize = 4; // R
pub const DEFAULT_RANGE_OVERSCAN_MULTIPLIER: usize = 3;
pub const DEFAULT_FRM_QUORUM_THRESHOLD: f64 = 0.90;
pub const DEFAULT_DRIFT_THRESHOLD: f64 = 0.40;
pub const DEFAULT_MIN_KEYS_FOR_DRIFT: usize = 50;
pub const DEFAULT_FRM_GRACE_PERIOD_SECS: u64 = 10;
pub const DEFAULT_PID_ADJUST_INTERVAL: usize = 100;
pub const DEFAULT_PID_TARGET_RATIO: f64 = 0.95;
pub const DEFAULT_PID_SCALE_STEP: f64 = 0.05;
pub const DEFAULT_PID_CENTERING_STEP: f64 = 0.01;
pub const DEFAULT_PID_UPPER_THRESHOLD: f64 = 0.05;
pub const DEFAULT_PID_MID_THRESHOLD: f64 = 0.02;
pub const DEFAULT_PID_MIN_SAMPLES: usize = 20;
pub const DEFAULT_PRUNE_ERROR_RATE: f64 = 0.30;
pub const DEFAULT_PRUNE_INACTIVE_SECS: u64 = 120;
pub const DEFAULT_STABILIZE_INTERVAL_SECS: u64 = 2;
pub const DEFAULT_FIX_FINGERS_INTERVAL_SECS: u64 = 5;
pub const DEFAULT_CHECK_PREDECESSOR_INTERVAL_SECS: u64 = 10;
pub const DEFAULT_HEARTBEAT_INTERVAL_SECS: u64 = 8;
pub const DEFAULT_RETRAIN_INTERVAL_SECS: u64 = 5;
pub const DEFAULT_JOIN_RETRY_COUNT: usize = 300;
pub const DEFAULT_JOIN_RETRY_DELAY_SECS: u64 = 1;

/// Complete runtime configuration for a LEAD node.
/// Encapsulates network addressing, ring topology, FRM learning,
/// PID anchor tuning, pruning, range query overscan, and maintenance intervals.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    // Network & Topology
    pub http_bind: String,
    pub grpc_bind: String,
    pub self_uri: String,
    pub join_uri: Option<String>,
    pub virtual_node_count: usize,
    pub successor_list_len: usize,

    // Range queries
    pub range_overscan_multiplier: usize,

    // Federated Learning (FRM)
    pub frm_quorum_threshold: f64,
    pub drift_threshold: f64,
    pub min_keys_for_drift: usize,
    pub frm_grace_period_secs: u64,

    // PID Anchor Tuning
    pub pid_adjust_interval: usize,
    pub pid_target_ratio: f64,
    pub pid_scale_step: f64,
    pub pid_centering_step: f64,
    pub pid_upper_threshold: f64,
    pub pid_mid_threshold: f64,
    pub pid_min_samples: usize,

    // Shadow Balancer / VNode Pruning
    pub prune_error_rate: f64,
    pub prune_inactive_secs: u64,

    // Maintenance loop intervals (seconds)
    pub stabilize_interval_secs: u64,
    pub fix_fingers_interval_secs: u64,
    pub check_predecessor_interval_secs: u64,
    pub heartbeat_interval_secs: u64,
    pub maybe_retrain_interval_secs: u64,
    pub join_retry_count: usize,
    pub join_retry_delay_secs: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            http_bind: DEFAULT_HTTP_BIND.to_string(),
            grpc_bind: DEFAULT_GRPC_BIND.to_string(),
            self_uri: DEFAULT_SELF_URI.to_string(),
            join_uri: None,
            virtual_node_count: DEFAULT_VIRTUAL_NODE_COUNT,
            successor_list_len: DEFAULT_SUCCESSOR_LIST_LEN,
            range_overscan_multiplier: DEFAULT_RANGE_OVERSCAN_MULTIPLIER,
            frm_quorum_threshold: DEFAULT_FRM_QUORUM_THRESHOLD,
            drift_threshold: DEFAULT_DRIFT_THRESHOLD,
            min_keys_for_drift: DEFAULT_MIN_KEYS_FOR_DRIFT,
            frm_grace_period_secs: DEFAULT_FRM_GRACE_PERIOD_SECS,
            pid_adjust_interval: DEFAULT_PID_ADJUST_INTERVAL,
            pid_target_ratio: DEFAULT_PID_TARGET_RATIO,
            pid_scale_step: DEFAULT_PID_SCALE_STEP,
            pid_centering_step: DEFAULT_PID_CENTERING_STEP,
            pid_upper_threshold: DEFAULT_PID_UPPER_THRESHOLD,
            pid_mid_threshold: DEFAULT_PID_MID_THRESHOLD,
            pid_min_samples: DEFAULT_PID_MIN_SAMPLES,
            prune_error_rate: DEFAULT_PRUNE_ERROR_RATE,
            prune_inactive_secs: DEFAULT_PRUNE_INACTIVE_SECS,
            stabilize_interval_secs: DEFAULT_STABILIZE_INTERVAL_SECS,
            fix_fingers_interval_secs: DEFAULT_FIX_FINGERS_INTERVAL_SECS,
            check_predecessor_interval_secs: DEFAULT_CHECK_PREDECESSOR_INTERVAL_SECS,
            heartbeat_interval_secs: DEFAULT_HEARTBEAT_INTERVAL_SECS,
            maybe_retrain_interval_secs: DEFAULT_RETRAIN_INTERVAL_SECS,
            join_retry_count: DEFAULT_JOIN_RETRY_COUNT,
            join_retry_delay_secs: DEFAULT_JOIN_RETRY_DELAY_SECS,
        }
    }
}

impl Config {
    /// Constructs a `PidTuner` initialized with the PID settings from this `Config`.
    pub fn pid_tuner(&self) -> PidTuner {
        PidTuner {
            target_ratio: self.pid_target_ratio,
            scale_step: self.pid_scale_step,
            centering_step: self.pid_centering_step,
            upper_threshold: self.pid_upper_threshold,
            mid_threshold: self.pid_mid_threshold,
            min_samples: self.pid_min_samples,
        }
    }

    /// Loads configuration by querying process environment variables.
    pub fn from_env() -> Self {
        Self::from_vars(std::env::vars())
    }

    /// Pure parser reading overrides from an iterator of key-value pairs.
    pub fn from_vars<I, K, V>(vars: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
        V: AsRef<str>,
    {
        let map: HashMap<String, String> = vars
            .into_iter()
            .map(|(k, v)| (k.as_ref().to_string(), v.as_ref().to_string()))
            .collect();

        let mut cfg = Self::default();

        if let Some(v) = map.get("HTTP_BIND") {
            cfg.http_bind = v.clone();
        }
        if let Some(v) = map.get("GRPC_BIND") {
            cfg.grpc_bind = v.clone();
        }
        if let Some(v) = map.get("SELF_URI") {
            cfg.self_uri = v.clone();
        }
        if let Some(v) = map.get("JOIN_URI") {
            if !v.trim().is_empty() {
                cfg.join_uri = Some(v.clone());
            }
        }

        parse_usize(&map, &["VIRTUAL_NODE_COUNT", "LEAD_VIRTUAL_NODE_COUNT"], &mut cfg.virtual_node_count);
        parse_usize(&map, &["LEAD_SUCCESSOR_LIST_LEN", "SUCCESSOR_LIST_LEN"], &mut cfg.successor_list_len);
        parse_usize(&map, &["LEAD_RANGE_OVERSCAN_MULTIPLIER", "RANGE_OVERSCAN_MULTIPLIER"], &mut cfg.range_overscan_multiplier);
        parse_f64(&map, &["LEAD_FRM_QUORUM_THRESHOLD", "FRM_QUORUM_THRESHOLD"], &mut cfg.frm_quorum_threshold);
        parse_f64(&map, &["LEAD_DRIFT_THRESHOLD", "DRIFT_THRESHOLD"], &mut cfg.drift_threshold);
        parse_usize(&map, &["LEAD_MIN_KEYS_FOR_DRIFT", "MIN_KEYS_FOR_DRIFT"], &mut cfg.min_keys_for_drift);
        parse_u64(&map, &["LEAD_FRM_GRACE_PERIOD_SECS", "FRM_GRACE_PERIOD_SECS"], &mut cfg.frm_grace_period_secs);
        parse_usize(&map, &["LEAD_PID_ADJUST_INTERVAL", "PID_ADJUST_INTERVAL"], &mut cfg.pid_adjust_interval);
        parse_f64(&map, &["LEAD_PID_TARGET_RATIO", "PID_TARGET_RATIO"], &mut cfg.pid_target_ratio);
        parse_f64(&map, &["LEAD_PID_SCALE_STEP", "PID_SCALE_STEP"], &mut cfg.pid_scale_step);
        parse_f64(&map, &["LEAD_PID_CENTERING_STEP", "PID_CENTERING_STEP"], &mut cfg.pid_centering_step);
        parse_f64(&map, &["LEAD_PID_UPPER_THRESHOLD", "PID_UPPER_THRESHOLD"], &mut cfg.pid_upper_threshold);
        parse_f64(&map, &["LEAD_PID_MID_THRESHOLD", "PID_MID_THRESHOLD"], &mut cfg.pid_mid_threshold);
        parse_usize(&map, &["LEAD_PID_MIN_SAMPLES", "PID_MIN_SAMPLES"], &mut cfg.pid_min_samples);
        parse_f64(&map, &["LEAD_PRUNE_ERROR_RATE", "PRUNE_ERROR_RATE"], &mut cfg.prune_error_rate);
        parse_u64(&map, &["LEAD_PRUNE_INACTIVE_SECS", "PRUNE_INACTIVE_SECS"], &mut cfg.prune_inactive_secs);
        parse_u64(&map, &["LEAD_STABILIZE_INTERVAL_SECS", "STABILIZE_INTERVAL_SECS"], &mut cfg.stabilize_interval_secs);
        parse_u64(&map, &["LEAD_FIX_FINGERS_INTERVAL_SECS", "FIX_FINGERS_INTERVAL_SECS"], &mut cfg.fix_fingers_interval_secs);
        parse_u64(&map, &["LEAD_CHECK_PREDECESSOR_INTERVAL_SECS", "CHECK_PREDECESSOR_INTERVAL_SECS"], &mut cfg.check_predecessor_interval_secs);
        parse_u64(&map, &["LEAD_HEARTBEAT_INTERVAL_SECS", "HEARTBEAT_INTERVAL_SECS"], &mut cfg.heartbeat_interval_secs);
        parse_u64(&map, &["LEAD_RETRAIN_INTERVAL_SECS", "RETRAIN_INTERVAL_SECS"], &mut cfg.maybe_retrain_interval_secs);
        parse_usize(&map, &["LEAD_JOIN_RETRY_COUNT", "JOIN_RETRY_COUNT"], &mut cfg.join_retry_count);
        parse_u64(&map, &["LEAD_JOIN_RETRY_DELAY_SECS", "JOIN_RETRY_DELAY_SECS"], &mut cfg.join_retry_delay_secs);

        cfg
    }
}

fn parse_usize(map: &HashMap<String, String>, keys: &[&str], target: &mut usize) {
    for key in keys {
        if let Some(v) = map.get(*key) {
            match v.trim().parse::<usize>() {
                Ok(parsed) => {
                    *target = parsed;
                    return;
                }
                Err(e) => {
                    warn!("Failed to parse {key}={v:?} as usize: {e}; keeping default {target}");
                }
            }
        }
    }
}

fn parse_u64(map: &HashMap<String, String>, keys: &[&str], target: &mut u64) {
    for key in keys {
        if let Some(v) = map.get(*key) {
            match v.trim().parse::<u64>() {
                Ok(parsed) => {
                    *target = parsed;
                    return;
                }
                Err(e) => {
                    warn!("Failed to parse {key}={v:?} as u64: {e}; keeping default {target}");
                }
            }
        }
    }
}

fn parse_f64(map: &HashMap<String, String>, keys: &[&str], target: &mut f64) {
    for key in keys {
        if let Some(v) = map.get(*key) {
            match v.trim().parse::<f64>() {
                Ok(parsed) => {
                    *target = parsed;
                    return;
                }
                Err(e) => {
                    warn!("Failed to parse {key}={v:?} as f64: {e}; keeping default {target}");
                }
            }
        }
    }
}
