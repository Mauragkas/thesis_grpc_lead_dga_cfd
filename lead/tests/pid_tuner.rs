use lead_node::lead::learning::PidTuner;
use lead_node::rmi::{Anchor, PidState};

fn default_tuner() -> PidTuner {
    PidTuner::default()
}

#[test]
fn below_min_samples_is_noop() {
    let t = default_tuner();
    let mut state = PidState::default();
    let mut anchor = Anchor::default();
    t.adjust(&mut state, &mut anchor, 5, 5); // total 10 < min_samples(20)
    assert_eq!(state, PidState::default());
    assert_eq!(anchor.scale, 1.0);
    assert_eq!(anchor.offset, 0.0);
}

#[test]
fn all_outside_scales_down_and_resets_integral() {
    let t = default_tuner();
    let mut state = PidState::default();
    let mut anchor = Anchor::default();
    // actual = 0/100 = 0.0, error = -0.95 → pid < 0.5 → scale down.
    t.adjust(&mut state, &mut anchor, 0, 100);
    assert!((anchor.scale - 0.95).abs() < 1e-9);
    assert_eq!(state.proportional, 0);
    assert_eq!(state.integral, 1);
    assert!(!state.derivative_sign);
    assert_eq!(state.derivative_mag, 0);
    // offset centering: error < 0 → offset increases toward window.
    assert!(anchor.offset > 0.0);
}

#[test]
fn on_target_accumulates_integral_without_scaling() {
    let t = default_tuner();
    let mut state = PidState::default();
    let mut anchor = Anchor::default();
    // actual = 95/100 = 0.95 == target → error 0.0 → within band.
    t.adjust(&mut state, &mut anchor, 95, 5);
    assert!((anchor.scale - 1.0).abs() < 1e-9);
    assert_eq!(state.proportional, 1);
    assert_eq!(state.integral, 1);
    assert!(!state.derivative_sign);
    assert_eq!(state.derivative_mag, 1);
    assert!((anchor.offset - 0.0).abs() < 1e-9);
}

#[test]
fn default_constants_match_original_magic_numbers() {
    let t = default_tuner();
    assert!((t.target_ratio - 0.95).abs() < 1e-9);
    assert!((t.scale_step - 0.05).abs() < 1e-9);
    assert!((t.centering_step - 0.01).abs() < 1e-9);
    assert!((t.upper_threshold - 0.05).abs() < 1e-9);
    assert!((t.mid_threshold - 0.02).abs() < 1e-9);
    assert_eq!(t.min_samples, 20);
}
