//! Thread-safe Circuit Breaker for gRPC transport resiliency.
//!
//! Prevents hammering degraded or dropping upstreams (such as during dynamic
//! worker scaling before Envoy DNS cache refreshes) by fast-pausing traffic
//! upon failure spikes and gradually probing recovery.

use std::sync::Mutex;
use std::time::{Duration, Instant};
use tonic::Status;
use tracing::{debug, info, warn};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Number of consecutive failures to trip the circuit from Closed to Open.
    pub failure_threshold: usize,
    /// Time to remain Open before transitioning to HalfOpen.
    pub recovery_timeout: Duration,
    /// Number of successful probe executions in HalfOpen to transition back to Closed.
    pub half_open_probes: usize,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            recovery_timeout: Duration::from_millis(1500),
            half_open_probes: 2,
        }
    }
}

struct Inner {
    state: CircuitState,
    consecutive_failures: usize,
    consecutive_successes: usize,
    opened_at: Option<Instant>,
    active_half_open_probes: usize,
}

/// SRP: Tracks failure and success health of upstream service and manages
/// circuit transition states (Closed, Open, HalfOpen).
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    inner: Mutex<Inner>,
}

impl CircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            inner: Mutex::new(Inner {
                state: CircuitState::Closed,
                consecutive_failures: 0,
                consecutive_successes: 0,
                opened_at: None,
                active_half_open_probes: 0,
            }),
        }
    }

    /// Returns the current state of the circuit breaker.
    pub fn state(&self) -> CircuitState {
        let mut inner = self.inner.lock().unwrap();
        self.check_transition(&mut inner);
        inner.state
    }

    /// Checks if a call is permitted to execute immediately.
    pub fn can_execute(&self) -> bool {
        let mut inner = self.inner.lock().unwrap();
        self.check_transition(&mut inner);
        match inner.state {
            CircuitState::Closed => true,
            CircuitState::Open => false,
            CircuitState::HalfOpen => {
                if inner.active_half_open_probes < self.config.half_open_probes {
                    inner.active_half_open_probes += 1;
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Asynchronously waits until the circuit is ready to accept a call, or
    /// returns an Unavailable status if `max_wait` is exceeded.
    pub async fn wait_until_ready(&self, max_wait: Duration) -> Result<(), Status> {
        let deadline = Instant::now() + max_wait;
        loop {
            let sleep_duration = {
                let mut inner = self.inner.lock().unwrap();
                self.check_transition(&mut inner);
                match inner.state {
                    CircuitState::Closed => return Ok(()),
                    CircuitState::HalfOpen => {
                        if inner.active_half_open_probes < self.config.half_open_probes {
                            inner.active_half_open_probes += 1;
                            return Ok(());
                        }
                        Duration::from_millis(50)
                    }
                    CircuitState::Open => {
                        if let Some(opened_at) = inner.opened_at {
                            let elapsed = opened_at.elapsed();
                            if elapsed >= self.config.recovery_timeout {
                                self.transition_to_half_open(&mut inner);
                                inner.active_half_open_probes += 1;
                                return Ok(());
                            } else {
                                self.config.recovery_timeout - elapsed
                            }
                        } else {
                            Duration::from_millis(50)
                        }
                    }
                }
            };

            if Instant::now() + sleep_duration > deadline {
                return Err(Status::unavailable(
                    "circuit breaker open: timeout exceeded while waiting for recovery",
                ));
            }
            tokio::time::sleep(sleep_duration.min(Duration::from_millis(200))).await;
        }
    }

    /// Records an upstream success.
    pub fn record_success(&self) {
        let mut inner = self.inner.lock().unwrap();
        match inner.state {
            CircuitState::Closed => {
                inner.consecutive_failures = 0;
            }
            CircuitState::HalfOpen => {
                if inner.active_half_open_probes > 0 {
                    inner.active_half_open_probes -= 1;
                }
                inner.consecutive_successes += 1;
                debug!(
                    "CircuitBreaker HalfOpen probe success ({}/{})",
                    inner.consecutive_successes, self.config.half_open_probes
                );
                if inner.consecutive_successes >= self.config.half_open_probes {
                    self.transition_to_closed(&mut inner);
                }
            }
            CircuitState::Open => {}
        }
    }

    /// Records an upstream failure.
    pub fn record_failure(&self) {
        let mut inner = self.inner.lock().unwrap();
        match inner.state {
            CircuitState::Closed => {
                inner.consecutive_failures += 1;
                debug!(
                    "CircuitBreaker failure registered ({}/{})",
                    inner.consecutive_failures, self.config.failure_threshold
                );
                if inner.consecutive_failures >= self.config.failure_threshold {
                    self.transition_to_open(&mut inner);
                }
            }
            CircuitState::HalfOpen => {
                if inner.active_half_open_probes > 0 {
                    inner.active_half_open_probes -= 1;
                }
                warn!("CircuitBreaker probe failed in HalfOpen; tripping back to Open");
                self.transition_to_open(&mut inner);
            }
            CircuitState::Open => {
                // Update opened_at to delay recovery if failures continue arriving
                inner.opened_at = Some(Instant::now());
            }
        }
    }

    /// Resets the circuit breaker to clean Closed state.
    pub fn reset(&self) {
        let mut inner = self.inner.lock().unwrap();
        self.transition_to_closed(&mut inner);
    }

    fn check_transition(&self, inner: &mut Inner) {
        if inner.state == CircuitState::Open {
            if let Some(opened_at) = inner.opened_at {
                if opened_at.elapsed() >= self.config.recovery_timeout {
                    self.transition_to_half_open(inner);
                }
            }
        }
    }

    fn transition_to_open(&self, inner: &mut Inner) {
        inner.state = CircuitState::Open;
        inner.opened_at = Some(Instant::now());
        inner.consecutive_failures = 0;
        inner.consecutive_successes = 0;
        inner.active_half_open_probes = 0;
        warn!(
            "CircuitBreaker TRIPPED to OPEN (cooling down for {:?})",
            self.config.recovery_timeout
        );
    }

    fn transition_to_half_open(&self, inner: &mut Inner) {
        inner.state = CircuitState::HalfOpen;
        inner.consecutive_successes = 0;
        inner.active_half_open_probes = 0;
        info!("CircuitBreaker transitioning to HALF-OPEN (probing upstream)");
    }

    fn transition_to_closed(&self, inner: &mut Inner) {
        inner.state = CircuitState::Closed;
        inner.consecutive_failures = 0;
        inner.consecutive_successes = 0;
        inner.opened_at = None;
        inner.active_half_open_probes = 0;
        info!("CircuitBreaker RECOVERED to CLOSED");
    }
}
