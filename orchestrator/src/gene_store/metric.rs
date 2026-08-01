/// SRP / OCP: a single capability — measuring dissimilarity between
/// two gene vectors. New metrics (cosine, manhattan, mahalanobis) are
/// added by implementing this trait without touching callers.
pub trait DistanceMetric: Send + Sync {
    fn distance(&self, a: &[f64], b: &[f64]) -> f64;
}

/// L2 (Euclidean) distance. Default for continuous gene spaces.
#[derive(Debug, Clone, Copy, Default)]
pub struct EuclideanDistance;

impl DistanceMetric for EuclideanDistance {
    fn distance(&self, a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}
