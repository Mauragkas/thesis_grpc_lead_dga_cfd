use std::collections::VecDeque;

/// A single evaluated training point (genes -> true fitness).
#[derive(Debug, Clone)]
pub struct SampleRecord {
    pub genes: Vec<f64>,
    pub fitness: f64,
}

/// Single Responsibility: maintains a bounded FIFO sliding window buffer
/// of evaluated design points for online surrogate model retraining.
pub struct SlidingWindowBuffer {
    capacity: usize,
    buffer: VecDeque<SampleRecord>,
    dim: Option<usize>,
}

impl SlidingWindowBuffer {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "Capacity must be positive");
        Self {
            capacity,
            buffer: VecDeque::with_capacity(capacity),
            dim: None,
        }
    }

    /// Inserts a single sample. If capacity is exceeded, the oldest sample is evicted.
    pub fn push(&mut self, genes: Vec<f64>, fitness: f64) -> bool {
        if genes.is_empty() {
            return false;
        }

        if let Some(d) = self.dim {
            if genes.len() != d {
                return false;
            }
        } else {
            self.dim = Some(genes.len());
        }

        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }

        self.buffer.push_back(SampleRecord { genes, fitness });
        true
    }

    /// Inserts multiple samples. Returns count of successfully ingested samples.
    pub fn push_batch(&mut self, samples: Vec<(Vec<f64>, f64)>) -> usize {
        let mut ingested = 0;
        for (g, f) in samples {
            if self.push(g, f) {
                ingested += 1;
            }
        }
        ingested
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn dimension(&self) -> Option<usize> {
        self.dim
    }

    /// Extracts the current sliding window contents into flat contiguous arrays:
    /// returns `(X_matrix, Y_vector, num_samples, dimension)`.
    pub fn extract_dataset(&self) -> Option<(Vec<f64>, Vec<f64>, usize, usize)> {
        let n = self.buffer.len();
        let dim = self.dim?;
        if n == 0 {
            return None;
        }

        let mut x = Vec::with_capacity(n * dim);
        let mut y = Vec::with_capacity(n);

        for rec in &self.buffer {
            x.extend_from_slice(&rec.genes);
            y.push(rec.fitness);
        }

        Some((x, y, n, dim))
    }

    /// Clears the entire buffer.
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.dim = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_sliding_window_eviction() {
        let mut buf = SlidingWindowBuffer::new(3);
        assert_eq!(buf.len(), 0);

        assert!(buf.push(vec![1.0, 2.0], 10.0));
        assert!(buf.push(vec![3.0, 4.0], 20.0));
        assert!(buf.push(vec![5.0, 6.0], 30.0));
        assert_eq!(buf.len(), 3);

        // 4th sample should evict the 1st
        assert!(buf.push(vec![7.0, 8.0], 40.0));
        assert_eq!(buf.len(), 3);

        let (x, y, n, dim) = buf.extract_dataset().unwrap();
        assert_eq!(n, 3);
        assert_eq!(dim, 2);
        assert_eq!(y, vec![20.0, 30.0, 40.0]);
        assert_eq!(x, vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_dimension_mismatch_rejected() {
        let mut buf = SlidingWindowBuffer::new(5);
        assert!(buf.push(vec![1.0, 2.0], 10.0));
        // Wrong dimension (3 instead of 2)
        assert!(!buf.push(vec![1.0, 2.0, 3.0], 20.0));
        assert_eq!(buf.len(), 1);
    }
}
