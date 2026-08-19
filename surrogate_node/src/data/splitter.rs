use crate::domain::{DatasetPartition, DatasetSplit};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

pub struct DatasetSplitter;

impl DatasetSplitter {
    /// Splits raw dataset matrix X (N x dim) and target vector y (N) into 60% Train, 20% Val, 20% Test.
    pub fn split_60_20_20(
        x: &[f64],
        y: &[f64],
        num_samples: usize,
        dim: usize,
        seed: u64,
    ) -> DatasetSplit {
        assert_eq!(x.len(), num_samples * dim, "X dimension mismatch");
        assert_eq!(y.len(), num_samples, "y dimension mismatch");

        let mut indices: Vec<usize> = (0..num_samples).collect();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        indices.shuffle(&mut rng);

        let n_train = (num_samples * 6) / 10;
        let n_val = (num_samples * 2) / 10;

        let train_indices = &indices[0..n_train];
        let val_indices = &indices[n_train..n_train + n_val];
        let test_indices = &indices[n_train + n_val..];

        let train = Self::gather_partition(x, y, train_indices, dim);
        let validation = Self::gather_partition(x, y, val_indices, dim);
        let test = Self::gather_partition(x, y, test_indices, dim);

        DatasetSplit {
            train,
            validation,
            test,
        }
    }

    fn gather_partition(
        x: &[f64],
        y: &[f64],
        indices: &[usize],
        dim: usize,
    ) -> DatasetPartition {
        let n = indices.len();
        let mut part_x = Vec::with_capacity(n * dim);
        let mut part_y = Vec::with_capacity(n);

        for &idx in indices {
            let start = idx * dim;
            part_x.extend_from_slice(&x[start..start + dim]);
            part_y.push(y[idx]);
        }

        DatasetPartition::new(part_x, part_y, n, dim)
    }
}
