use gp_node::{BackendFactory, GpHyperparameters, KernelType, ReferenceKernel};

#[test]
fn test_matern52_kernel_reference() {
    let x1 = vec![0.0, 1.0];
    let x2 = vec![0.0, 1.0];
    let params = GpHyperparameters::new(vec![1.0, 1.0], 2.5, 1e-4, KernelType::Matern52);

    // Self distance is 0 -> kernel should equal signal_variance (2.5)
    let val_self = ReferenceKernel::eval_kernel(&x1, &x2, &params);
    assert!((val_self - 2.5).abs() < 1e-12);

    let x3 = vec![1.0, 1.0];
    // Distance = 1.0, r = 1.0
    // k = 2.5 * (1 + sqrt(5) + 5/3) * exp(-sqrt(5))
    let sqrt5 = 5.0f64.sqrt();
    let expected = 2.5 * (1.0 + sqrt5 + 5.0 / 3.0) * (-sqrt5).exp();
    let val_dist1 = ReferenceKernel::eval_kernel(&x1, &x3, &params);
    assert!((val_dist1 - expected).abs() < 1e-12);
}

#[test]
fn test_cholesky_and_triangular_solver() {
    let backend = BackendFactory::create_best_backend();

    // Positive definite 3x3 matrix:
    // A = [ [4.0, 12.0, -16.0],
    //       [12.0, 37.0, -43.0],
    //       [-16.0, -43.0, 98.0] ]
    // Lower Cholesky L should be:
    // L = [ [2.0, 0.0, 0.0],
    //       [6.0, 1.0, 0.0],
    //       [-8.0, 5.0, 3.0] ]
    let a = vec![
        4.0, 12.0, -16.0,
        12.0, 37.0, -43.0,
        -16.0, -43.0, 98.0,
    ];
    let n = 3;

    let l = backend.cholesky(&a, n).expect("Cholesky failed");

    let expected_l = vec![
        2.0, 0.0, 0.0,
        6.0, 1.0, 0.0,
        -8.0, 5.0, 3.0,
    ];

    for i in 0..9 {
        assert!((l[i] - expected_l[i]).abs() < 1e-6, "Mismatch at index {}: got {}, expected {}", i, l[i], expected_l[i]);
    }

    // Verify L * L^T = A
    for i in 0..n {
        for j in 0..n {
            let mut dot = 0.0;
            for k in 0..n {
                dot += l[i * n + k] * l[j * n + k];
            }
            assert!((dot - a[i * n + j]).abs() < 1e-6);
        }
    }

    // Test linear solve A * x = b
    let b = vec![1.0, 2.0, 3.0];
    let x = backend.compute_alpha(&l, &b, n).expect("Alpha compute failed");

    // Check A * x = b
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..n {
            sum += a[i * n + j] * x[j];
        }
        assert!((sum - b[i]).abs() < 1e-6);
    }
}
