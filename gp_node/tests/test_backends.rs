use gp_node::{
    BackendFactory, ComputeBackend, CpuOpenMpBackend, GpHyperparameters, KernelType, ReferenceKernel,
};

#[test]
fn test_cpu_openmp_backend_covariance_parity() {
    let cpu_backend = CpuOpenMpBackend::new().expect("CPU backend failed to initialize");

    let x1 = vec![
        0.0, 1.0, 2.0,
        1.5, 0.5, -1.0,
        3.0, 2.0, 1.0,
    ];
    let n1 = 3;
    let dim = 3;

    let params = GpHyperparameters::new(vec![1.2, 0.8, 2.0], 1.5, 0.01, KernelType::Matern52);

    let k_matrix = cpu_backend
        .compute_covariance(&x1, n1, &x1, n1, dim, &params, true)
        .expect("Covariance calculation failed");

    assert_eq!(k_matrix.len(), n1 * n1);

    // Verify against ReferenceKernel
    for i in 0..n1 {
        let row_i = &x1[i * dim..(i + 1) * dim];
        for j in 0..n1 {
            let row_j = &x1[j * dim..(j + 1) * dim];
            let mut expected = ReferenceKernel::eval_kernel(row_i, row_j, &params);
            if i == j {
                expected += params.noise_variance;
            }
            let actual = k_matrix[i * n1 + j];
            assert!(
                (actual - expected).abs() < 1e-6,
                "Mismatch at ({}, {}): actual {}, expected {}",
                i, j, actual, expected
            );
        }
    }
}

#[test]
fn test_device_probing() {
    let devices = BackendFactory::probe_devices();
    assert!(!devices.is_empty(), "Must detect at least one compute device");
    println!("Detected devices: {:?}", devices);
}
