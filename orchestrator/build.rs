fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile(
            &[
                "proto/eval.proto",
                "proto/lead.proto",
                "proto/ring.proto",
                "proto/surrogate.proto",
            ],
            &["proto"],
        )?;
    Ok(())
}
