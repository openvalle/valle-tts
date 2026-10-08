fn main() {
    println!("cargo:rerun-if-changed=native");
    println!("cargo:rerun-if-changed=vendor/qwentts");
    println!("cargo:rerun-if-changed=vendor/ggml");
    #[cfg(feature = "qwen3")]
    {
        let dst = cmake::Config::new("native")
            .profile("Release")
            .define("BUILD_SHARED_LIBS", "OFF")
            .define("GGML_NATIVE", "OFF")
            .define("GGML_CPU", "ON")
            .define("GGML_METAL", "OFF")
            .define("GGML_ACCELERATE", "OFF")
            .define("GGML_BLAS", "OFF")
            .define("GGML_OPENMP", "OFF")
            .define("GGML_CUDA", "OFF")
            .define("GGML_VULKAN", "OFF")
            .define("GGML_BUILD_TESTS", "OFF")
            .define("GGML_BUILD_EXAMPLES", "OFF")
            .build();
        println!("cargo:rustc-link-search=native={}/lib", dst.display());
        for lib in ["valle_qwen", "ggml", "ggml-cpu", "ggml-base"] {
            println!("cargo:rustc-link-lib=static={lib}");
        }
        let target = std::env::var("TARGET").expect("Cargo sets TARGET");
        if target.contains("apple") {
            println!("cargo:rustc-link-lib=c++");
        } else if !target.contains("msvc") {
            println!("cargo:rustc-link-lib=stdc++");
            println!("cargo:rustc-link-lib=pthread");
            println!("cargo:rustc-link-lib=dl");
        }
    }
}
