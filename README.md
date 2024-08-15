# Marlin

Welcome to **Marlin**! This repository is dedicated to exploring and experimenting with Rust-based solutions for deep learning inferencing. Our goal is to leverage Rust's performance and safety features to improve the efficiency of serving large deep learning models.

## Overview

Marlin is an experimental project focused on evaluating Rust's capabilities in handling deep learning inference tasks. We compare Rust-based implementations with Python counterparts to assess performance, latency, and scalability.

## Features

- **Rust-based Inference**: Implementation of deep learning models using Rust, with emphasis on performance and thread safety.
- **Model Serving**: Use of `Candle` from Hugging Face and `actix-web` for serving models.
- **Benchmarking**: Tools and scripts for performance and encoding time benchmarks.
- **Comparative Analysis**: Side-by-side comparison with Python-based implementations to evaluate performance differences.

## Getting Started

### Prerequisites

- Rust (1.67.0 or later)
- `cargo` (Rust package manager and build tool)

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/AbhishekBose/marlin.git
   cd marlin
   ```

2. Build the project:
   ```sh
   cargo build
   ```

### Running the Project

1. **Start the Rust-based server**:
   ```sh
   cargo run --release
   ```

2. **Run the Python-based server** (if applicable):
   ```sh
   cd scripts
   pip fastapi uvicorn sentence-transformers
   uvicorn main:app --reload
   ```

3. **Perform Load and Encoding Benchmarks**:
    - Use the provided benchmarking scripts in the `scripts/` directory to test performance and compare results.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have suggestions, improvements, or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or discussions, feel free to reach out to [Abhishek Bose](mailto:abose550@gmail.com).

---

Happy experimenting! 🚀