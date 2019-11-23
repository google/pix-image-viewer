set -xe
cargo fmt -- --check
cargo +stable test
