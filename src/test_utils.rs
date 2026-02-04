//! Test utilities for loading reference data and comparing outputs.
//!
//! This module provides helpers for validating Rust implementations against
//! Python reference outputs stored as .npy files.

#[cfg(test)]
use anyhow::{Context, Result};
#[cfg(test)]
use ndarray::ArrayD;
#[cfg(test)]
use ndarray_npy::ReadNpyExt;
#[cfg(test)]
use std::fs::File;
#[cfg(test)]
use std::io::BufReader;
#[cfg(test)]
use std::path::Path;

/// Load a tensor from an .npy file.
#[cfg(test)]
pub fn load_npy<P: AsRef<Path>>(path: P) -> Result<ArrayD<f32>> {
    let file = File::open(path.as_ref())
        .with_context(|| format!("Failed to open: {}", path.as_ref().display()))?;
    let reader = BufReader::new(file);
    let arr = ArrayD::<f32>::read_npy(reader)
        .with_context(|| format!("Failed to read npy: {}", path.as_ref().display()))?;
    Ok(arr)
}

/// Load test data from the test_data directory.
#[cfg(test)]
pub fn load_test_data(name: &str) -> Result<ArrayD<f32>> {
    let path = format!("test_data/{}.npy", name);
    load_npy(&path)
}

/// Compare two arrays with given tolerances.
#[cfg(test)]
pub fn assert_tensors_close(
    reference: &ArrayD<f32>,
    actual: &ArrayD<f32>,
    rtol: f32,
    atol: f32,
    name: &str,
) {
    assert_eq!(
        reference.shape(),
        actual.shape(),
        "{}: shape mismatch: {:?} vs {:?}",
        name,
        reference.shape(),
        actual.shape()
    );

    let mut max_abs_diff: f32 = 0.0;
    let mut max_rel_diff: f32 = 0.0;
    let mut fail_count = 0;

    for (r, a) in reference.iter().zip(actual.iter()) {
        let abs_diff = (r - a).abs();
        let rel_diff = abs_diff / (r.abs() + 1e-10);

        max_abs_diff = max_abs_diff.max(abs_diff);
        max_rel_diff = max_rel_diff.max(rel_diff);

        if abs_diff > atol && rel_diff > rtol {
            fail_count += 1;
        }
    }

    let total = reference.len();
    let pass_rate = (total - fail_count) as f32 / total as f32 * 100.0;

    if fail_count > 0 {
        panic!(
            "{}: tensor mismatch\n\
             Max abs diff: {:.2e}\n\
             Max rel diff: {:.2e}\n\
             Pass rate: {:.1}% ({}/{})\n\
             Reference mean: {:.4}\n\
             Actual mean: {:.4}",
            name,
            max_abs_diff,
            max_rel_diff,
            pass_rate,
            total - fail_count,
            total,
            reference.mean().unwrap_or(0.0),
            actual.mean().unwrap_or(0.0),
        );
    }

    println!(
        "{}: PASS (max_abs={:.2e}, max_rel={:.2e})",
        name, max_abs_diff, max_rel_diff
    );
}

/// Check if test data exists.
#[cfg(test)]
pub fn test_data_exists(name: &str) -> bool {
    Path::new(&format!("test_data/{}.npy", name)).exists()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_test_data() {
        // Skip if test data doesn't exist
        if !test_data_exists("rms_norm_input") {
            println!("Skipping: test_data not generated. Run: ./scripts/reference_forward.py");
            return;
        }

        let input = load_test_data("rms_norm_input").unwrap();
        assert_eq!(input.shape(), &[1, 10, 1280]);

        let output = load_test_data("rms_norm_output").unwrap();
        assert_eq!(output.shape(), &[1, 10, 1280]);
    }
}
