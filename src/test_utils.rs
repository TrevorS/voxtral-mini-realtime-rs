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
