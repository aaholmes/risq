//! Integration tests for the refactored context-based architecture

use risq::RisqContext;
use risq::RisqError;

#[test]
fn test_context_creation_invalid_files() {
    let result = RisqContext::from_files("nonexistent.json", "nonexistent.fcidump");
    assert!(result.is_err());
    
    if let Err(e) = result {
        println!("Expected error: {}", e);
        assert!(e.to_string().contains("nonexistent.json"));
    }
}

#[test]
fn test_context_error_types() {
    // Test IO error
    let result = RisqContext::from_files("/dev/null/impossible", "test");
    match result {
        Err(RisqError::Io { path, .. }) => {
            assert!(path.contains("impossible"));
        },
        _ => panic!("Expected IO error"),
    }
}

#[test] 
fn test_context_json_validation() {
    use std::io::Write;
    use tempfile::NamedTempFile;
    
    // Create invalid JSON file
    let mut temp_file = NamedTempFile::new().unwrap();
    writeln!(temp_file, "{{ invalid json }").unwrap();
    
    let result = RisqContext::from_files(temp_file.path(), "nonexistent");
    match result {
        Err(RisqError::Json { .. }) => {
            // Expected JSON parsing error
        },
        _ => panic!("Expected JSON parsing error"),
    }
}

#[test]
fn test_config_validation() {
    use std::io::Write;
    use tempfile::NamedTempFile;
    
    // Create config with invalid values
    let mut temp_file = NamedTempFile::new().unwrap();
    writeln!(temp_file, r#"{{
        "norb": -1,
        "norb_core": 0,
        "nup": 1,
        "ndn": 1,
        "eps_var": 1e-4,
        "eps_pt_dtm": 1e-6,
        "target_uncertainty": 1e-4,
        "n_samples_per_batch": 1000,
        "n_batches": 100
    }}"#).unwrap();
    
    let result = RisqContext::from_files(temp_file.path(), "nonexistent");
    match result {
        Err(RisqError::InvalidConfig { message }) => {
            assert!(message.contains("norb must be positive"));
        },
        _ => panic!("Expected invalid config error"),
    }
}