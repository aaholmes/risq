//! # Error Handling (`error`)
//!
//! Comprehensive error type for all risq operations, replacing panic-prone
//! `.unwrap()` calls with proper error propagation.

use thiserror::Error;

/// Comprehensive error type for all risq operations
#[derive(Error, Debug)]
pub enum RisqError {
    #[error("IO error reading file '{path}': {source}")]
    Io { 
        path: String, 
        #[source] source: std::io::Error 
    },
    
    #[error("JSON parsing error in '{path}': {source}")]
    Json { 
        path: String, 
        #[source] source: serde_json::Error 
    },
    
    #[error("Invalid configuration: {message}")]
    InvalidConfig { message: String },
    
    #[error("FCIDUMP parsing error: {message} at line {line}")]
    FcidumpParse { message: String, line: usize },
    
    #[error("Davidson diagonalization failed after {iterations} iterations: {reason}")]
    DavidsonFailure { iterations: usize, reason: String },
    
    #[error("Convergence failed: {algorithm} after {iterations} iterations (threshold: {threshold})")]
    ConvergenceFailure { 
        algorithm: String, 
        iterations: usize, 
        threshold: f64 
    },
    
    #[error("Invalid determinant configuration: {details}")]
    InvalidDeterminant { details: String },
    
    #[error("Numerical instability in {operation}: {details}")]
    NumericalError { operation: String, details: String },
    
    #[error("Memory allocation failed: requested {requested_bytes} bytes")]
    MemoryError { requested_bytes: usize },
    
    #[error("Index out of bounds: {index} >= {size} in {operation}")]
    IndexError { index: usize, size: usize, operation: String },
    
    #[error("Feature not implemented: {feature}")]
    NotImplemented { feature: String },
    
    #[error("Invalid matrix dimensions: expected {expected}, got {actual}")]
    MatrixDimensionError { expected: String, actual: String },
    
    #[error("Random number generation error: {details}")]
    RandomError { details: String },
    
    #[error("Wavefunction error: {details}")]
    WavefunctionError { details: String },
    
    #[error("Hamiltonian matrix element error: {details}")]
    HamiltonianError { details: String },
    
    #[error("Excitation generation error: {details}")]
    ExcitationError { details: String },
}

/// Convenience result type
pub type RisqResult<T> = Result<T, RisqError>;

/// Helper macros for common error patterns
#[macro_export]
macro_rules! risq_bail {
    ($variant:ident { $($field:ident: $value:expr),* }) => {
        return Err(crate::error::RisqError::$variant { $($field: $value),* })
    };
}

#[macro_export]
macro_rules! risq_ensure {
    ($cond:expr, $variant:ident { $($field:ident: $value:expr),* }) => {
        if !($cond) {
            crate::risq_bail!($variant { $($field: $value),* });
        }
    };
}

impl RisqError {
    /// Create an IO error with context
    pub fn io_error<P: AsRef<std::path::Path>>(path: P, source: std::io::Error) -> Self {
        Self::Io {
            path: path.as_ref().to_string_lossy().to_string(),
            source,
        }
    }
    
    /// Create a JSON parsing error with context
    pub fn json_error<P: AsRef<std::path::Path>>(path: P, source: serde_json::Error) -> Self {
        Self::Json {
            path: path.as_ref().to_string_lossy().to_string(),
            source,
        }
    }
    
    /// Create an invalid config error
    pub fn invalid_config<S: Into<String>>(message: S) -> Self {
        Self::InvalidConfig {
            message: message.into(),
        }
    }
}