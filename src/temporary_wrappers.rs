//! Temporary wrapper functions to interface between the new context-based
//! architecture and the existing function signatures during refactoring.
//! TODO: Remove this module once all functions are updated.

use crate::context::RisqContext;
use crate::error::RisqResult;
use crate::utils::read_input::Global;
use crate::wf::{init_var_wf as lib_init_var_wf, VarWf};
use crate::var::variational as lib_variational;
use crate::pt::perturbative as lib_perturbative;

/// Temporary wrapper for init_var_wf
pub fn init_var_wf(context: &RisqContext) -> RisqResult<VarWf> {
    let global = context.as_global();
    Ok(lib_init_var_wf(&global, &context.hamiltonian, &context.excitation_generator))
}

/// Temporary wrapper for variational
pub fn variational(context: &mut RisqContext, wf: &mut VarWf) -> RisqResult<f64> {
    let global = context.as_global();
    lib_variational(&global, &context.hamiltonian, &context.excitation_generator, wf);
    Ok(wf.wf.energy)
}

/// Temporary wrapper for perturbative
pub fn perturbative(context: &RisqContext, wf: &crate::wf::Wf) -> RisqResult<f64> {
    let global = context.as_global();
    lib_perturbative(&global, &context.hamiltonian, &context.excitation_generator, wf);
    // TODO: Return actual PT2 energy
    Ok(0.0)
}