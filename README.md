# Rust Implementation of Semistochastic Quantum chemistry (RISQ)
[![Build Status](https://app.travis-ci.com/aaholmes/risq.svg?token=bBbFNau8Nd1xwi7f2Ehx&branch=main)](https://app.travis-ci.com/aaholmes/risq)

Rust implementation of the following electronic structure algorithms:
1. Heat-bath Configuration Interaction (HCI) - an efficient, deterministic, variational active space solver
2. Semistochastic multireference perturbation theories:
   1. Strongly-Contracted NEVPT2 (WIP)
   2. Epstein-Nesbet (often referred to as Semistochastic HCI (SHCI) when applied to an HCI reference)
3. Dynamic Semistochastic Full CI Quantum Monte Carlo (DS-FCIQMC) - a semistochastic projector method with dynamic division between deterministic and stochastic components
