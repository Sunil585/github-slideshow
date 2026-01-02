#!/usr/bin/env python3
"""
================================================================================
          12-6-4-3B FORCE FIELD FOR Zn2+ IN WATER - VERBOSE PRODUCTION
================================================================================

Fully verbose output showing every step of the simulation.

FORCE FIELD COMPONENTS:
  1. 12-6 LJ:  From prmtop (Li/Merz parameters for Zn2+)
               Zn: Rmin/2 = 1.455 Å, ε = 0.02662782 kcal/mol
               Combined Zn-O: Rmin = 3.223 Å, ε = 0.0636 kcal/mol

  2. Coulomb:  PME electrostatics (Q_Zn = +2, Q_O = -0.834)

  3. C4 term:  E = -C4/r^4 × fc(r)  [charge-induced dipole]
               Switch: fc(r) = 1 for r < 3.0 Å
                       fc(r) = 0.5*(1+cos(π(r-3.0)/1.2)) for 3.0-4.2 Å
                       fc(r) = 0 for r > 4.2 Å
               Li/Merz C4 = 231.6 Å⁴·kcal/mol (for 12-6-4 alone)
               For 12-6-4-3B: C4 ≈ 600-800 Å⁴·kcal/mol (to compensate 3B)

  4. 3B term:  E_3B = A*exp(-α(r12+r13))*G(θ) + B*(r12*r13)^(-n)*H(θ)
               Acts on Zn-O-O triplets within first shell

PROTOCOL:
  1. Minimization:      10,000 steps
  2. NVT Heating:       0 → 300 K in 100 ps
  3. NPT Equilibration: 100 ps at 300 K, 1 bar
  4. NVT Production:    500 ps at 300 K

Target: CN = 6.0, IOD = 2.08 Å (experimental Zn2+ hydration)

================================================================================
"""

import argparse
import math
import numpy as np
from datetime import datetime
import os
import sys
import time
try:
    from mdtraj.reporters import NetCDFReporter
    HAS_NETCDF = True
except ImportError:
    HAS_NETCDF = False
from openmm.app import DCDReporter
from openmm import unit, Platform, MonteCarloBarostat, Vec3
from openmm import CustomNonbondedForce, CustomManyParticleForce
from openmm.app import (AmberPrmtopFile, AmberInpcrdFile, Simulation,
                        PME, HBonds, PDBFile, StateDataReporter)
from openmm import LangevinMiddleIntegrator

# ==============================================================================
#                              CONSTANTS
# ==============================================================================

KCAL_TO_KJ = 4.184
KJ_TO_KCAL = 1.0 / 4.184
NM_TO_A = 10.0
A_TO_NM = 0.1
PI = 3.14159265358979323846

# C4 Switch function cutoffs
C4_RSWITCH_A = 3.0   # Å - switch starts
C4_RCUT_A = 4.2      # Å - cutoff
C4_RSWITCH_NM = C4_RSWITCH_A * A_TO_NM
C4_RCUT_NM = C4_RCUT_A * A_TO_NM

# 3B Parameters (OpenMM units: nm, kJ/mol) - FROM QM FITTING
PARAMS_3B = {
    'A': 159.534687,        # kJ/mol
    'alpha': 1.85422071,    # nm^-1
    'c1': 0.23456559,
    'c2': 0.11677340,
    'B': 1.81529439e-02,    # kJ*nm^(2n)/mol
    'n': 2.43039648,
    'zeta': 1.22502916,
}

# Experimental reference for Zn2+
EXPT = {'CN': 6.0, 'IOD': 2.08}

# Li/Merz 12-6-4 reference
LI_MERZ = {'C4': 231.6, 'CN': 6.0, 'IOD': 2.09}

# ==============================================================================
#                           LOGGING
# ==============================================================================

class Logger:
    def __init__(self, filename=None):
        self.terminal = sys.stdout
        self.log = open(filename, 'w') if filename else None

    def write(self, message):
        self.terminal.write(message)
        if self.log:
            self.log.write(message)
            self.log.flush()

    def flush(self):
        self.terminal.flush()
        if self.log:
            self.log.flush()

    def close(self):
        if self.log:
            self.log.close()

def print_banner(text, char='=', width=80):
    print("\n" + char * width)
    print(f" {text}")
    print(char * width)

def print_section(text):
    print(f"\n  {text}")
    print(f"  {'-' * len(text)}")

# ==============================================================================
#                           HELPER FUNCTIONS
# ==============================================================================

def is_zn(atom):
    """Check if atom is zinc."""
    if hasattr(atom, 'element') and atom.element is not None:
        return atom.element.symbol.upper() == "ZN"
    return atom.name.upper().startswith("ZN") if hasattr(atom, 'name') else False

def is_water_oxygen(atom):
    """Check if atom is water oxygen."""
    if not (hasattr(atom, 'element') and atom.element and atom.element.symbol.upper() == "O"):
        return False
    if hasattr(atom, 'residue'):
        return atom.residue.name.upper() in ("WAT", "HOH", "TIP3", "SOL", "TP3", "TIP3P", "OPC", "OPC3")
    return False

def _minimum_image(delta, box):
    """Apply minimum-image convention for triclinic boxes.

    Parameters:
      delta: vector in Å
      box: 3x3 box matrix in Å (rows are box vectors)
    """
    inv_box = np.linalg.inv(box)
    fractional = np.dot(inv_box, delta)
    fractional -= np.round(fractional)
    return np.dot(box, fractional)

def compute_structure(positions, zn_idx, water_o_indices, box_vectors, cn_cutoff=3.0):
    """
    Compute CN, IOD, distances, and O-Zn-O angles.

    CN is computed with FIXED 3.0 Å cutoff (standard for Zn2+).
    """
    zn_pos = np.array([positions[zn_idx].x, positions[zn_idx].y, positions[zn_idx].z]) * NM_TO_A
    box = np.array([
        [box_vectors[0].x, box_vectors[0].y, box_vectors[0].z],
        [box_vectors[1].x, box_vectors[1].y, box_vectors[1].z],
        [box_vectors[2].x, box_vectors[2].y, box_vectors[2].z],
    ]) * NM_TO_A

    first_shell_r = []
    first_shell_vec = []
    all_distances = []

    for o_idx in water_o_indices:
        o_pos = np.array([positions[o_idx].x, positions[o_idx].y, positions[o_idx].z]) * NM_TO_A
        delta = o_pos - zn_pos
        delta = _minimum_image(delta, box)  # PBC
        r = np.sqrt(np.sum(delta**2))
        all_distances.append(r)
        if r < cn_cutoff:
            first_shell_r.append(r)
            first_shell_vec.append(delta)

    cn = len(first_shell_r)
    iod = np.mean(first_shell_r) if first_shell_r else 0.0

    # O-Zn-O angles
    angles = []
    for i in range(len(first_shell_vec)):
        for j in range(i + 1, len(first_shell_vec)):
            v1, v2 = first_shell_vec[i], first_shell_vec[j]
            r1, r2 = first_shell_r[i], first_shell_r[j]
            cos_theta = np.clip(np.dot(v1, v2) / (r1 * r2), -1.0, 1.0)
            angles.append(np.degrees(np.arccos(cos_theta)))

    return cn, iod, first_shell_r, all_distances, angles

# ==============================================================================
#                         FORCE FIELD SETUP
# ==============================================================================

def add_c4_force(system, topology, C4_kcal_A4):
    """
    Add C4 term: E = -C4/r^4 × fc(r)

    This represents the charge-induced dipole interaction.

    Switch function fc(r):
      fc(r) = 1                                    for r < 3.0 Å
      fc(r) = 0.5*(1 + cos(π*(r-3.0)/(4.2-3.0)))  for 3.0 ≤ r ≤ 4.2 Å
      fc(r) = 0                                    for r > 4.2 Å

    Parameters:
      C4_kcal_A4: C4 coefficient in kcal/mol·Å⁴

    Returns:
      C4 in OpenMM units (kJ/mol·nm⁴)
    """
    # Unit conversion: kcal/mol·Å⁴ → kJ/mol·nm⁴
    C4_kJ_nm4 = C4_kcal_A4 * KCAL_TO_KJ * (A_TO_NM)**4

    # Energy expression with switch function
    expr = (
        f"-C4 * (isZn1*isO2 + isO1*isZn2) / (r^4) * fc;"
        f"fc = step(rswitch - r) + step(r - rswitch)*step(rcut - r)*"
        f"0.5*(1 + cos({PI}*(r - rswitch)/(rcut - rswitch)))"
    )

    force = CustomNonbondedForce(expr)
    force.addGlobalParameter("C4", C4_kJ_nm4)
    force.addGlobalParameter("rswitch", C4_RSWITCH_NM)
    force.addGlobalParameter("rcut", C4_RCUT_NM)
    force.addPerParticleParameter("isZn")
    force.addPerParticleParameter("isO")
    force.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
    force.setCutoffDistance(C4_RCUT_NM * unit.nanometer)

    # Assign particle parameters
    n_zn, n_o = 0, 0
    for atom in topology.atoms():
        if is_zn(atom):
            force.addParticle([1.0, 0.0])
            n_zn += 1
        elif is_water_oxygen(atom):
            force.addParticle([0.0, 1.0])
            n_o += 1
        else:
            force.addParticle([0.0, 0.0])

    # Copy exclusions from NonbondedForce
    for f in system.getForces():
        if f.__class__.__name__ == "NonbondedForce":
            for i in range(f.getNumExceptions()):
                p1, p2, _, _, _ = f.getExceptionParameters(i)
                force.addExclusion(int(p1), int(p2))
            break

    force.setForceGroup(1)
    system.addForce(force)

    return C4_kJ_nm4, n_zn, n_o


def add_3b_force(system, topology, cutoff_A=3.5):
    """
    Add 3-body force for Zn-O-O triplets.

    E_3B = A*exp(-α*(r_ZnO1 + r_ZnO2))*G(θ) + B*(r_ZnO1*r_ZnO2)^(-n)*H(θ)

    Where:
      G(θ) = 1 + c1*cos(θ) + c2*cos(2θ)
      H(θ) = 1 + ζ*cos(θ)
      θ = O-Zn-O angle

    Parameters are from QM fitting.
    """
    if cutoff_A <= 0:
        return 0, 0  # Disable 3B

    cutoff_nm = cutoff_A * A_TO_NM

    energy_expr = (
        "A * exp(-alpha*(r12 + r13)) * G_theta + "
        "B * (r12*r13)^(-n_exp) * H_theta;"
        "G_theta = 1 + c1*cos(theta) + c2*cos(2*theta);"
        "H_theta = 1 + zeta*cos(theta);"
        "r12 = distance(p1, p2);"
        "r13 = distance(p1, p3);"
        "theta = angle(p2, p1, p3)"
    )

    force = CustomManyParticleForce(3, energy_expr)

    # Add parameters
    force.addGlobalParameter("A", PARAMS_3B['A'])
    force.addGlobalParameter("alpha", PARAMS_3B['alpha'])
    force.addGlobalParameter("c1", PARAMS_3B['c1'])
    force.addGlobalParameter("c2", PARAMS_3B['c2'])
    force.addGlobalParameter("B", PARAMS_3B['B'])
    force.addGlobalParameter("n_exp", PARAMS_3B['n'])
    force.addGlobalParameter("zeta", PARAMS_3B['zeta'])

    # Particle types: 0=Zn (central), 1=O, 2=other
    force.setPermutationMode(CustomManyParticleForce.UniqueCentralParticle)
    force.setTypeFilter(0, {0})  # p1 = Zn only
    force.setTypeFilter(1, {1})  # p2 = O only
    force.setTypeFilter(2, {1})  # p3 = O only

    n_zn, n_o = 0, 0
    for atom in topology.atoms():
        if is_zn(atom):
            force.addParticle([], 0)
            n_zn += 1
        elif is_water_oxygen(atom):
            force.addParticle([], 1)
            n_o += 1
        else:
            force.addParticle([], 2)

    force.setNonbondedMethod(CustomManyParticleForce.CutoffPeriodic)
    force.setCutoffDistance(cutoff_nm * unit.nanometer)
    force.setForceGroup(2)
    system.addForce(force)

    return n_zn, n_o

# ==============================================================================
#                              MAIN SIMULATION
# ==============================================================================

def run_simulation(args):
    """Run full MD simulation with verbose output."""

    start_time = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logging
    log_file = os.path.join(args.output_dir, 'simulation.log')
    logger = Logger(log_file)
    sys.stdout = logger

    # ==========================================================================
    #                              HEADER
    # ==========================================================================
    print()
    print("=" * 80)
    print("          12-6-4-3B FORCE FIELD FOR Zn2+ IN WATER")
    print("                    VERBOSE PRODUCTION RUN")
    print("=" * 80)
    print()
    print(f"  Run started:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Topology:         {args.prmtop}")
    print(f"  Coordinates:      {args.inpcrd}")
    print(f"  Output:           {args.output_dir}/")
    print()

    print_banner("FORCE FIELD PARAMETERS")
    print()
    print("  12-6 LJ (from prmtop, Li/Merz 2014):")
    print("    Zn2+: Rmin/2 = 1.455 Å, ε = 0.02662782 kcal/mol")
    print("    O:    Rmin/2 = 1.768 Å, ε = 0.1520 kcal/mol")
    print("    Zn-O: Rmin   = 3.223 Å, ε = 0.0636 kcal/mol (Lorentz-Berthelot)")
    print()
    print("  C4 term (charge-induced dipole):")
    print(f"    C4 = {args.C4} kcal/mol·Å⁴")
    print(f"    Switch: {C4_RSWITCH_A} → {C4_RCUT_A} Å")
    print(f"    (Li/Merz 12-6-4 uses C4 = {LI_MERZ['C4']} for CN={LI_MERZ['CN']}, IOD={LI_MERZ['IOD']} Å)")
    print()
    print("  3B term (angular preference):")
    if args.cutoff_3b > 0:
        print(f"    Cutoff: {args.cutoff_3b} Å")
        print(f"    A     = {PARAMS_3B['A']:.6f} kJ/mol")
        print(f"    α     = {PARAMS_3B['alpha']:.6f} nm⁻¹")
        print(f"    c1    = {PARAMS_3B['c1']:.6f}")
        print(f"    c2    = {PARAMS_3B['c2']:.6f}")
        print(f"    B     = {PARAMS_3B['B']:.6e} kJ·nm^(2n)/mol")
        print(f"    n     = {PARAMS_3B['n']:.6f}")
        print(f"    ζ     = {PARAMS_3B['zeta']:.6f}")
    else:
        print("    DISABLED (cutoff_3b = 0)")
    print()
    print("  Target (experimental Zn2+):")
    print(f"    CN  = {EXPT['CN']}")
    print(f"    IOD = {EXPT['IOD']} Å")
    print()

    print_banner("SIMULATION PROTOCOL")
    print()
    print(f"  1. Minimization:      {args.min_steps} steps")
    print(f"  2. NVT Heating:       0 → 300 K in {args.heat_time} ps")
    print(f"  3. NPT Equilibration: {args.npt_time} ps at 300 K, 1 bar")
    print(f"  4. NVT Production:    {args.prod_time} ps at 300 K")
    print(f"  Timestep:             {args.timestep} fs")
    print(f"  Save interval:        {args.save_interval} ps")
    print()

    # ==========================================================================
    #                         1. LOAD SYSTEM
    # ==========================================================================
    print_banner("1. LOADING SYSTEM")

    prmtop = AmberPrmtopFile(args.prmtop)
    inpcrd = AmberInpcrdFile(args.inpcrd)

    atoms = list(prmtop.topology.atoms())
    zn_indices = [i for i, a in enumerate(atoms) if is_zn(a)]
    water_o_indices = [i for i, a in enumerate(atoms) if is_water_oxygen(a)]

    if len(zn_indices) != 1:
        raise ValueError(f"Expected exactly 1 Zn atom, found {len(zn_indices)}")

    print()
    print(f"  Total atoms:        {len(atoms)}")
    print(f"  Zn atoms:           {len(zn_indices)} (index: {zn_indices})")
    print(f"  Water O atoms:      {len(water_o_indices)}")
    print(f"  Water molecules:    {len(water_o_indices)}")

    bv = inpcrd.boxVectors
    box_a = bv[0].x * NM_TO_A
    box_b = bv[1].y * NM_TO_A
    box_c = bv[2].z * NM_TO_A
    box_vol = box_a * box_b * box_c

    print()
    print("  Box dimensions:")
    print(f"    a = {box_a:.4f} Å")
    print(f"    b = {box_b:.4f} Å")
    print(f"    c = {box_c:.4f} Å")
    print(f"    V = {box_vol:.2f} Å³")

    # ==========================================================================
    #                       2. CREATE SYSTEM
    # ==========================================================================
    print_banner("2. CREATING OPENMM SYSTEM")

    print()
    print("  Creating base system with PME electrostatics...")

    system = prmtop.createSystem(
        nonbondedMethod=PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=HBonds
    )
    system.setDefaultPeriodicBoxVectors(*inpcrd.boxVectors)

    print()
    print("  Base forces from prmtop:")
    for i, f in enumerate(system.getForces()):
        print(f"    [{i}] {f.__class__.__name__}")

    # Add C4 force
    print_section("Adding C4 Force")
    C4_converted, n_zn_c4, n_o_c4 = add_c4_force(system, prmtop.topology, args.C4)
    print()
    print(f"  C4 = {args.C4} kcal/mol·Å⁴")
    print(f"     = {C4_converted:.6e} kJ/mol·nm⁴ (OpenMM units)")
    print(f"  Switch: rswitch = {C4_RSWITCH_A} Å, rcut = {C4_RCUT_A} Å")
    print(f"  Particles: {n_zn_c4} Zn, {n_o_c4} O")
    print(f"  ForceGroup: 1")

    # Expected E_C4 at target geometry
    E_C4_expected = 6 * (-args.C4 / EXPT['IOD']**4)
    print(f"  Expected E_C4 (CN=6, IOD=2.08 Å): {E_C4_expected:.2f} kcal/mol")

    # Add 3B force
    print_section("Adding 3-Body Force")
    if args.cutoff_3b > 0:
        n_zn_3b, n_o_3b = add_3b_force(system, prmtop.topology, args.cutoff_3b)
        print()
        print(f"  Cutoff: {args.cutoff_3b} Å")
        print(f"  Mode: UniqueCentralParticle (Zn as center)")
        print(f"  Particles: {n_zn_3b} Zn (type 0), {n_o_3b} O (type 1)")
        print(f"  ForceGroup: 2")
    else:
        print()
        print("  3B force DISABLED (cutoff_3b = 0)")

    # ==========================================================================
    #                      3. CREATE SIMULATION
    # ==========================================================================
    print_banner("3. CREATING SIMULATION")

    timestep_ps = args.timestep * 0.001
    integrator = LangevinMiddleIntegrator(
        300 * unit.kelvin,
        2.0 / unit.picosecond,
        timestep_ps * unit.picoseconds
    )

    print()
    print(f"  Integrator: LangevinMiddleIntegrator")
    print(f"    Temperature: 300 K")
    print(f"    Friction:    2.0 ps⁻¹")
    print(f"    Timestep:    {args.timestep} fs")

    # Select platform
    try:
        platform = Platform.getPlatformByName(args.platform)
        if args.platform == "CUDA":
            properties = {"DeviceIndex": args.device, "Precision": "mixed"}
        else:
            properties = {}
        print(f"  Platform: {args.platform}")
        if args.platform == "CUDA":
            print(f"    Device: {args.device}")
            print(f"    Precision: mixed")
    except Exception:
        platform = Platform.getPlatformByName("CPU")
        properties = {}
        print(f"  Platform: CPU (fallback)")

    sim = Simulation(prmtop.topology, system, integrator, platform, properties)
    sim.context.setPositions(inpcrd.positions)
    sim.context.setPeriodicBoxVectors(*inpcrd.boxVectors)

    # Helper functions
    def get_energies():
        """Get energies from each force group."""
        state_c4 = sim.context.getState(getEnergy=True, groups={1})
        E_c4 = state_c4.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) * KJ_TO_KCAL

        if args.cutoff_3b > 0:
            state_3b = sim.context.getState(getEnergy=True, groups={2})
            E_3b = state_3b.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) * KJ_TO_KCAL
        else:
            E_3b = 0.0

        state_tot = sim.context.getState(getEnergy=True)
        E_tot = state_tot.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) * KJ_TO_KCAL

        return E_c4, E_3b, E_tot

    def get_structure():
        """Get structural properties."""
        state = sim.context.getState(getPositions=True)
        pos = state.getPositions()
        bv = state.getPeriodicBoxVectors()
        return compute_structure(pos, zn_indices[0], water_o_indices, bv, cn_cutoff=3.0)

    def get_temperature():
        """Get instantaneous temperature."""
        state = sim.context.getState(getEnergy=True)
        ke = state.getKineticEnergy()
        dof = 3 * len(atoms) - 3 - system.getNumConstraints()
        T = 2 * ke / (dof * unit.MOLAR_GAS_CONSTANT_R)
        return T.value_in_unit(unit.kelvin)

    # ==========================================================================
    #                       4. MINIMIZATION
    # ==========================================================================
    print_banner("4. ENERGY MINIMIZATION")

    cn, iod, _, _, _ = get_structure()
    E_c4, E_3b, E_tot = get_energies()

    print()
    print("  Initial state:")
    print(f"    CN        = {cn}")
    print(f"    IOD       = {iod:.4f} Å")
    print(f"    E_total   = {E_tot:.2f} kcal/mol")
    print(f"    E_C4      = {E_c4:.2f} kcal/mol")
    print(f"    E_3B      = {E_3b:.2f} kcal/mol")

    print()
    print(f"  Running {args.min_steps} minimization steps...")
    print()
    print(f"  {'Step':<10} {'E_total':<14} {'E_C4':<12} {'E_3B':<12} {'CN':<6} {'IOD(Å)':<10}")
    print(f"  {'-' * 68}")

    steps_per_report = max(args.min_steps // 10, 1)
    for i in range(10):
        sim.minimizeEnergy(maxIterations=steps_per_report)
        cn, iod, _, _, _ = get_structure()
        E_c4, E_3b, E_tot = get_energies()
        step = (i + 1) * steps_per_report
        print(f"  {step:<10} {E_tot:<14.2f} {E_c4:<12.2f} {E_3b:<12.2f} {cn:<6} {iod:<10.4f}")

    print()
    print("  Minimization complete.")

    # ==========================================================================
    #                       5. NVT HEATING
    # ==========================================================================
    print_banner("5. NVT HEATING (0 → 300 K)")

    heat_steps = max(int(args.heat_time / timestep_ps), 1)
    report_interval = max(int(1.0 / timestep_ps), 1)  # Every 1 ps

    print()
    print(f"  Duration: {args.heat_time} ps")
    print(f"  Steps: {heat_steps}")
    print(f"  Reporting every 1 ps")
    print()
    print(f"  {'Step':<10} {'Time(ps)':<10} {'T(K)':<8} {'CN':<6} {'IOD(Å)':<10} {'E_C4':<12} {'E_3B':<12}")
    print(f"  {'-' * 76}")

    # Initialize velocities at 0 K
    sim.context.setVelocitiesToTemperature(1 * unit.kelvin)

    remaining = heat_steps
    completed = 0
    while remaining > 0:
        steps = min(report_interval, remaining)
        completed += steps
        frac = completed / heat_steps
        T_target = 300.0 * frac
        integrator.setTemperature(T_target * unit.kelvin)

        sim.step(steps)

        cn, iod, _, _, _ = get_structure()
        E_c4, E_3b, _ = get_energies()
        T_inst = get_temperature()
        time_ps = completed * timestep_ps

        print(f"  {completed:<10} {time_ps:<10.1f} {T_inst:<8.1f} {cn:<6} {iod:<10.4f} {E_c4:<12.2f} {E_3b:<12.2f}")
        remaining -= steps

    print()
    print("  Heating complete. Final T = 300 K")

    # ==========================================================================
    #                     6. NPT EQUILIBRATION
    # ==========================================================================
    print_banner("6. NPT EQUILIBRATION")

    print()
    print(f"  Duration: {args.npt_time} ps at 300 K, 1 bar")
    print("  Adding Monte Carlo barostat...")

    barostat = MonteCarloBarostat(1.0 * unit.bar, 300 * unit.kelvin, 25)
    system.addForce(barostat)
    sim.context.reinitialize(preserveState=True)

    npt_steps = max(int(args.npt_time / timestep_ps), 1)
    report_interval = max(int(5.0 / timestep_ps), 1)  # Every 5 ps

    print()
    print(f"  {'Step':<10} {'Time(ps)':<10} {'CN':<6} {'IOD(Å)':<10} {'E_C4':<12} {'E_3B':<12} {'Box(Å)':<10}")
    print(f"  {'-' * 78}")

    remaining = npt_steps
    completed = 0
    while remaining > 0:
        steps = min(report_interval, remaining)
        sim.step(steps)
        completed += steps

        cn, iod, _, _, _ = get_structure()
        E_c4, E_3b, _ = get_energies()
        state = sim.context.getState(getPositions=True)
        bv = state.getPeriodicBoxVectors()
        box_len = bv[0].x * NM_TO_A

        time_ps = completed * timestep_ps

        print(f"  {completed:<10} {time_ps:<10.1f} {cn:<6} {iod:<10.4f} {E_c4:<12.2f} {E_3b:<12.2f} {box_len:<10.4f}")
        remaining -= steps

    # Get final box
    state = sim.context.getState(getPositions=True)
    final_bv = state.getPeriodicBoxVectors()
    final_box_vol = final_bv[0].x * final_bv[1].y * final_bv[2].z * (NM_TO_A)**3

    # Remove barostat for NVT production
    for i in range(system.getNumForces() - 1, -1, -1):
        if isinstance(system.getForce(i), MonteCarloBarostat):
            system.removeForce(i)
            break
    sim.context.reinitialize(preserveState=True)

    print()
    print(f"  NPT complete. Final box volume: {final_box_vol:.2f} Å³")

    # ==========================================================================
    #                       7. NVT PRODUCTION
    # ==========================================================================
    print_banner("7. NVT PRODUCTION")

    prod_steps = int(args.prod_time / timestep_ps)
    if prod_steps <= 0:
        raise ValueError("Production time too short for the given timestep.")

    save_steps = max(int(args.save_interval / timestep_ps), 1)
    if save_steps > prod_steps:
        save_steps = prod_steps

    n_frames = math.ceil(prod_steps / save_steps)

    print()
    print(f"  Duration: {args.prod_time} ps ({args.prod_time / 1000:.2f} ns) at 300 K")
    print(f"  Save interval: {args.save_interval} ps")
    print(f"  Total frames: {n_frames}")

    # Data storage
    cn_list, iod_list = [], []
    e_c4_list, e_3b_list = [], []
    all_distances = []
    all_angles = []
    first_shell_distances = []

    # Report interval
    report_freq = max(n_frames // 50, 1)

    print()
    print(f"  {'Frame':<8} {'Time(ps)':<12} {'CN':<6} {'IOD(Å)':<10} {'E_C4':<12} {'E_3B':<12}")
    print(f"  {'-' * 64}")
    # Add trajectory reporter
    if HAS_NETCDF:
        traj_file = os.path.join(args.output_dir, 'trajectory.nc')
        sim.reporters.append(NetCDFReporter(traj_file, save_steps))
    else:
        traj_file = os.path.join(args.output_dir, 'trajectory.dcd')
        sim.reporters.append(DCDReporter(traj_file, save_steps))

    remaining = prod_steps
    completed = 0
    frame = 0
    while remaining > 0:
        steps = min(save_steps, remaining)
        sim.step(steps)
        completed += steps
        frame += 1

        cn, iod, fs_r, all_r, angles = get_structure()
        E_c4, E_3b, _ = get_energies()

        cn_list.append(cn)
        iod_list.append(iod)
        e_c4_list.append(E_c4)
        e_3b_list.append(E_3b)
        all_distances.extend(all_r)
        all_angles.extend(angles)
        first_shell_distances.extend(fs_r)

        if frame % report_freq == 0 or frame == 1:
            time_ps = completed * timestep_ps
            print(f"  {frame:<8} {time_ps:<12.1f} {cn:<6} {iod:<10.4f} {E_c4:<12.2f} {E_3b:<12.2f}")

        remaining -= steps

    print()
    print(f"  Production complete. {frame} frames collected.")

    # ==========================================================================
    #                         8. ANALYSIS
    # ==========================================================================
    print_banner("8. ANALYSIS")

    cn_arr = np.array(cn_list)
    iod_arr = np.array(iod_list)
    e_c4_arr = np.array(e_c4_list)
    e_3b_arr = np.array(e_3b_list)

    print_section("Statistics")
    print()
    print(f"  {'Property':<15} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print(f"  {'-' * 63}")
    print(f"  {'CN':<15} {cn_arr.mean():<12.3f} {cn_arr.std():<12.3f} {cn_arr.min():<12} {cn_arr.max():<12}")
    print(f"  {'IOD (Å)':<15} {iod_arr.mean():<12.4f} {iod_arr.std():<12.4f} {iod_arr.min():<12.4f} {iod_arr.max():<12.4f}")
    print(f"  {'E_C4 (kcal)':<15} {e_c4_arr.mean():<12.2f} {e_c4_arr.std():<12.2f} {e_c4_arr.min():<12.2f} {e_c4_arr.max():<12.2f}")
    print(f"  {'E_3B (kcal)':<15} {e_3b_arr.mean():<12.2f} {e_3b_arr.std():<12.2f} {e_3b_arr.min():<12.2f} {e_3b_arr.max():<12.2f}")

    print_section("CN Distribution")
    print()
    unique_cn, counts = np.unique(cn_arr, return_counts=True)
    print(f"  {'CN':<6} {'Count':<10} {'%':<10} {'Bar'}")
    print(f"  {'-' * 50}")
    for c, n in zip(unique_cn, counts):
        pct = 100 * n / len(cn_arr)
        bar = '█' * int(pct / 2)
        print(f"  {c:<6} {n:<10} {pct:<10.2f} {bar}")

    print_section("Comparison to Experiment")
    print()
    cn_err = abs(cn_arr.mean() - EXPT['CN'])
    iod_err = abs(iod_arr.mean() - EXPT['IOD'])

    print(f"  {'Property':<10} {'Simulation':<15} {'Experiment':<15} {'Error':<10} {'Status'}")
    print(f"  {'-' * 60}")
    status_cn = '✓' if cn_err < 0.5 else '✗'
    status_iod = '✓' if iod_err < 0.1 else '✗'
    print(f"  {'CN':<10} {cn_arr.mean():<15.2f} {EXPT['CN']:<15.1f} {cn_err:<10.2f} {status_cn}")
    print(f"  {'IOD (Å)':<10} {iod_arr.mean():<15.3f} {EXPT['IOD']:<15.2f} {iod_err:<10.3f} {status_iod}")

    # ==========================================================================
    #                        9. SAVE OUTPUT
    # ==========================================================================
    print_banner("9. SAVING OUTPUT")

    # Timeseries
    ts_file = os.path.join(args.output_dir, 'timeseries.dat')
    time_arr = np.arange(1, len(cn_list) + 1) * args.save_interval
    np.savetxt(ts_file, np.column_stack([time_arr, cn_list, iod_list, e_c4_list, e_3b_list]),
               header='Time(ps) CN IOD(A) E_C4(kcal/mol) E_3B(kcal/mol)',
               fmt=['%.1f', '%d', '%.4f', '%.2f', '%.2f'])
    print(f"\n  Timeseries: {ts_file}")

    # ==========================================================================
    #                         FINAL SUMMARY
    # ==========================================================================
    elapsed = time.time() - start_time

    print_banner("SIMULATION COMPLETE")
    print()
    print(f"  Total time: {elapsed / 60:.1f} min ({elapsed:.1f} s)")
    print()
    print("  FINAL RESULTS:")
    print(f"    CN  = {cn_arr.mean():.2f} ± {cn_arr.std():.2f}  (Expt: {EXPT['CN']})")
    print(f"    IOD = {iod_arr.mean():.3f} ± {iod_arr.std():.3f} Å  (Expt: {EXPT['IOD']} Å)")
    print(f"    E_C4 = {e_c4_arr.mean():.2f} ± {e_c4_arr.std():.2f} kcal/mol")
    print(f"    E_3B = {e_3b_arr.mean():.2f} ± {e_3b_arr.std():.2f} kcal/mol")
    print()
    print(f"  Parameters used:")
    print(f"    C4 = {args.C4} kcal/mol·Å⁴")
    print(f"    3B cutoff = {args.cutoff_3b} Å")
    print()
    print(f"  Output: {args.output_dir}/")
    print()
    print("=" * 80)

    logger.close()
    sys.stdout = logger.terminal


def main():
    parser = argparse.ArgumentParser(
        description="12-6-4-3B Force Field - Verbose Production Run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python zn_1264_3b_verbose.py --C4 700
  python zn_1264_3b_verbose.py --C4 231.6 --cutoff_3b 0  # Li/Merz 12-6-4 only
  python zn_1264_3b_verbose.py --C4 800 --prod_time 1000
        """
    )

    # Input files
    parser.add_argument("--prmtop", default="Zn_tip3p_1264.prmtop",
                        help="AMBER topology file")
    parser.add_argument("--inpcrd", default="Zn_tip3p_1264.inpcrd",
                        help="AMBER coordinate file")

    # Force field parameters
    parser.add_argument("--C4", type=float, default=700.0,
                        help="C4 coefficient in kcal/mol·Å⁴ (default: 700)")
    parser.add_argument("--cutoff_3b", type=float, default=3.5,
                        help="3B cutoff in Å (default: 3.5, use 0 to disable)")

    # Simulation parameters
    parser.add_argument("--min_steps", type=int, default=10000,
                        help="Minimization steps (default: 10000)")
    parser.add_argument("--heat_time", type=float, default=100.0,
                        help="NVT heating time in ps (default: 100)")
    parser.add_argument("--npt_time", type=float, default=100.0,
                        help="NPT equilibration time in ps (default: 100)")
    parser.add_argument("--prod_time", type=float, default=500.0,
                        help="Production time in ps (default: 500)")
    parser.add_argument("--timestep", type=float, default=2.0,
                        help="Timestep in fs (default: 2.0)")
    parser.add_argument("--save_interval", type=float, default=1.0,
                        help="Save interval in ps (default: 1.0)")

    # Platform
    parser.add_argument("--platform", default="CUDA",
                        help="OpenMM platform (default: CUDA)")
    parser.add_argument("--device", default="0",
                        help="CUDA device index (default: 0)")

    # Output
    parser.add_argument("--output_dir", default="output_verbose",
                        help="Output directory (default: output_verbose)")

    args = parser.parse_args()
    run_simulation(args)


if __name__ == "__main__":
    main()
