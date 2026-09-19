import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

# Define 20 target orbits: (SMA, ECC, INC, RAAN, AOP)
target_orbits = [
    (7078.14, 0.0001, 56.2, 0.0, 0.0),
    (7078.14, 0.0001, 65.0, 0.0, 0.0),
    (7078.14, 0.0001, 82.0, 0.0, 0.0),
    (7078.14, 0.0001, 90.0, 0.0, 0.0),
    (7078.14, 0.0001, 98.0, 0.0, 0.0),
    (7178.14, 0.0001, 56.2, 0.0, 0.0),
    (7178.14, 0.0001, 65.0, 0.0, 0.0),
    (7178.14, 0.0001, 82.0, 0.0, 0.0),
    (7178.14, 0.0001, 90.0, 0.0, 0.0),
    (7178.14, 0.0001, 98.0, 0.0, 0.0),
    (7278.14, 0.0001, 56.2, 0.0, 0.0),
    (7278.14, 0.0001, 65.0, 0.0, 0.0),
    (7278.14, 0.0001, 82.0, 0.0, 0.0),
    (7278.14, 0.0001, 90.0, 0.0, 0.0),
    (7278.14, 0.0001, 98.0, 0.0, 0.0),
    (7378.14, 0.0001, 56.2, 0.0, 0.0),
    (7378.14, 0.0001, 65.0, 0.0, 0.0),
    (7378.14, 0.0001, 82.0, 0.0, 0.0),
    (7378.14, 0.0001, 90.0, 0.0, 0.0),
    (7378.14, 0.0001, 98.0, 0.0, 0.0),
]

script_dir = Path(__file__).parent.resolve()
results_base = script_dir / "results"

master_dir = Path.home() / "Downloads" / "MASTER_8.2.3_windows"
input_inp_path = master_dir / "input" / "master.inp"
exe_path = master_dir / "master-win64.exe"


def verify_paths():
  """Checks that critical directories and executables exist before running."""
  print("--- [DEBUG] Pre-flight Path Verification ---")
  print(f"Script Workspace Directory : {script_dir}")
  print(f"MASTER Working Directory   : {master_dir}")
  print(f"MASTER Executable Path     : {exe_path} (Exists: {exe_path.exists()})")
  print(
      f"MASTER Input Template Path : {input_inp_path} (Exists:"
      f" {input_inp_path.exists()})\n"
  )
  if not exe_path.exists():
    print("[FATAL] master-win64.exe not found. Halting.")
    sys.exit(1)


def clean_temp_files():
  """Removes leftover MASTER scratch files."""
  temp_patterns = ["*.tmp", "fort.*", "scratch.*"]
  for pattern in temp_patterns:
    for filepath in master_dir.glob(pattern):
      try:
        filepath.unlink()
      except OSError:
        pass

def update_master_inp(run_id, sma, ecc, inc, raan, aop):
  """Generates master.inp matching the exact working ESA-MASTER structure."""
  inp_content = f"""#----<ESA-MASTER>--------------------------------------------------------------
#
#           __/_/   __/__/   __/__/    _/__/_/ __/__/__/ __/__/__/ __/_/__/
#          __/ _/__/ __/ __/   __/ __/        __/    __/        __/    __/
#         __/  __/  __/ __/__/__/  __/__/    __/    __/__/    __/_/__/
#        __/      __/ __/   __/      __/   __/    __/        __/    __/
#       __/      __/ __/   __/ _/__/_/    __/    __/__/__/ __/     __/
#
# __ ESA Meteoroid & Space Debris Terrestrial Environment Reference Model__
#-----------------------------------------------------------------------------
# Purpose: Definition of execution parameters of the MASTER command line tool.
#-----------------------------------------------------------------------------
# Note: Lines which may be edited by the user are NOT marked by comment
#       sign ('#').
#-----------------------------------------------------------------------------
#
#-----------------------------< Run Settings >--------------------------------
# Note: Give the program run a name and a comment. Do NOT modify or
#       remove the '-(27 ' sequence since it serves as End of String marker.
#-----------------------------------:-----------------------------------------
# Run identifier (used as output file prefix)
#-------------------------|
{run_id:<27}-(27 char)-
#
# Run comment (2 lines, appears as header in any output file and in plots)
#----------------------------------------------------------------------|
ESA-MASTER Model

#
#----------------------------< Time Settings >--------------------------------
# Note: Specify start and end of the analysis epoch.
#-----------------------------------:-----------------------------------------
# Begin and end of analysis time interval
 2026 02 01 00  -(yyyy mm dd hh)- Begin
 2026 02 01 00  -(yyyy mm dd hh)- End
#
#--------------------------< Impactor Settings >------------------------------
# Note: Specify which source terms and size/mass ranges shall be considered
#       for the analysis.
#-----------------------------------------------------------------------------
# Source switches (0=off,1=on)
  1                -(0,1)-            Explosion Fragments
  1                -(0,1)-            Collision Fragments
  1                -(0,1)-            Launch/Mission related objects
  0                -(0,1)-            NaK-droplets
  0                -(0,1)-            SRM slag
  0                -(0,1)-            SRM Al2O3 dust
  0                -(0,1)-            Paint flakes
  0                -(0,1)-            Ejecta
  0                -(0,1)-            MLI
  1                -(0,1)-            Human-made population
  0                -(0,1)-            Meteoroids
  0                -(0,1)-            Clouds
  0                -(0,1)-            SIM overlay
#
# Cloud file identifier
  FY-1C                -(ccccc)-
#
# SIM file identifier
  mysim                -(ccccc)-
#
# Constellation projection switch (0=off,1=on)
  0                -(0,1)-            0 = no constellation projection for TLE
#                                     1 = use const. projection for TLE Backgr.
#
# Annual meteoroid stream consideration switch
  0                -(0:2)-            0 = no seasonal met. streams (averaging)
#                                     1 = seasonal met. streams (Jenniskens)
#                                     2 = seasonal met. streams (Cour-Palais)
#
# Background meteoroid model consideration swith
  0                -(0:2)-            0 = no background meteoroids
#                                     1 = background meteoroids (Divine-Staubach)
#                                     2 = background meteoroids ( Grun )
#
# Background meteoroid population switches for Devine-Staubach (0=off,1=on)
  0                -(0,1)-            Core population
  0                -(0,1)-            Asteroidal population
  0                -(0,1)-            A population
  0                -(0,1)-            B population
  0                -(0,1)-            C population
#
# Velocity distribution for GRÜN meteoroid model
  0                -(0:1)-            0 = Grün (constant velocity 20 km/s)
#                                     1 = Taylor distribution
#
# Analysis size/mass thresholds
  1.00000e-03 m   -(value (m,kg))-   Lower threshold
  1.00000e-02 m   -(value (m,kg))-   Upper threshold
#
#---------------------------< Target Settings >-------------------------------
# Note: Specify the type of target to be analyzed.
#-----------------------------------:-----------------------------------------
# Analysis mode
  1                -(1:4)-            0 = orbiting target, static population (Earth-bound)
#                                     1 = orbiting target (Earth-bound)
#                                     2 = inertial volume (Earth-bound)
#                                     3 = spatial density (Earth-bound)
#                                     4 = lagrange point (non-Earth-bound)
#
# Target type
  1                -(1:3)-            1 = sphere
#                                     2 = randomly tumbling plate
#                                     3 = oriented surface (defined in .sdf file)
#
# Target properties (only if analysis mode 1 and propagation requested)
# Value____________Unit______________Description___________________
  9710.0            -(kg)-             Mass                   (default: 9710.0)
  39.13             -(m**2)-           Cross section (drag)   (default: 39.13)
  39.13             -(m**2)-           Cross section (SRP)    (default: 39.13)
  2.2               -( - )-            Drag coefficient       (default: 2.2)
  0.0               -(/d)-             Drag coefficient rate  (default: 0.0)
  1.3               -( - )-            Reflection coefficient (default: 1.3)
#
# Target orbit propagation resolution (only in analysis mode 1)
  2                -(1:4)-            1 = 1 month
#                                     2 = 3 months
#                                     3 = 6 months
#                                     4 = 1 year
#
# Target orbit(s) (only in analysis mode 1) in Earth-centered inertial frame (ECI)
# Prop.sw   start epoch        end epoch    SMA        ECC        INC      RAAN     AoP
#___0/1____yyyy_mm_dd_hh__yyyy_mm_dd_hh___[km]_______[-]______[deg]____[deg]____[deg]___
     0     2026 02 01 00  2026 02 01 00  {sma:<8.1f} {ecc:<8.3f} {inc:<6.2f} {raan:<7.4f} {aop:<7.4f}
#
# Orbital arc (considered only in analysis mode 1)
   0.0000          -(deg)-            Lower argument of true latitude
   360.00          -(deg)-            Upper argument of true latitude
#
# Inertial volume position (considered only in analysis mode 2)
   7178.0          -(km)-             Geocentric distance
   0.0000          -(deg)-            Right ascension
   0.0000          -(deg)-            Declination
#
# Spatial density profile range (considered only in analysis mode 3)
   186.00          -(km)-             Lower altitude limit
 36786.00          -(km)-             Upper altitude limit
  -90.000          -(deg)-            Lower declination limit
   90.000          -(deg)-            Upper declination limit
  -180.00          -(deg)-            Lower right ascension limit
   180.00          -(deg)-            Upper right ascension limit
#
#---------------------< Definition of Input File Names >----------------------
# Note: Give the name of additional input files to be used (do NOT modify or
#       remove the '-(120 ' sequence since it serves as End of String marker.
#-----------------------------------:-----------------------------------------
  default.def      -(120 char)-       Basic output spectrum definition file
  default.sdf      -(120 char)-       Surface description file
  default.con      -(120 char)-       Constellation description file
#
#------------------------< Basic Output Settings >----------------------------
# Note: Activate or de-activate spectrum and data output.
#-----------------------------------:-----------------------------------------
# Differential spectra
  1                -(0,1)-            0 = don't generate differential spectra
#                                     1 = generate differential spectrum files
# Cumulative spectra
  1                -(0,1)-            0 = don't generate cumulative spectra
#                                     1 = generate cumulative spectrum files
# Reverse cumulative spectra
  1                -(0,1)-            0 = don't generate reverse cumulative spec.
#                                     1 = generate reverse cumulative spec. file
# Additional dump of CPE/STENVI data (experts feature)
  0                -(0,1)-            0 = don't dump cell passage characteristics
#                                     1 = dump CPE        (ATTENTION: spacious!)
  0                -(0,1)-            0 = don't dump STENVI
#                                     1 = dump STENVI     (ATTENTION: spacious!)
#
# STENVI definition of output spectrum
# Bin        Min        Max
    10 -1.80000e+02  1.80000e+02     Azimuth   [deg]
    10 -9.00000e+01  9.00000e+01     Elevation [deg]
    10  5.00000e-01  4.05000e+01     Velocity  [km/s]
    10  1.00000e-03  1.00000e-02     Diameter  [m]
     1  0.00000e+00  3.60000e+02     Argument of true Latitude [deg]
     1  0.00000e+00  5.00000e+00     Density   [g/cm^3]
#
# Switch for indication of uncertainty bars (2D-plot)
  1                -(0,1)-            0 = don't plot uncertainty bars
#                                     1 = plot uncertainty bars
#
#--------------------------< Damage Law Settings >----------------------------
# Note: Set calibration parameters for the conchoidal diameter damage equation
#       if you plan to analyze flux vs. impact feature size on britte surfaces
#       (see user manual for details).
#-----------------------------------:-----------------------------------------
# Calibration parameters for conchoidal diameter damage equation
   1.0000          -(--)-             Dh/dp ratio
   1.0000          -(--)-             Correction factor to the Taylor formula
   0.0000          -(mu)-             Taylor diameter reduction
   0.0000          -(mu)-             Minimum Taylor diameter Dmin
   12000           -(mu)-             Conchoidal interception diameter
   100.00          -(mu)-             Mean diameter for Gauss filter
   4.0000          -(--)-             Standard deviation for Gauss filter
   0.80000         -(--)-             Gauss factor
#
#
#--eof------------------------------------------------------------------eof---
"""
  input_inp_path.parent.mkdir(parents=True, exist_ok=True)
  with open(input_inp_path, "w", encoding="utf-8", newline="\n") as f:
    f.write(inp_content)
verify_paths()

for idx, orb in enumerate(target_orbits, start=1):
  run_id = f"task_{idx:02d}"
  task_dir = results_base / f"task-{idx}"
  task_dir.mkdir(parents=True, exist_ok=True)

  print("=" * 60)
  print(f"RUNNING ORBIT {idx}/20: SMA={orb[0]}, INC={orb[2]} (ID: {run_id})")
  print("=" * 60)

  clean_temp_files()
  update_master_inp(run_id, *orb)

  # Track directory contents before execution
  before_files = set(master_dir.rglob("*"))

  print("-> Executing MASTER binary...")
  result = subprocess.run(
      [str(exe_path)], cwd=str(master_dir), capture_output=True, text=True
  )

  # Log stdout and stderr
  if result.stdout.strip():
    print(f"\n--- STDOUT ---\n{result.stdout}")
  if result.stderr.strip():
    print(f"\n--- STDERR ---\n{result.stderr}")

  # Check MASTER logfile if created
  logfile = master_dir / "logfile"
  if logfile.exists():
    log_content = logfile.read_text(errors="ignore")
    print("\n--- MASTER LOGFILE (Last 12 lines) ---")
    print("\n".join(log_content.splitlines()[-12:]))

  # Diff directory contents to see created files
  after_files = set(master_dir.rglob("*"))
  new_files = [f for f in (after_files - before_files) if f.is_file()]

  print("\n--- NEWLY GENERATED FILES ---")
  if new_files:
    for f in new_files:
      print(f" + {f.relative_to(master_dir)}")
  else:
    print(" [WARNING] No new files detected in MASTER directory!")

  # Find generated output files matching target run_id
  matching_files = list(master_dir.glob(f"{run_id}*")) + list(
      (master_dir / "output").glob(f"{run_id}*")
  )

  print(f"\n--- MOVING OUTPUTS TO {task_dir.relative_to(script_dir)} ---")
  if matching_files:
    for file_path in matching_files:
      if file_path.is_file():
        dest = task_dir / file_path.name
        shutil.move(str(file_path), str(dest))
        print(f" Moved: {file_path.name} -> {dest}")
  else:
    print(
        f" [ERROR] No files starting with '{run_id}' were found to move."
    )

  # Stop on first error so you can debug the terminal output immediately
  if not matching_files or "FATAL ERROR" in logfile.read_text(errors="ignore"):
    print(
        "\n[HALTED FOR DEBUGGING] Run failed on step 1. Review the logs above."
    )
    sys.exit(1)

  time.sleep(0.5)

print("\nCompleted all orbits successfully!")