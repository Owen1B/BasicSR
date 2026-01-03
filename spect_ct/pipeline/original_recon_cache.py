from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from spect_ct.pipeline.osem import run_osem_reconstruction


@dataclass(frozen=True)
class PatientPaths:
    patient: str
    patient_dir: Path
    proj: Path
    atten: Path
    par: Path


def resolve_patient_paths(spect229_dir: Path, patient: str) -> PatientPaths:
    """Resolve the canonical SPECT229 file layout under datasets/SPECT229/<patient>/."""
    pdir = Path(spect229_dir) / str(patient)
    proj = pdir / f"{patient}_Proj4Filter.dat"
    atten = pdir / f"{patient}_PostAtten.dat"
    par = pdir / f"{patient}_ParamFile.par"
    missing = [str(x) for x in [proj, atten, par] if not x.exists()]
    if missing:
        raise FileNotFoundError(f"Missing SPECT229 files for patient={patient}: {missing}")
    return PatientPaths(patient=str(patient), patient_dir=pdir, proj=proj, atten=atten, par=par)


def ensure_original_recon(
    *,
    spect229_dir: Path,
    patient: str,
    iterations: int = 10,
    views_per_subset: int | None = None,
    orbit_file: Optional[Path] = None,
    overwrite: bool = False,
    timeout_sec: int = 600,
) -> Path:
    """Ensure original projection OSEM recon exists under datasets/SPECT229/<patient>/.

    Output:
      datasets/SPECT229/<patient>/<patient>_OSEMReconed_20s_new_Iter{N}.dat

    Notes:
    - Uses the same external OSEM pipeline as current code (run_osem_reconstruction).
    - Uses patient's ParamFile.par + PostAtten.dat, and orbit.orb defaults to spect_ct/osemreocnexe/orbit.orb.
    """
    it = int(iterations)
    paths = resolve_patient_paths(Path(spect229_dir), str(patient))

    # Encode key recon parameters in filename to avoid overwriting when settings change.
    # User convention:
    #   <patient>_OSEMReconed_20s_new_Iter{N}_Sub{S}.dat
    # If views_per_subset is not provided, fall back to legacy naming without _Sub.
    if views_per_subset is not None:
        out = paths.patient_dir / f"{paths.patient}_OSEMReconed_20s_new_Iter{it}_Sub{int(views_per_subset)}.dat"
    else:
        out = paths.patient_dir / f"{paths.patient}_OSEMReconed_20s_new_Iter{it}.dat"
    if out.exists() and not overwrite:
        return out

    if orbit_file is None:
        orbit_file = Path(__file__).resolve().parents[1] / "osemreocnexe" / "orbit.orb"
    orbit_file = Path(orbit_file)
    if not orbit_file.exists():
        raise FileNotFoundError(f"orbit file not found: {orbit_file}")

    # Make OSEM output filename match the user's convention (no extra output_name segment).
    run_osem_reconstruction(
        proj_file=paths.proj,
        patient_name=paths.patient,
        par_file=paths.par,
        orbit_file=orbit_file,
        atten_file=paths.atten,
        output_name="original",  # not used when output_filename is provided
        iterations=it,
        views_per_subset=views_per_subset,
        final_output_dir=paths.patient_dir,
        output_filename=out.name,
        timeout_sec=int(timeout_sec),
    )

    if not out.exists():
        raise FileNotFoundError(f"original recon was not generated: {out}")
    return out


