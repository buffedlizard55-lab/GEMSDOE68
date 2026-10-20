"""Validate a candidate submission GeoTIFF against the published format rules.

Checks include one float32 band, EPSG:32611, 100 m resolution, exact reference-grid
shape/transform/CRS, values in [0, 1], and NaN-only outside the official footprint.
Checks that require the reference/features rasters are SKIPPED without them; SKIPPED is
never treated as a pass.

CLI: python -m gems.validate_submission FILE.tif [--reference data/labels.tif]
     [--features data/training_features.tif]
Exit code 0 means every required check passed; otherwise 1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .constants import EPSG, FEATURE_NODATA_SENTINEL, PIXEL_M, SHAPE_HW


def _check(name, ok, detail, status=None):
    return {"check": name, "status": status or ("PASS" if ok else "FAIL"), "detail": detail}


def _same_transform(a, b, tolerance: float = 1e-6) -> bool:
    return all(abs(float(x) - float(y)) < tolerance for x, y in zip(a[:6], b[:6]))


def footprint_from_features(features_path: str | Path, shape_hw=None) -> np.ndarray:
    """Footprint = cells whose first feature band is finite and above the nodata sentinel."""
    import rasterio

    with rasterio.open(features_path) as src:
        band = src.read(1).astype(np.float64)
    fp = np.isfinite(band) & (band > FEATURE_NODATA_SENTINEL)
    if shape_hw is not None and fp.shape != tuple(shape_hw):
        raise ValueError(f"features shape {fp.shape} != expected {shape_hw}")
    return fp


def validate(path: str | Path, reference_path: str | Path | None = None,
             features_path: str | Path | None = None) -> dict:
    import rasterio

    path = Path(path)
    checks: list[dict] = []
    if not path.exists():
        return {"file": str(path), "passed": False,
                "checks": [_check("file exists", False, "missing")],
                "has_failures": True, "skipped_checks_mean_unverified": False}

    ref_shape = None
    ref_transform = None
    ref_crs = None
    if reference_path is not None:
        with rasterio.open(reference_path) as ref:
            ref_shape = (ref.height, ref.width)
            ref_transform = ref.transform
            ref_crs = ref.crs

    with rasterio.open(path) as src:
        out_shape = (src.height, src.width)
        out_transform = src.transform
        out_crs = src.crs
        checks.append(_check("single band", src.count == 1, f"count={src.count}"))
        checks.append(_check("dtype float32", src.dtypes[0] == "float32", f"dtype={src.dtypes[0]}"))
        checks.append(_check("CRS EPSG:32611", out_crs is not None and out_crs.to_string() == EPSG,
                             f"crs={out_crs.to_string() if out_crs else None}"))
        expected_shape = ref_shape if ref_shape is not None else SHAPE_HW
        checks.append(_check("shape matches", out_shape == tuple(expected_shape),
                             f"file={out_shape} expected={tuple(expected_shape)}"))
        res_ok = abs(abs(src.res[0]) - PIXEL_M) < 1e-9 and abs(abs(src.res[1]) - PIXEL_M) < 1e-9
        checks.append(_check("resolution 100 m", res_ok, f"res={src.res}"))
        if ref_transform is not None:
            checks.append(_check("transform matches reference", _same_transform(out_transform, ref_transform),
                                 f"file={tuple(out_transform)[:6]} ref={tuple(ref_transform)[:6]}"))
            checks.append(_check("CRS matches reference",
                                 out_crs is not None and ref_crs is not None and out_crs == ref_crs,
                                 f"file={out_crs} ref={ref_crs}"))
        else:
            checks.append(_check("transform matches reference", False,
                                 "SKIPPED: no reference raster supplied", status="SKIPPED"))
            checks.append(_check("CRS matches reference", False,
                                 "SKIPPED: no reference raster supplied", status="SKIPPED"))
        nodata = src.nodata
        nodata_ok = nodata is None or (isinstance(nodata, (float, np.floating)) and np.isnan(nodata))
        checks.append(_check("nodata tag is none or NaN", nodata_ok,
                             f"nodata={nodata} (finite nodata tags are rejected conservatively)"))
        arr = src.read(1).astype(np.float64)

    finite = np.isfinite(arr)
    n_inf = int(np.isinf(arr).sum())
    checks.append(_check("no infinities", n_inf == 0, f"inf_count={n_inf}"))
    vals = arr[finite]
    if vals.size:
        vmin, vmax = float(vals.min()), float(vals.max())
    else:
        vmin = vmax = float("nan")
    checks.append(_check("values within [0, 1]", vals.size > 0 and vmin >= 0.0 and vmax <= 1.0,
                         f"min={vmin} max={vmax} finite_cells={int(finite.sum())}"))

    if features_path is not None:
        with rasterio.open(features_path) as feats:
            band = feats.read(1).astype(np.float64)
            feat_shape = (feats.height, feats.width)
            feat_transform = feats.transform
            feat_crs = feats.crs
        fp = np.isfinite(band) & (band > FEATURE_NODATA_SENTINEL)
        if ref_shape is not None:
            grid_shape = ref_shape
            grid_transform = ref_transform
            grid_crs = ref_crs
            grid_label = "reference"
        else:
            grid_shape = out_shape
            grid_transform = out_transform
            grid_crs = out_crs
            grid_label = "candidate (reference absent)"
        checks.append(_check("features grid shape matches", feat_shape == grid_shape,
                             f"features={feat_shape} {grid_label}={grid_shape}"))
        checks.append(_check("features transform matches reference grid",
                             _same_transform(feat_transform, grid_transform),
                             f"features={tuple(feat_transform)[:6]} {grid_label}={tuple(grid_transform)[:6]}"))
        checks.append(_check("features CRS matches reference grid",
                             feat_crs is not None and grid_crs is not None and feat_crs == grid_crs,
                             f"features={feat_crs} {grid_label}={grid_crs}"))
        if ref_shape is None:
            # Alignment to the candidate is useful but does not establish official-grid alignment.
            checks.append(_check("features/reference alignment verified", False,
                                 "SKIPPED: no official reference raster supplied", status="SKIPPED"))

        if fp.shape != arr.shape:
            checks.append(_check("footprint shape matches candidate", False,
                                 f"footprint={fp.shape} candidate={arr.shape}"))
        else:
            nan_inside = int((~finite & fp).sum())
            checks.append(_check("no NaN inside footprint", nan_inside == 0,
                                 f"nan_inside={nan_inside} footprint_cells={int(fp.sum())}"))
            outside_finite = int((finite & ~fp).sum())
            checks.append(_check("NaN-only outside footprint", outside_finite == 0,
                                 f"finite_outside={outside_finite}; only NaN is allowed outside"))
    else:
        checks.append(_check("footprint checks", False,
                             "SKIPPED: no features raster supplied", status="SKIPPED"))

    nonzero = int(np.nansum(arr > 0))
    checks.append(_check("info: emitted cells (value > 0)", True, f"emitted={nonzero}", status="INFO"))
    failed = [c for c in checks if c["status"] == "FAIL"]
    skipped = [c for c in checks if c["status"] == "SKIPPED"]
    from .submission_writer import sha256_file
    return {"file": str(path), "sha256": sha256_file(path),
            "passed": not failed and not skipped, "has_failures": bool(failed),
            "checks": checks, "skipped_checks_mean_unverified": bool(skipped)}


def validate_portal_ready(path: str | Path, sample_sub_path: str | Path,
                          features_path: str | Path,
                          catalogue_path: str | Path | None = None) -> dict:
    """Strict DrivenData upload-portal preflight check.

    Specifically catches IR-68-PORTAL-01: the 3,061-cell mismatch where
    ``sample_submission.tif`` is finite (5,167,373 cells) while band 1 of
    ``training_features.tif`` carries nodata (5,165,852 cells). Any NaN inside
    ``np.isfinite(sample_submission)`` causes DrivenData's upload check
    ``((pred[valid] >= 0) & (pred[valid] <= 1)).all()`` to fail with
    ``"Predicted values must be in range [0, 1]"``.
    """
    import rasterio
    from .submission_writer import sha256_file

    path = Path(path)
    with rasterio.open(sample_sub_path) as ref:
        sub = ref.read(1)
        fp_sub = np.isfinite(sub)
        ref_shape = (ref.height, ref.width)
        ref_crs = ref.crs
        ref_transform = ref.transform

    fp_feat = footprint_from_features(features_path, ref_shape)
    with rasterio.open(path) as src:
        arr = src.read(1)
        out_shape = (src.height, src.width)
        out_crs = src.crs
        out_transform = src.transform
        out_dtype = src.dtypes[0]
        out_count = src.count
        out_nodata = src.nodata

    finite = np.isfinite(arr)
    nan_in_sample_sub = int((~finite & fp_sub).sum())
    nan_in_union = int((~finite & (fp_sub | fp_feat)).sum())
    n_inf = int(np.isinf(arr).sum())
    vals = arr[finite]
    vmin = float(vals.min()) if vals.size else float("nan")
    vmax = float(vals.max()) if vals.size else float("nan")
    in_range = bool(vals.size > 0 and vmin >= 0.0 and vmax <= 1.0)
    all_pixels_in_0_1 = bool(((arr >= 0.0) & (arr <= 1.0)).all())

    on_cat = None
    if catalogue_path is not None:
        with rasterio.open(catalogue_path) as cat_ds:
            cat = cat_ds.read(1) > 0
        on_cat = int(((arr > 0) & finite & cat).sum())

    checks = [
        _check("single band", out_count == 1, f"count={out_count}"),
        _check("dtype float32", out_dtype == "float32", f"dtype={out_dtype}"),
        _check("CRS EPSG:32611", out_crs is not None and out_crs == ref_crs, f"crs={out_crs}"),
        _check("shape 3730x3292", out_shape == ref_shape, f"shape={out_shape}"),
        _check("transform matches sample_submission", _same_transform(out_transform, ref_transform),
               f"transform={tuple(out_transform)[:6]}"),
        _check("no infinities anywhere", n_inf == 0, f"inf={n_inf}"),
        _check("finite values strictly in [0, 1]", in_range, f"min={vmin} max={vmax}"),
        _check("zero NaN inside sample_submission footprint (5,167,373 cells)",
               nan_in_sample_sub == 0,
               f"nan_in_sample_sub={nan_in_sample_sub}"),
        _check("zero NaN inside union(sample_submission, features) (5,168,913 cells)",
               nan_in_union == 0,
               f"nan_in_union={nan_in_union}"),
    ]
    if on_cat is not None:
        checks.append(_check("zero positive dots on mapped catalogue", on_cat == 0,
                             f"on_catalogue={on_cat}"))

    passed = all(c["status"] == "PASS" for c in checks)
    return {
        "file": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "nodata": None if (out_nodata is None or np.isnan(out_nodata)) else float(out_nodata),
        "all_12279160_pixels_finite_in_0_1": all_pixels_in_0_1,
        "nan_in_sample_sub": nan_in_sample_sub,
        "nan_total": int((~finite).sum()),
        "emitted_positive_pixels": int(np.nansum(arr > 0)),
        "min": vmin,
        "max": vmax,
        "on_catalogue_positive_pixels": on_cat,
        "passed": passed,
        "checks": checks,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("file")
    ap.add_argument("--reference", help="labels.tif (or any file with the official grid)")
    ap.add_argument("--features", help="training features raster, for the footprint check")
    args = ap.parse_args(argv)
    report = validate(args.file, args.reference, args.features)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
