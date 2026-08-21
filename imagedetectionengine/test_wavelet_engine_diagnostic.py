"""Diagnostic test harness for the wavelet-domain forgery engine.

Runs the engine over three cases (authentic camera images, manipulated images,
AI-generated images), records the full computation trace for every image, and
writes every result to disk so the findings survive the session.

Nothing here is part of the engine. This file only measures it.

Usage:
    python3 test_wavelet_engine_diagnostic.py [--tag before|after]

The --tag argument names the output subfolder so a run made before an
enhancement and a run made after it can be compared side by side.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image, ExifTags

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wavelet_engine.contracts import EngineInput, EngineOutput, ImageMetadata
from wavelet_engine.engine import WaveletEngine


@lru_cache(maxsize=None)
def load_standard_jpeg_quantization_table(quality_factor: int) -> tuple:
    """Return the IJG luminance table Pillow's encoder emits at a quality factor.

    Obtained by encoding a blank probe and reading the table back, so the IJG
    scaling rule stays in the library that owns it and no constants are
    hardcoded here. Verified: the QF50 probe reproduces the published IJG base
    table whose first row is 16, 11, 10, 16, 24, 40, 51, 61.

    Args:
        quality_factor: JPEG quality in 1..100.

    Returns:
        Flat 64-entry tuple of quantization steps in raster order.
    """
    probe = Image.fromarray(np.zeros((8, 8), dtype=np.uint8))
    buffer = io.BytesIO()
    probe.save(buffer, format="JPEG", quality=int(quality_factor))
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return tuple(decoded.quantization[0])

REAL_IMAGES_FOLDER = (
    "/home/kingsley-judewin/dsp project/"
    "imagedetectionengine/test images/real images"
)
FAKE_IMAGES_FOLDER = (
    "/home/kingsley-judewin/dsp project/"
    "imagedetectionengine/test images/fake images"
)
AI_IMAGES_FOLDER = (
    "/home/kingsley-judewin/dsp project/"
    "imagedetectionengine/test images/ai generated images"
)
OUTPUT_ROOT = (
    "/home/kingsley-judewin/dsp project/"
    "imagedetectionengine/test results/wavelet_engine"
)

# The AI folder named in the task specification does not exist on disk. Two
# files inside the fake-images folder are named for AI generation (gen.jpeg,
# genratedimage.jpeg) and are used as the Case 3 stand-in. This substitution is
# declared in every report rather than being applied silently.
AI_SUBSTITUTE_FILENAMES = ("gen.jpeg", "genratedimage.jpeg")
CASE_2_PREFERRED_FILENAMES = ("fake .jpeg", "fake.jpeg")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")

# Verdict bands used to convert a probability into a label. These are the
# harness's reading of the engine's output, not engine constants.
REAL_VERDICT_CEILING = 0.30
FAKE_VERDICT_FLOOR = 0.70

# A PNG is lossless, so its "compression level" is reported at the top of the
# quality scale; the engine reads this as "no quantization loss".
LOSSLESS_COMPRESSION_LEVEL = 100.0


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class ImageFacts:
    """Everything measurable about the file before the engine sees it."""

    path: str
    filename: str
    format: str
    height: int
    width: int
    file_size_bytes: int
    estimated_quality_factor: float
    quality_estimation_method: str
    has_exif: bool
    is_resized: bool
    resize_evidence: str
    loadable: bool
    note: str = ""


@dataclass
class ImageResult:
    """One image's engine output plus the harness's assessment of it."""

    facts: dict
    case_type: str
    ground_truth: str
    raw_score: float
    probability: Optional[float]
    confidence: float
    is_reliable: bool
    reliability_note: str
    processing_time_ms: float
    engine_verdict: str
    verdict_correct: bool
    surprise_level: str
    reasoning: str
    mathematical_explanation: str
    computation_steps: list = field(default_factory=list)
    evidence_map_path: str = ""
    evidence_map_colour_path: str = ""
    deep_probe: dict = field(default_factory=dict)
    error: str = ""


@dataclass
class CaseResult:
    """Aggregate of the two images belonging to one case."""

    case_number: int
    case_type: str
    ground_truth: str
    folder: str
    images: list = field(default_factory=list)
    average_probability: Optional[float] = None
    score_spread: Optional[float] = None
    correct_verdicts: int = 0
    case_verdict: str = ""
    summary: str = ""


# ---------------------------------------------------------------------------
# Phase 0 - loading and metadata
# ---------------------------------------------------------------------------

def load_images_from_folder(folder_path: str, n: int = 2,
                            preferred: tuple = (),
                            require_all: bool = True) -> list:
    """Load the first n loadable images from a folder.

    Args:
        folder_path: Directory to read.
        n: How many images are wanted.
        preferred: Filenames to try first, in order, before falling back to
            alphabetical order.
        require_all: When True, fewer than n loadable images is an error. Set
            False to enumerate whatever the folder holds.

    Returns:
        List of (path, cv2_image, pil_image) tuples.

    Raises:
        FileNotFoundError: If the folder does not exist.
        RuntimeError: If require_all and fewer than n images could be loaded.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise FileNotFoundError(f"folder does not exist: {folder_path}")

    candidates = sorted(
        entry for entry in folder.iterdir()
        if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS
    )
    ordered = ([folder / name for name in preferred if (folder / name).is_file()]
               + [entry for entry in candidates
                  if entry.name not in preferred])

    loaded = []
    for entry in ordered:
        if len(loaded) >= n:
            break
        try:
            cv_image = cv2.imread(str(entry), cv2.IMREAD_COLOR)
            if cv_image is None:
                print(f"    SKIP (cv2 returned None): {entry.name}")
                continue
            pil_image = Image.open(str(entry))
            pil_image.load()
            loaded.append((str(entry), cv_image, Image.open(str(entry))))
        except Exception as error:  # noqa: BLE001 - harness must not die here
            print(f"    SKIP (unreadable: {error}): {entry.name}")

    if require_all and len(loaded) < n:
        raise RuntimeError(
            f"only {len(loaded)} of {n} required images could be loaded from "
            f"{folder_path}"
        )
    return loaded


def estimate_quality_factor_from_table(luminance_table: list) -> tuple:
    """Recover the JPEG quality factor that produced a quantization table.

    The table is compared against the table Pillow's own encoder emits at every
    quality factor from 1 to 100, and the closest match by absolute difference
    is returned. This uses the encoder that owns the IJG scaling rule instead of
    reimplementing it, so no scaling constants are hardcoded here.

    Args:
        luminance_table: Flat 64-entry luminance quantization table.

    Returns:
        Tuple of (quality factor, mean absolute residual of the best match).
    """
    observed = np.asarray(luminance_table, dtype=np.float64).reshape(8, 8)
    best_quality, best_residual = 1, float("inf")
    for quality in range(1, 101):
        candidate = np.asarray(load_standard_jpeg_quantization_table(quality),
                               dtype=np.float64).reshape(8, 8)
        residual = float(np.mean(np.abs(candidate - observed)))
        if residual < best_residual:
            best_quality, best_residual = quality, residual
    return float(best_quality), best_residual


def _read_exif(pil_image: Image.Image) -> dict:
    """Read EXIF tags into a name-keyed dictionary.

    Args:
        pil_image: Opened PIL image.

    Returns:
        Dictionary of tag name to value; empty when no EXIF block exists.
    """
    try:
        raw = pil_image._getexif()  # noqa: SLF001 - the documented accessor
    except Exception:  # noqa: BLE001
        raw = None
    if not raw:
        return {}
    return {ExifTags.TAGS.get(tag, str(tag)): value for tag, value in raw.items()}


def _estimate_compression(path: str, pil_image: Image.Image) -> tuple:
    """Estimate a 0-100 compression quality for any supported container.

    Args:
        path: Path on disk, for the size-based WEBP estimate.
        pil_image: Opened PIL image.

    Returns:
        Tuple of (estimated quality factor, method description).
    """
    container = (pil_image.format or "").upper()

    if container in ("JPEG", "JPG", "MPO"):
        tables = getattr(pil_image, "quantization", None)
        if tables:
            quality, residual = estimate_quality_factor_from_table(tables[0])
            return quality, (f"matched the embedded luminance quantization "
                             f"table against Pillow's encoder tables for "
                             f"QF 1-100; best match QF={quality:.0f} with mean "
                             f"absolute residual {residual:.3f}")
        return 75.0, "JPEG with no readable quantization table; default assumed"

    if container == "PNG":
        return LOSSLESS_COMPRESSION_LEVEL, "PNG is lossless; quality set to 100"

    if container == "WEBP":
        width, height = pil_image.size
        bits_per_pixel = (os.path.getsize(path) * 8.0) / float(width * height)
        # Anchored so that ~1 bit/pixel maps near the top of the scale, which is
        # roughly where visually-lossless WEBP sits.
        quality = float(np.clip(bits_per_pixel * 100.0, 1.0, 100.0))
        return quality, (f"WEBP estimated from bitrate {bits_per_pixel:.3f} "
                         f"bits/pixel (no quantization table is exposed)")

    return 50.0, f"unrecognised container {container}; conservative default"


def compute_metadata(image_path: str,
                     pil_image: Image.Image) -> tuple:
    """Compute genuine ImageMetadata from an actual image file.

    No field is a placeholder. The compression level comes from the embedded
    quantization table for JPEG, from losslessness for PNG, and from bitrate for
    WEBP; the resize flag comes from EXIF pixel dimensions when present.

    Args:
        image_path: Path on disk.
        pil_image: Opened PIL image.

    Returns:
        Tuple of (ImageMetadata, ImageFacts).
    """
    width, height = pil_image.size
    exif = _read_exif(pil_image)
    quality, method = _estimate_compression(image_path, pil_image)

    exif_width = exif.get("ExifImageWidth")
    exif_height = exif.get("ExifImageHeight")
    if exif_width and exif_height:
        is_resized = (int(exif_width), int(exif_height)) != (width, height)
        evidence = (f"EXIF records {exif_width}x{exif_height}, file is "
                    f"{width}x{height}")
    else:
        is_resized = False
        evidence = "no EXIF pixel dimensions; resize state unknown, reported False"

    metadata = ImageMetadata(
        estimated_compression_level=quality,
        is_resized=is_resized,
        color_space="BGR",
        resolution=(height, width),
        format=(pil_image.format or "UNKNOWN").upper(),
        has_exif=bool(exif),
    )
    facts = ImageFacts(
        path=image_path,
        filename=Path(image_path).name,
        format=metadata.format,
        height=height,
        width=width,
        file_size_bytes=os.path.getsize(image_path),
        estimated_quality_factor=quality,
        quality_estimation_method=method,
        has_exif=bool(exif),
        is_resized=is_resized,
        resize_evidence=evidence,
        loadable=True,
    )
    return metadata, facts


def print_pre_run_summary(entries: list) -> str:
    """Print and return the pre-run image summary table.

    Args:
        entries: List of (ImageFacts, case label) tuples.

    Returns:
        The rendered table as a string.
    """
    header = (f"{'Image':<22} | {'Format':<6} | {'Resolution':<12} | "
              f"{'Est.QF':>6} | {'EXIF':<4} | {'Size KB':>8} | Case")
    lines = [header, "-" * len(header)]
    for facts, case_label in entries:
        lines.append(
            f"{facts.filename:<22} | {facts.format:<6} | "
            f"{facts.width}x{facts.height:<7} | "
            f"{facts.estimated_quality_factor:>6.0f} | "
            f"{'Yes' if facts.has_exif else 'No':<4} | "
            f"{facts.file_size_bytes / 1024.0:>8.1f} | {case_label}"
        )
    table = "\n".join(lines)
    print(table)
    return table


# ---------------------------------------------------------------------------
# Engine execution
# ---------------------------------------------------------------------------

def run_engine_safely(engine: WaveletEngine,
                      image: np.ndarray,
                      metadata: ImageMetadata) -> dict:
    """Run the engine, capturing any exception instead of propagating it.

    Args:
        engine: Configured engine instance.
        image: BGR uint8 image.
        metadata: Real metadata for this image.

    Returns:
        Dictionary holding either the EngineOutput or a full traceback.
    """
    started = time.perf_counter()
    try:
        output = engine.analyse(EngineInput(image=image, metadata=metadata))
        return {
            "ok": True,
            "output": output,
            "wall_clock_ms": (time.perf_counter() - started) * 1000.0,
            "error": "",
        }
    except Exception as error:  # noqa: BLE001 - the point of this function
        return {
            "ok": False,
            "output": None,
            "wall_clock_ms": (time.perf_counter() - started) * 1000.0,
            "error": f"{type(error).__name__}: {error}\n{traceback.format_exc()}",
        }


def feature_space_anatomy(image: np.ndarray, sample_blocks: int = 400) -> dict:
    """Measure the feature-space scale that Eq. 27's threshold is applied to.

    Eq. 27 declares two blocks duplicates when S = 1/(1+rho) >= T, i.e. when
    their Euclidean distance rho <= 1/T - 1. That is an ABSOLUTE cutoff, so it
    only means anything if the feature space has a known scale. The features
    here are contrast-normalised blur invariants up to 7th order, whose scale
    is set by the recursion of Eq. 12 and is not normalised anywhere. This
    records the actual numbers so the threshold can be judged against them.

    Args:
        image: BGR uint8 image.
        sample_blocks: How many blocks to sample for the distance statistics.

    Returns:
        Dictionary of feature magnitudes, pairwise distances, and how many
        sampled pairs fall inside the engine's own match radius.
    """
    from wavelet_engine import constants as wavelet_constants
    from wavelet_engine.computer import CopyMoveDetector
    from wavelet_engine.preprocessor import WaveletPreprocessor

    preprocessor = WaveletPreprocessor()
    detector = CopyMoveDetector()
    grayscale = preprocessor.prepare(image, metadata=None).grayscale
    ll_subband = preprocessor.extract_ll_subband(grayscale)
    blocks = preprocessor.tile_blocks(ll_subband, wavelet_constants.DEFAULT_BLOCK_SIZE)

    step = max(1, len(blocks) // sample_blocks)
    sampled = blocks[::step][:sample_blocks]
    vectors = np.stack([detector.build_feature_vector(b.pixels) for b in sampled])

    radius = (1.0 / wavelet_constants.SIMILARITY_THRESHOLD) - 1.0
    distances = np.linalg.norm(vectors[:, None, :] - vectors[None, :, :], axis=2)
    upper = np.triu_indices(len(sampled), 1)
    pairwise = distances[upper]
    magnitude = np.abs(vectors)

    return {
        "ll_shape": [int(ll_subband.shape[0]), int(ll_subband.shape[1])],
        "total_blocks": int(len(blocks)),
        "sampled_blocks": int(len(sampled)),
        "feature_dimension": int(vectors.shape[1]),
        "feature_abs_min": float(f"{magnitude.min():.6g}"),
        "feature_abs_max": float(f"{magnitude.max():.6g}"),
        "feature_abs_median": float(f"{np.median(magnitude):.6g}"),
        "match_radius_from_threshold": round(radius, 6),
        "pair_distance_min": float(f"{pairwise.min():.6g}"),
        "pair_distance_median": float(f"{np.median(pairwise):.6g}"),
        "pair_distance_max": float(f"{pairwise.max():.6g}"),
        "sampled_pairs": int(pairwise.size),
        "sampled_pairs_inside_radius": int(np.count_nonzero(pairwise <= radius)),
        "orders_of_magnitude_gap": round(float(
            np.log10(max(np.median(pairwise), 1e-30) / max(radius, 1e-30))), 2),
    }


def per_order_feature_scale(image: np.ndarray, sample_blocks: int = 200) -> dict:
    """Report how the invariant magnitude grows with moment order.

    Eq. 17 divides B(p,q) by (R/2)^r * mu_00, which should make the invariants
    dimensionless. If the magnitude still climbs steeply with order r, the
    recursion of Eq. 12 is not being tamed by that normalisation and the top
    orders will dominate every Euclidean distance and every PCA component.

    Args:
        image: BGR uint8 image.
        sample_blocks: How many blocks to sample.

    Returns:
        Dictionary mapping each moment order to its median |feature| value.
    """
    from wavelet_engine import constants as wavelet_constants
    from wavelet_engine.computer import CopyMoveDetector
    from wavelet_engine.preprocessor import WaveletPreprocessor

    preprocessor = WaveletPreprocessor()
    detector = CopyMoveDetector()
    grayscale = preprocessor.prepare(image, metadata=None).grayscale
    ll_subband = preprocessor.extract_ll_subband(grayscale)
    blocks = preprocessor.tile_blocks(ll_subband, wavelet_constants.DEFAULT_BLOCK_SIZE)
    step = max(1, len(blocks) // sample_blocks)
    vectors = np.stack([detector.build_feature_vector(b.pixels)
                        for b in blocks[::step][:sample_blocks]])

    orders, index = {}, 0
    for order in range(wavelet_constants.MINIMUM_MOMENT_ORDER,
                       wavelet_constants.MAXIMUM_MOMENT_ORDER + 1):
        width = order + 1
        column = np.abs(vectors[:, index:index + width])
        orders[f"order_{order}_median_abs"] = float(f"{np.median(column):.6g}")
        index += width
    return orders


def deep_probe(image: np.ndarray, metadata: ImageMetadata,
               output: EngineOutput, image_path: str) -> dict:
    """Collect the internals the verdict rests on, plus two independent checks.

    Args:
        image: BGR uint8 image.
        metadata: Real metadata for this image.
        output: The EngineOutput already produced for this image.
        image_path: Path on disk (unused here; kept for signature parity).

    Returns:
        Dictionary of the internal quantities.
    """
    probe = {"steps": {step["name"]: step["key_values"]
                       for step in output.computation_steps}}
    for label, function in (("feature_space", feature_space_anatomy),
                            ("feature_scale_by_order", per_order_feature_scale)):
        try:
            probe[label] = function(image)
        except Exception as error:  # noqa: BLE001 - probe must not kill the run
            probe[label] = {"error": f"{type(error).__name__}: {error}"}
    return probe


# ---------------------------------------------------------------------------
# Output persistence
# ---------------------------------------------------------------------------

def save_evidence_map(evidence_map: Optional[np.ndarray],
                      output_path: str) -> tuple:
    """Write the evidence map as a plain PNG and a colorized PNG.

    Args:
        evidence_map: BGR uint8 array from the engine, or None.
        output_path: Destination path for the plain PNG.

    Returns:
        Tuple of (plain path, colour path); empty strings when nothing was
        rendered.
    """
    if evidence_map is None:
        return "", ""

    array = np.asarray(evidence_map)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, array)

    grey = (cv2.cvtColor(array, cv2.COLOR_BGR2GRAY) if array.ndim == 3
            else array.astype(np.uint8))
    spread = float(grey.max()) - float(grey.min())
    normalised = (np.zeros_like(grey, dtype=np.uint8) if spread <= 0 else
                  ((grey.astype(np.float64) - float(grey.min())) * 255.0
                   / spread).astype(np.uint8))
    colour_path = output_path.replace(".png", "_color.png")
    cv2.imwrite(colour_path, cv2.applyColorMap(normalised, cv2.COLORMAP_JET))
    return output_path, colour_path


def save_report_json(report: dict, output_path: str) -> None:
    """Write the structured report as formatted JSON.

    Args:
        report: Report dictionary.
        output_path: Destination path.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)


def _format_image_block(image: dict) -> list:
    """Render one image's section of the human-readable report.

    Args:
        image: Serialised ImageResult.

    Returns:
        List of report lines.
    """
    facts = image["facts"]
    probability = image["probability"]
    lines = [
        "=" * 74,
        f"IMAGE: {facts['filename']}   [{image['case_type']}]",
        f"PATH:  {facts['path']}",
        "=" * 74,
        f"  Resolution   : {facts['width']}x{facts['height']}",
        f"  Format       : {facts['format']}",
        f"  File size    : {facts['file_size_bytes'] / 1024.0:.1f} KB",
        f"  Est. quality : {facts['estimated_quality_factor']:.0f}  "
        f"({facts['quality_estimation_method']})",
        f"  Has EXIF     : {facts['has_exif']}",
        f"  Is resized   : {facts['is_resized']}  ({facts['resize_evidence']})",
        "",
        f"  Raw score    : {image['raw_score']:.6f}",
        f"  Probability  : "
        f"{'None' if probability is None else f'{probability:.6f}'}",
        f"  Confidence   : {image['confidence']:.4f}",
        f"  Is reliable  : {image['is_reliable']}",
        f"  Time         : {image['processing_time_ms']:.1f} ms",
        f"  Reliability  : {image['reliability_note']}",
        "",
        "  COMPUTATION TRACE:",
    ]
    for step in image["computation_steps"]:
        lines.append(f"    Step {step['step']}: {step['name']}")
        lines.append(f"      {step['description']}")
        lines.append(f"      in={step['input_shape']}  out={step['output_shape']}")
        for key, value in step["key_values"].items():
            lines.append(f"        {key} = {value}")
    lines += [
        "",
        "  REASONING:",
        f"    {image['reasoning']}",
        "",
        "  MATHEMATICAL EXPLANATION:",
        f"    {image['mathematical_explanation']}",
        "",
        f"  Engine says   : {image['engine_verdict']}",
        f"  Ground truth  : {image['ground_truth']}",
        f"  Correct       : {image['verdict_correct']}",
        f"  Surprise level: {image['surprise_level']}",
        f"  Evidence map  : {image['evidence_map_path'] or 'none'}",
        "",
    ]
    return lines


def save_report_text(report: dict, output_path: str) -> None:
    """Write the report in human-readable form.

    Args:
        report: Report dictionary.
        output_path: Destination path.
    """
    lines = [
        "=" * 74,
        f"WAVELET-DOMAIN ENGINE DIAGNOSTIC REPORT  [{report['tag']}]",
        f"Generated: {report['generated_at']}",
        "=" * 74,
        "",
        report["pre_run_table"],
        "",
    ]
    if report["notices"]:
        lines.append("NOTICES:")
        lines += [f"  - {notice}" for notice in report["notices"]]
        lines.append("")

    for case in report["cases"]:
        lines += ["", "#" * 74,
                  f"CASE {case['case_number']}: {case['case_type']} "
                  f"(ground truth {case['ground_truth']})",
                  f"folder: {case['folder']}", "#" * 74, ""]
        for image in case["images"]:
            lines += _format_image_block(image)
        average = case["average_probability"]
        lines += [
            f"  CASE {case['case_number']} SUMMARY",
            f"    average probability : "
            f"{'n/a' if average is None else f'{average:.4f}'}",
            f"    score spread        : "
            f"{'n/a' if case['score_spread'] is None else f'{case_and_spread(case)}'}",
            f"    correct verdicts    : {case['correct_verdicts']} / "
            f"{len(case['images'])}",
            f"    case verdict        : {case['case_verdict']}",
            f"    {case['summary']}",
            "",
        ]

    statistics = report["statistics"]
    lines += [
        "", "=" * 74, "OVERALL", "=" * 74,
        f"  real average probability     : {statistics['real_average']}",
        f"  manipulated average          : {statistics['fake_average']}",
        f"  ai-generated average         : {statistics['ai_average']}",
        f"  separation (fake - real)     : {statistics['separation']}",
        f"  raw-score separation         : {statistics['raw_separation']}",
        f"  reliable results             : {statistics['reliable_count']} / "
        f"{statistics['total_images']}",
        f"  average processing time (ms) : {statistics['average_time_ms']}",
        "",
    ]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def case_and_spread(case: dict) -> str:
    """Render a case's score spread for the text report.

    Args:
        case: Serialised CaseResult.

    Returns:
        Formatted spread string.
    """
    return f"{case['score_spread']:.4f}"


# ---------------------------------------------------------------------------
# Reasoning
# ---------------------------------------------------------------------------

def engine_sweep_settings() -> str:
    """Describe the configuration the engine is running under.

    Returns:
        One-line description of the scored pipeline and its parameters.
    """
    from wavelet_engine import constants as wavelet_constants
    return (f"Pipeline C only is scored: single-level "
            f"'{wavelet_constants.PIPELINE_C_WAVELET_FAMILY}' DWT, LL subband "
            f"tiled into {wavelet_constants.DEFAULT_BLOCK_SIZE}x"
            f"{wavelet_constants.DEFAULT_BLOCK_SIZE} blocks at stride "
            f"{wavelet_constants.BLOCK_STRIDE_PIXELS}; blur invariants for all "
            f"(p,q) with {wavelet_constants.MINIMUM_MOMENT_ORDER}<=p+q<="
            f"{wavelet_constants.MAXIMUM_MOMENT_ORDER}, contrast-normalised "
            f"(Eq. 17), PCA to "
            f"{wavelet_constants.PCA_EXPLAINED_VARIANCE_TARGET:.0%} explained "
            f"variance; candidate pairs at S=1/(1+rho) >= "
            f"{wavelet_constants.SIMILARITY_THRESHOLD} (Euclidean radius "
            f"{1.0 / wavelet_constants.SIMILARITY_THRESHOLD - 1.0:.6f}), "
            f"confirmed by {wavelet_constants.NEIGHBOUR_CHECK_COUNT} neighbour "
            f"offsets within {wavelet_constants.NEIGHBOUR_CHECK_MAX_OFFSET_PIXELS} "
            f"px and a minimum separation of "
            f"{wavelet_constants.MINIMUM_SEPARATION_BLOCK_MULTIPLE} block "
            f"widths; raw_score = fraction of blocks in a confirmed pair; "
            f"probability via a PROVISIONAL logistic (midpoint "
            f"{wavelet_constants.PROVISIONAL_SIGMOID_MIDPOINT}, slope "
            f"{wavelet_constants.PROVISIONAL_SIGMOID_SLOPE}). Pipelines A and B "
            f"run unscored")


def classify_verdict(probability: Optional[float], is_reliable: bool) -> str:
    """Convert a probability into a REAL / UNCERTAIN / FAKE label.

    Args:
        probability: Calibrated probability, or None.
        is_reliable: Whether the engine stood behind the measurement.

    Returns:
        Verdict label.
    """
    if not is_reliable or probability is None:
        return "NO VOTE (unreliable)"
    if probability < REAL_VERDICT_CEILING:
        return "REAL"
    if probability > FAKE_VERDICT_FLOOR:
        return "FAKE"
    return "UNCERTAIN"


def analyze_result(engine_output: EngineOutput,
                   facts: ImageFacts,
                   case_type: str,
                   ground_truth: str,
                   expectation: dict,
                   probe: dict) -> ImageResult:
    """Assess one engine output against what the SKILL file predicted.

    Args:
        engine_output: The engine's result.
        facts: Pre-run measurements of the file.
        case_type: Case label.
        ground_truth: REAL, FAKE or AI-GENERATED.
        expectation: Predicted probability band for this case.
        probe: Per-cell grid detail from deep_probe.

    Returns:
        Fully populated ImageResult.
    """
    verdict = classify_verdict(engine_output.probability,
                               engine_output.is_reliable)
    expected_truth = "FAKE" if ground_truth != "REAL" else "REAL"
    correct = verdict == expected_truth

    probability = engine_output.probability
    low, high = expectation["probability_range"]
    within_band = (probability is not None and low <= probability <= high)
    if verdict.startswith("NO VOTE"):
        # An engine that declines to vote has not got the answer wrong. It is
        # scored separately so abstention is never counted as a correct call
        # either.
        surprise = "ABSTAINED"
    elif correct and within_band:
        surprise = "EXPECTED"
    elif within_band or correct:
        surprise = "SURPRISING"
    else:
        surprise = "WRONG"

    steps = probe.get("steps", {})
    pipeline_c = next((v for k, v in steps.items() if k.startswith("Pipeline C")), {})
    pipeline_a = next((v for k, v in steps.items() if k.startswith("Pipeline A")), {})
    pipeline_b = next((v for k, v in steps.items() if k.startswith("Pipeline B")), {})
    space = probe.get("feature_space", {})
    by_order = probe.get("feature_scale_by_order", {})

    reasoning = (
        f"Engine reported raw_score={engine_output.raw_score:.6f} at confidence "
        f"{engine_output.confidence:.4f}, is_reliable={engine_output.is_reliable}, "
        f"flagged_regions="
        f"{0 if engine_output.flagged_regions is None else len(engine_output.flagged_regions)}"
        f" confirmed pair(s). Pipeline C reported {pipeline_c}; Pipeline A "
        f"(unscored) {pipeline_a}; Pipeline B (unscored, low-trust) "
        f"{pipeline_b}. The LL subband is {space.get('ll_shape')} and tiled into "
        f"{space.get('total_blocks')} overlapping blocks. Eq. 27's threshold of "
        f"{0.95} corresponds to a maximum Euclidean distance of "
        f"{space.get('match_radius_from_threshold')} in the "
        f"{space.get('feature_dimension')}-dimensional invariant space, but the "
        f"measured pairwise distances over {space.get('sampled_blocks')} sampled "
        f"blocks run min {space.get('pair_distance_min')}, median "
        f"{space.get('pair_distance_median')}, max {space.get('pair_distance_max')} "
        f"- a gap of {space.get('orders_of_magnitude_gap')} orders of magnitude - "
        f"and {space.get('sampled_pairs_inside_radius')} of "
        f"{space.get('sampled_pairs')} sampled pairs fall inside it. Feature "
        f"magnitude by moment order: {by_order}."
    )
    mathematical_explanation = (
        f"Pipeline C's premise, from Kashyap & Joshi via the SKILL, is that a "
        f"copied region keeps its blur-invariant moments through the paste, so "
        f"two distant blocks with matching invariants are a duplication. The "
        f"engine builds those invariants by the recursion of Eq. 12 and "
        f"contrast-normalises them by Eq. 17's (R/2)^r * mu_00. For this image "
        f"that normalisation does not bring the orders onto a common scale - "
        f"the median |invariant| per moment order is {by_order} - so the "
        f"highest orders dominate every Euclidean distance and the population "
        f"spans {space.get('feature_abs_min')} to {space.get('feature_abs_max')}. "
        f"Eq. 27's similarity S = 1/(1+rho) is a function of an ABSOLUTE "
        f"distance, so applying the unsourced threshold "
        f"{0.95} to that unnormalised space compares a cutoff of "
        f"{space.get('match_radius_from_threshold')} against typical distances of "
        f"{space.get('pair_distance_median')}, which is why raw_score came out "
        f"{engine_output.raw_score:.6f} with "
        f"{pipeline_c.get('confirmed_pairs')} confirmed pairs."
    )

    return ImageResult(
        facts=asdict(facts),
        case_type=case_type,
        ground_truth=ground_truth,
        raw_score=float(engine_output.raw_score),
        probability=(None if probability is None else float(probability)),
        confidence=float(engine_output.confidence),
        is_reliable=bool(engine_output.is_reliable),
        reliability_note=engine_output.reliability_note,
        processing_time_ms=float(engine_output.processing_time_ms),
        engine_verdict=verdict,
        verdict_correct=bool(correct),
        surprise_level=surprise,
        reasoning=reasoning,
        mathematical_explanation=mathematical_explanation,
        computation_steps=engine_output.computation_steps,
        deep_probe=probe,
    )


def run_case(engine: WaveletEngine,
             image_folder: str,
             case_number: int,
             case_type: str,
             ground_truth: str,
             expectation: dict,
             output_folder: str,
             preferred: tuple = ()) -> CaseResult:
    """Run every image of one case and assemble the case result.

    Args:
        engine: Configured engine instance.
        image_folder: Folder to draw images from.
        case_number: 1, 2 or 3.
        case_type: Human label for the case.
        ground_truth: REAL, FAKE or AI-GENERATED.
        expectation: Predicted probability band.
        output_folder: Where evidence maps are written.
        preferred: Filenames to prefer when picking images.

    Returns:
        CaseResult with both images analysed.
    """
    print(f"\n--- CASE {case_number}: {case_type} ---")
    result = CaseResult(case_number=case_number, case_type=case_type,
                        ground_truth=ground_truth, folder=image_folder)

    for index, (path, cv_image, pil_image) in enumerate(
            load_images_from_folder(image_folder, 2, preferred), start=1):
        metadata, facts = compute_metadata(path, pil_image)
        print(f"  [{index}] {facts.filename}  {facts.width}x{facts.height} "
              f"QF~{facts.estimated_quality_factor:.0f}")

        run = run_engine_safely(engine, cv_image, metadata)
        if not run["ok"]:
            print(f"      ENGINE RAISED: {run['error'].splitlines()[0]}")
            failure = ImageResult(
                facts=asdict(facts), case_type=case_type,
                ground_truth=ground_truth, raw_score=0.0, probability=None,
                confidence=0.0, is_reliable=False,
                reliability_note="harness caught an exception",
                processing_time_ms=run["wall_clock_ms"],
                engine_verdict="ERROR", verdict_correct=False,
                surprise_level="WRONG",
                reasoning="engine raised before producing output",
                mathematical_explanation="no computation completed",
                error=run["error"])
            result.images.append(failure)
            continue

        output = run["output"]
        probe = deep_probe(cv_image, metadata, output, path)
        analysed = analyze_result(output, facts, case_type, ground_truth,
                                  expectation, probe)

        stem = (f"wavelet_case{case_number}_img{index}_"
                f"{Path(facts.filename).stem.replace(' ', '_')}")
        plain, colour = save_evidence_map(
            output.evidence_map, str(Path(output_folder) / f"{stem}_heatmap.png"))
        analysed.evidence_map_path = plain
        analysed.evidence_map_colour_path = colour
        result.images.append(analysed)

        print(f"      raw={analysed.raw_score:.6f}  "
              f"prob={'None' if analysed.probability is None else f'{analysed.probability:.4f}'}"
              f"  conf={analysed.confidence:.3f}  "
              f"reliable={analysed.is_reliable}  "
              f"verdict={analysed.engine_verdict}  "
              f"({analysed.surprise_level})")

    probabilities = [image.probability for image in result.images
                     if image.probability is not None]
    if probabilities:
        result.average_probability = float(np.mean(probabilities))
        result.score_spread = float(max(probabilities) - min(probabilities))
    result.correct_verdicts = sum(1 for image in result.images
                                  if image.verdict_correct)
    return result


def generate_full_report(case_results: list,
                         pre_run_table: str,
                         notices: list,
                         tag: str) -> dict:
    """Compute overall statistics and assemble the structured report.

    Args:
        case_results: One CaseResult per case.
        pre_run_table: Rendered pre-run summary table.
        notices: Harness-level warnings to surface in every report.
        tag: Run label, e.g. "before" or "after".

    Returns:
        Report dictionary ready for JSON and text serialisation.
    """
    def average_probability(case: CaseResult) -> Optional[float]:
        values = [image.probability for image in case.images
                  if image.probability is not None]
        return float(np.mean(values)) if values else None

    def average_raw(case: CaseResult) -> Optional[float]:
        values = [image.raw_score for image in case.images]
        return float(np.mean(values)) if values else None

    by_truth = {case.ground_truth: case for case in case_results}
    real_average = average_probability(by_truth["REAL"])
    fake_average = average_probability(by_truth["FAKE"])
    ai_average = average_probability(by_truth["AI-GENERATED"])
    real_raw = average_raw(by_truth["REAL"])
    fake_raw = average_raw(by_truth["FAKE"])

    all_images = [image for case in case_results for image in case.images]
    separation = (None if real_average is None or fake_average is None
                  else fake_average - real_average)
    raw_separation = (None if real_raw is None or fake_raw is None
                      else fake_raw - real_raw)

    return {
        "tag": tag,
        "engine": "wavelet_domain_forgery",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pre_run_table": pre_run_table,
        "notices": notices,
        "cases": [asdict(case) for case in case_results],
        "statistics": {
            "total_images": len(all_images),
            "real_average": real_average,
            "fake_average": fake_average,
            "ai_average": ai_average,
            "separation": separation,
            "real_raw_average": real_raw,
            "fake_raw_average": fake_raw,
            "raw_separation": raw_separation,
            "reliable_count": sum(1 for image in all_images if image.is_reliable),
            "correct_verdicts": sum(1 for image in all_images
                                    if image.verdict_correct),
            "abstained_count": sum(1 for image in all_images
                                   if image.surprise_level == "ABSTAINED"),
            "wrong_verdicts": sum(1 for image in all_images
                                  if image.surprise_level == "WRONG"),
            "average_time_ms": float(np.mean(
                [image.processing_time_ms for image in all_images])),
        },
    }


def save_summary(report: dict, output_path: str) -> None:
    """Write the one-page summary that is read first next session.

    Args:
        report: Report dictionary.
        output_path: Destination path.
    """
    statistics = report["statistics"]
    lines = [
        "WAVELET-DOMAIN FORGERY ENGINE - DIAGNOSTIC SUMMARY",
        f"run tag       : {report['tag']}",
        f"generated     : {report['generated_at']}",
        "",
        f"images tested : {statistics['total_images']}",
        f"reliable      : {statistics['reliable_count']} / "
        f"{statistics['total_images']}",
        f"correct       : {statistics['correct_verdicts']} / "
        f"{statistics['total_images']}",
        f"wrong         : {statistics['wrong_verdicts']} / "
        f"{statistics['total_images']}",
        f"abstained     : {statistics['abstained_count']} / "
        f"{statistics['total_images']}",
        "",
        "PROBABILITY BY CASE",
        f"  real        : {statistics['real_average']}",
        f"  manipulated : {statistics['fake_average']}",
        f"  ai-generated: {statistics['ai_average']}",
        f"  separation  : {statistics['separation']}",
        "",
        "RAW SCORE BY CASE",
        f"  real        : {statistics['real_raw_average']}",
        f"  manipulated : {statistics['fake_raw_average']}",
        f"  separation  : {statistics['raw_separation']}",
        "",
        "NOTICES",
    ]
    lines += [f"  - {notice}" for notice in report["notices"]]
    lines += ["", "PER-IMAGE"]
    for case in report["cases"]:
        for image in case["images"]:
            probability = image["probability"]
            lines.append(
                f"  {image['facts']['filename']:<22} truth="
                f"{image['ground_truth']:<13} raw={image['raw_score']:.4f} "
                f"prob="
                f"{'None' if probability is None else f'{probability:.4f}'} "
                f"conf={image['confidence']:.3f} "
                f"verdict={image['engine_verdict']}"
            )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> int:
    """Run all three cases and persist every output.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="before",
                        help="output subfolder name for this run")
    arguments = parser.parse_args()

    output_folder = str(Path(OUTPUT_ROOT) / arguments.tag)
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    notices = []
    ai_folder = AI_IMAGES_FOLDER
    if not Path(ai_folder).is_dir():
        ai_folder = FAKE_IMAGES_FOLDER
        notices.append(
            f"The AI-generated images folder named in the specification does "
            f"not exist ({AI_IMAGES_FOLDER}). Case 3 falls back to the two "
            f"files inside the fake-images folder whose names denote AI "
            f"generation: {', '.join(AI_SUBSTITUTE_FILENAMES)}. Their AI "
            f"provenance is inferred from the filenames, not verified.")

    # Phase 0 - inspect everything before the engine runs.
    print("=" * 74)
    print("PHASE 0 - PRE-RUN IMAGE SUMMARY")
    print("=" * 74)
    pre_run_entries = []
    for folder, label in ((REAL_IMAGES_FOLDER, "Real"),
                          (FAKE_IMAGES_FOLDER, "Fake/AI")):
        for path, _, pil_image in load_images_from_folder(folder, 99,
                                                          require_all=False):
            _, facts = compute_metadata(path, pil_image)
            pre_run_entries.append((facts, label))
    pre_run_table = print_pre_run_summary(pre_run_entries)

    notices.append(f"Engine configuration: {engine_sweep_settings()}.")
    notices.append(
        "The SKILL scopes this engine's score to Pipeline C alone: Pipeline "
        "A's output is 'not a scalar score on its own', and Pipeline B is "
        "documented as 100%-defeatable by a knowledgeable adversary (Stamm & "
        "Liu 2010) and is 'low-trust by default'. Both run here but never "
        "affect raw_score.")
    notices.append(
        "Pipeline C's summary scalar is the SKILL's own engineering "
        "recommendation, not a corpus value: 'not explicitly defined in the "
        "source paper as a summary scalar'. Its calibration is likewise "
        "provisional.")

    engine = WaveletEngine()
    expectations = {
        "REAL": {"probability_range": (0.0, 0.30)},
        "FAKE": {"probability_range": (0.60, 1.0)},
        "AI-GENERATED": {"probability_range": (0.60, 1.0)},
    }

    cases = [
        run_case(engine, REAL_IMAGES_FOLDER, 1, "Case 1 Real", "REAL",
                 expectations["REAL"], output_folder),
        run_case(engine, FAKE_IMAGES_FOLDER, 2, "Case 2 Manipulated", "FAKE",
                 expectations["FAKE"], output_folder,
                 preferred=CASE_2_PREFERRED_FILENAMES),
        run_case(engine, ai_folder, 3, "Case 3 AI-generated", "AI-GENERATED",
                 expectations["AI-GENERATED"], output_folder,
                 preferred=AI_SUBSTITUTE_FILENAMES),
    ]

    report = generate_full_report(cases, pre_run_table, notices, arguments.tag)
    save_report_json(report, str(Path(output_folder) / "test_results.json"))
    save_report_text(report, str(Path(output_folder) / "test_report.txt"))
    save_summary(report, str(Path(output_folder) / "summary.txt"))

    statistics = report["statistics"]
    print("\n" + "=" * 74)
    print(f"real avg prob = {statistics['real_average']}")
    print(f"fake avg prob = {statistics['fake_average']}")
    print(f"ai   avg prob = {statistics['ai_average']}")
    print(f"separation    = {statistics['separation']}")
    print(f"raw separation= {statistics['raw_separation']}")
    print(f"written to    : {output_folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
