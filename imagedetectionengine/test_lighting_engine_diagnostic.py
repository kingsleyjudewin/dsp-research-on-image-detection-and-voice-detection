"""Diagnostic test harness for the lighting / illumination engine.

Runs the engine over three cases (authentic camera images, manipulated images,
AI-generated images), records the full computation trace for every image, and
writes every result to disk so the findings survive the session.

Nothing here is part of the engine. This file only measures it.

Usage:
    python3 test_lighting_engine_diagnostic.py [--tag before|after]

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

from lighting_engine.contracts import EngineInput, EngineOutput, ImageMetadata
from lighting_engine.engine import LightingEngine


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
    "imagedetectionengine/test results/lighting_engine"
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

def run_engine_safely(engine: LightingEngine,
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


def gradient_percentiles(grayscale: np.ndarray) -> dict:
    """Describe how extreme max_grad is within its own population.

    raw_score is one pixel - the single largest gradient magnitude in the frame
    - divided by the population median. This records where that one pixel sits
    relative to the rest of the distribution, and where in the image it is, so a
    reader can judge how much of the score rests on a single sample.

    Args:
        grayscale: Float64 single-plane image.

    Returns:
        Dictionary of percentile values, the argmax location, and the ratio.
    """
    from lighting_engine.utils import compute_gradient_magnitude

    magnitude = compute_gradient_magnitude(grayscale)
    flat = magnitude.ravel()
    row, column = np.unravel_index(int(np.argmax(magnitude)), magnitude.shape)
    median = float(np.median(flat))
    percentiles = {f"p{q}": round(float(np.percentile(flat, q)), 4)
                   for q in (50, 90, 99, 99.9, 99.99)}
    return {
        **percentiles,
        "max": round(float(flat.max()), 4),
        "argmax_row": int(row),
        "argmax_col": int(column),
        "height": int(magnitude.shape[0]),
        "width": int(magnitude.shape[1]),
        "on_border": bool(row in (0, magnitude.shape[0] - 1)
                          or column in (0, magnitude.shape[1] - 1)),
        "ratio_max_over_median": round(float(flat.max() / max(median, 1e-9)), 4),
        "ratio_p9999_over_median": round(
            float(np.percentile(flat, 99.99) / max(median, 1e-9)), 4),
        "pixels_within_1pct_of_max": int(
            np.count_nonzero(flat >= 0.99 * flat.max())),
    }


def nuisance_sensitivity(image: np.ndarray, metadata: ImageMetadata) -> dict:
    """Measure how much raw_score moves under changes that alter NO lighting.

    The SKILL's own honest assessment of this statistic is that it is "at most a
    generic edge/gradient-strength feature ... not specifically
    lighting-direction inconsistency". This quantifies that: each transform
    below leaves the scene's illumination untouched and changes only how sharply
    it is sampled or coded. Whatever the score does here, it is not doing on
    account of lighting.

    Args:
        image: BGR uint8 image.
        metadata: Real metadata for this image.

    Returns:
        Dictionary of raw_score under each transform, and the spread.
    """
    from lighting_engine.engine import LightingEngine as _Engine

    def score(array: np.ndarray) -> float:
        facts = ImageMetadata(
            estimated_compression_level=metadata.estimated_compression_level,
            is_resized=metadata.is_resized, color_space=metadata.color_space,
            resolution=array.shape[:2], format=metadata.format,
            has_exif=metadata.has_exif)
        return float(_Engine().analyse(
            EngineInput(image=array, metadata=facts)).raw_score)

    height, width = image.shape[:2]
    variants = {
        "original": image,
        "downscale_0.5x": cv2.resize(image, (width // 2, height // 2),
                                     interpolation=cv2.INTER_AREA),
        "upscale_2x": cv2.resize(image, (width * 2, height * 2),
                                 interpolation=cv2.INTER_LINEAR),
        "gaussian_blur_3": cv2.GaussianBlur(image, (3, 3), 0),
        "unsharp_mask": cv2.addWeighted(image, 1.8,
                                        cv2.GaussianBlur(image, (0, 0), 3),
                                        -0.8, 0),
    }
    recoded = cv2.imdecode(
        cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 40])[1],
        cv2.IMREAD_COLOR)
    variants["jpeg_q40"] = recoded

    results = {}
    for label, array in variants.items():
        try:
            results[label] = round(score(array), 4)
        except Exception as error:  # noqa: BLE001 - probe must not kill the run
            results[label] = f"{type(error).__name__}: {error}"
    numeric = [v for v in results.values() if isinstance(v, float)]
    if numeric:
        results["spread_max_minus_min"] = round(max(numeric) - min(numeric), 4)
        results["ratio_max_over_min"] = round(max(numeric) / max(min(numeric), 1e-9), 4)
    return results


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
    from lighting_engine.preprocessor import LightingPreprocessor

    probe = {"steps": {step["name"]: step["key_values"]
                       for step in output.computation_steps}}
    try:
        prepared = LightingPreprocessor().prepare(image)
        probe["gradient_distribution"] = gradient_percentiles(prepared.grayscale)
    except Exception as error:  # noqa: BLE001
        probe["gradient_distribution"] = {"error": f"{type(error).__name__}: {error}"}
    try:
        probe["nuisance_sensitivity"] = nuisance_sensitivity(image, metadata)
    except Exception as error:  # noqa: BLE001
        probe["nuisance_sensitivity"] = {"error": f"{type(error).__name__}: {error}"}
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
        f"LIGHTING ENGINE DIAGNOSTIC REPORT  [{report['tag']}]",
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
        One-line description of the scoring route and the confidence ceiling.
    """
    from lighting_engine import constants as lighting_constants
    return (f"Pipeline A only (Sobel gradient magnitude); raw_score = "
            f"max_grad / median(gradient_mag); probability via a PROVISIONAL "
            f"logistic with midpoint "
            f"{lighting_constants.PROVISIONAL_SIGMOID_MIDPOINT} and slope "
            f"{lighting_constants.PROVISIONAL_SIGMOID_SLOPE} that approximates "
            f"no corpus threshold; confidence hard-capped at "
            f"{lighting_constants.MAXIMUM_CONFIDENCE_CEILING} per the SKILL's "
            f"own Corpus Gap instruction. Pipelines B and C not implemented")


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

    from lighting_engine import constants as lighting_constants

    steps = probe.get("steps", {})
    gradient = next((v for k, v in steps.items() if "Gradient" in k), {})
    calibration = next((v for k, v in steps.items() if "Calibration" in k), {})
    distribution = probe.get("gradient_distribution", {})
    nuisance = probe.get("nuisance_sensitivity", {})

    reasoning = (
        f"Engine reported raw_score={engine_output.raw_score:.6f} - the single "
        f"largest gradient magnitude in the frame divided by that frame's "
        f"median gradient magnitude - at confidence "
        f"{engine_output.confidence:.4f}, is_reliable="
        f"{engine_output.is_reliable}, flagged_regions="
        f"{engine_output.flagged_regions}. Gradient stage reported {gradient}. "
        f"Calibration stage reported {calibration}. The whole score rests on "
        f"ONE pixel: the maximum sits at row {distribution.get('argmax_row')}, "
        f"column {distribution.get('argmax_col')} of a "
        f"{distribution.get('height')}x{distribution.get('width')} field "
        f"(on_border={distribution.get('on_border')}), and only "
        f"{distribution.get('pixels_within_1pct_of_max')} pixel(s) in the "
        f"entire image lie within 1% of it. The gradient distribution runs "
        f"p50={distribution.get('p50')}, p99={distribution.get('p99')}, "
        f"p99.9={distribution.get('p99.9')}, max={distribution.get('max')}, so "
        f"max/median={distribution.get('ratio_max_over_median')} against a "
        f"far steadier p99.99/median of "
        f"{distribution.get('ratio_p9999_over_median')}. Under transformations "
        f"that change NO lighting whatsoever the same score moves to "
        f"{nuisance}."
    )
    mathematical_explanation = (
        f"This engine computes sqrt(Gx^2 + Gy^2) per pixel and takes its "
        f"maximum, exactly as the source paper's MATLAB snippet does, then "
        f"divides by the image's own median gradient magnitude because the "
        f"SKILL's Output section recommends normalising per-image rather than "
        f"using an absolute cutoff. For this image that gives "
        f"{engine_output.raw_score:.6f}. What the number cannot do is speak to "
        f"illumination: the SKILL's own honest assessment is that this is 'at "
        f"most a generic edge/gradient-strength feature ... not specifically "
        f"lighting-direction inconsistency', and its Corpus Gap section states "
        f"that zero papers in the folder perform photometric lighting analysis "
        f"at all. The source's decision rule - 'if large gradients or multiple "
        f"light directions are detected, this may suggest manipulation' - is "
        f"additionally self-contradictory, because a scalar magnitude carries "
        f"no directional information from which multiple light directions "
        f"could be recovered. Confidence is therefore capped at "
        f"{lighting_constants.MAXIMUM_CONFIDENCE_CEILING} unconditionally, "
        f"which is the engine correctly refusing to present this as evidence."
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


def run_case(engine: LightingEngine,
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

        stem = (f"lighting_case{case_number}_img{index}_"
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
        "engine": "illumination_inconsistency",
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
        "LIGHTING / ILLUMINATION ENGINE - DIAGNOSTIC SUMMARY",
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
        "The SKILL file states this module has 'the thinnest evidentiary base "
        "of all nine' and 'should carry the lowest reliability weight of the "
        "nine detectors in the fusion layer'. Every number below should be "
        "read against that.")

    engine = LightingEngine()
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
