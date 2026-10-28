"""Parity gates must fail closed on incomplete or nonfinite output records."""
from copy import deepcopy
import json

import pytest

from deepstarnoodles_inference.validation import compare


def arm(index, offset=0):
    points = [[10.0 + offset, 20.0], [30.0 + offset, 40.0], [50.0 + offset, 60.0]]
    return {"arm_index": index, "polyline_crop": points, "polyline_source": deepcopy(points),
            "identity": [0.2 + index * 0.1, 0.3, 0.4, 0.5], "stop_reason": "base", "steps": 7,
            "tip_score": 0.8, "base_score": 0.7, "state_confidence": 0.9, "length_px": 56.5685}


@pytest.fixture
def records():
    stars = []
    for index, offset in enumerate((0.0, 100.0)):
        stars.append({"detection_index": index, "star_track_id": None,
            "crop": {"left": offset, "top": 0.0, "side": 512.0, "confidence": 0.875,
                     "bbox_xyxy": [offset, 0.0, offset + 40.0, 40.0]},
            "mask_polygon_source": [[offset, 0.0], [offset + 40.0, 0.0], [offset + 40.0, 40.0]],
            "arms": [arm(0), arm(1, 100.0)]})
        for current in stars[-1]["arms"]:
            for point in current["polyline_source"]:
                point[0] += offset
    return [{"frame": 0, "image_path": "/inputs/frame.png", "source_shape": [512, 1024], "stars": stars}]


def gate(tmp_path, reference, candidate):
    left, right = tmp_path / "reference.jsonl", tmp_path / "candidate.jsonl"
    # JSON's permissive NaN encoding deliberately emulates malformed producer
    # output; a validator must reject it rather than trusting serialization.
    for path, values in ((left, reference), (right, candidate)):
        path.write_text("".join(json.dumps(value) + "\n" for value in values))
    return compare(left, right)


def test_identical_outputs_and_reordered_detections_arms_pass(tmp_path, records):
    candidate = deepcopy(records)
    candidate[0]["stars"].reverse()
    for index, star in enumerate(candidate[0]["stars"]):
        star["detection_index"] = index
        star["arms"].reverse()
        star["mask_polygon_source"].reverse()
        for arm_index, current in enumerate(star["arms"]):
            current["arm_index"] = arm_index
    result = gate(tmp_path, records, candidate)
    assert result["pass"] is True
    assert result["issue_count"] == 0


@pytest.mark.parametrize("level", ["source", "star", "arm"])
def test_changed_counts_fail(tmp_path, records, level):
    candidate = deepcopy(records)
    if level == "source":
        candidate.append(deepcopy(candidate[0]))
    elif level == "star":
        candidate[0]["stars"].pop()
    else:
        candidate[0]["stars"][0]["arms"].pop()
    assert gate(tmp_path, records, candidate)["pass"] is False


@pytest.mark.parametrize("field", ["bbox_xyxy", "left", "side", "polyline_crop", "polyline_source",
                                   "identity", "confidence", "steps", "stop_reason", "length_px"])
def test_changed_geometry_identity_or_decoder_values_fail(tmp_path, records, field):
    candidate = deepcopy(records)
    star = candidate[0]["stars"][0]
    current = star["arms"][0]
    if field == "bbox_xyxy":
        star["crop"][field][0] += 10.0
    elif field in ("left", "side", "confidence"):
        star["crop"][field] += 10.0
    elif field.startswith("polyline_"):
        current[field][1][0] += 10.0
    elif field == "identity":
        current[field][0] += 0.5
    elif field == "stop_reason":
        current[field] = "lost"
    else:
        current[field] += 10
    assert gate(tmp_path, records, candidate)["pass"] is False


@pytest.mark.parametrize("field", ["tip_score", "base_score", "state_confidence", "length_px",
                                   "steps", "stop_reason", "polyline_source", "identity"])
def test_omitted_arm_keys_fail_without_crashing(tmp_path, records, field):
    candidate = deepcopy(records)
    del candidate[0]["stars"][0]["arms"][0][field]
    assert gate(tmp_path, records, candidate)["pass"] is False


@pytest.mark.parametrize("field", ["bbox_xyxy", "confidence", "side"])
def test_omitted_crop_keys_fail_without_crashing(tmp_path, records, field):
    candidate = deepcopy(records)
    del candidate[0]["stars"][0]["crop"][field]
    assert gate(tmp_path, records, candidate)["pass"] is False


@pytest.mark.parametrize("field", ["bbox_xyxy", "polyline_crop", "polyline_source", "identity",
                                   "confidence", "mask_polygon_source", "length_px"])
def test_nonfinite_values_in_either_output_fail_without_crashing(tmp_path, records, field):
    # Corrupt both sides: matching invalid values are still invalid evidence.
    reference = deepcopy(records)
    star = reference[0]["stars"][0]
    current = star["arms"][0]
    if field in ("bbox_xyxy", "confidence"):
        if field == "bbox_xyxy":
            star["crop"][field][0] = float("nan")
        else:
            star["crop"][field] = float("nan")
    elif field == "mask_polygon_source":
        star[field][0][0] = float("nan")
    elif field.startswith("polyline_"):
        current[field][0][0] = float("nan")
    elif field == "identity":
        current[field][0] = float("nan")
    else:
        current[field] = float("nan")
    assert gate(tmp_path, reference, deepcopy(reference))["pass"] is False


def test_identity_shape_cannot_broadcast_to_false_parity(tmp_path, records):
    reference = deepcopy(records)
    reference[0]["stars"][0]["arms"][0]["identity"] = [0.2]
    candidate = deepcopy(reference)
    candidate[0]["stars"][0]["arms"][0]["identity"] = [0.2, 0.2]
    assert gate(tmp_path, reference, candidate)["pass"] is False


@pytest.mark.parametrize("field", ["source_shape", "image_path", "mask_polygon_source", "steps"])
def test_missing_required_fields_on_both_sides_do_not_count_as_parity(tmp_path, records, field):
    reference = deepcopy(records)
    if field in ("source_shape", "image_path"):
        del reference[0][field]
    elif field == "mask_polygon_source":
        del reference[0]["stars"][0][field]
    else:
        del reference[0]["stars"][0]["arms"][0][field]
    assert gate(tmp_path, reference, deepcopy(reference))["pass"] is False


def test_empty_files_are_not_successful_validation(tmp_path):
    assert gate(tmp_path, [], [])["pass"] is False


def test_empty_frames_are_valid_when_mapping_is_complete(tmp_path, records):
    reference = deepcopy(records)
    reference[0]["stars"] = []
    assert gate(tmp_path, reference, deepcopy(reference))["pass"] is True


def test_changed_source_mapping_fails(tmp_path, records):
    candidate = deepcopy(records)
    candidate[0]["image_path"] = "/inputs/a_different_frame.png"
    assert gate(tmp_path, records, candidate)["pass"] is False
