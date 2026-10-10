"""Build a self-contained two-car, four-part editor from completed Day5 caches.

No inference, model weights or later ground-truth masks are embedded. The fixed
test scene is test_crossing_6200, chosen before inspecting the test results. A
test build refuses to render it until the full 12-clip report and caches exist.
Run --self-test for fabricated CPU/JavaScript checks that touch no scene seeds.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
DAY4 = HERE.parent / "day4"
if str(DAY4) not in sys.path:
    sys.path.insert(0, str(DAY4))
from build_demo import encode_rle, decode_rle
import parent_benchmark as renderer

SCHEMA = "vjepa.hierarchical-part-edit-recipe"
PARTS = [
    {"id": "car_A.front_door", "car": "car_A", "label": "Front door", "mask_id": 1, "parent_mask_id": 101},
    {"id": "car_A.window", "car": "car_A", "label": "Window", "mask_id": 2, "parent_mask_id": 101},
    {"id": "car_B.front_door", "car": "car_B", "label": "Front door", "mask_id": 3, "parent_mask_id": 102},
    {"id": "car_B.window", "car": "car_B", "label": "Window", "mask_id": 4, "parent_mask_id": 102},
]
COLORS = [[42, 137, 240], [249, 184, 53], [80, 213, 151], [219, 102, 218]]
PARENT_INDEX = [0, 0, 1, 1]
TEST_SCENE = "test_crossing_6200"
SMOKE_SCENE = "smoke_crossing_5200"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def default_recipe(sequence):
    return {"schema": SCHEMA, "version": 1, "sequence": sequence, "containment": False,
            "parts": [{"id": part["id"], "enabled": True, "color": color.copy(), "strength": .65}
                      for part, color in zip(PARTS, COLORS)]}


def validate_recipe(recipe, sequence):
    """Validate fully and return a detached, canonical stable-ID ordering."""
    # JSON has one numeric type in JavaScript: 1 and 1.0 both pass
    # Number.isInteger. Match that behavior while excluding Python booleans.
    def json_integer(value):
        return type(value) is int or (type(value) is float and math.isfinite(value) and value.is_integer())

    if not isinstance(recipe, dict) or set(recipe) != {"schema", "version", "sequence", "containment", "parts"}:
        raise ValueError("Recipe needs exactly schema, version, sequence, containment and parts.")
    if recipe["schema"] != SCHEMA or not json_integer(recipe["version"]) or recipe["version"] != 1:
        raise ValueError("Unsupported recipe schema/version.")
    if recipe["sequence"] != sequence:
        raise ValueError("Recipe belongs to a different video sequence.")
    if type(recipe["containment"]) is not bool:
        raise ValueError("containment must be true or false.")
    if not isinstance(recipe["parts"], list) or len(recipe["parts"]) != len(PARTS):
        raise ValueError("Recipe must contain all four registered parts exactly once.")
    expected, found = {p["id"] for p in PARTS}, {}
    for part in recipe["parts"]:
        if not isinstance(part, dict) or set(part) != {"id", "enabled", "color", "strength"}:
            raise ValueError("Unexpected or missing part controls.")
        pid = part["id"]
        if not isinstance(pid, str) or pid not in expected or pid in found:
            raise ValueError("Unknown or duplicate stable part ID.")
        if type(part["enabled"]) is not bool:
            raise ValueError("enabled must be true or false.")
        if not isinstance(part["color"], list) or len(part["color"]) != 3 or any(not json_integer(c) or not 0 <= c <= 255 for c in part["color"]):
            raise ValueError("Color needs three integer RGB channels from 0 to 255.")
        strength = part["strength"]
        if type(strength) not in (int, float) or not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("Strength must be a finite number from 0 to 1.")
        found[pid] = copy.deepcopy(part)
        found[pid]["color"] = [int(c) for c in part["color"]]
    return {**recipe, "version": 1, "parts": [found[p["id"]] for p in PARTS]}


def effective_masks(children, parents, containment=False):
    """Keep the original raw overlap veto, including every disabled child."""
    children, parents = np.asarray(children), np.asarray(parents)
    if children.dtype != bool or parents.dtype != bool or children.ndim not in (3, 4) or children.shape[-3] != 4:
        raise ValueError("Expected boolean child masks [...,4,H,W].")
    if parents.shape != (*children.shape[:-3], 2, *children.shape[-2:]):
        raise ValueError("Expected aligned boolean parent masks [...,2,H,W].")
    result = children & (children.sum(axis=-3, keepdims=True) == 1)
    if containment:
        result &= np.take(parents, PARENT_INDEX, axis=-3)
    return result


def composite_frame(rgb, children, parents, recipe):
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.shape != (*children.shape[-2:], 3):
        raise ValueError("Expected aligned uint8 RGB frame.")
    valid = validate_recipe(recipe, recipe["sequence"])
    effective = effective_masks(children, parents, valid["containment"])
    output, permitted = rgb.copy(), np.zeros(rgb.shape[:2], bool)
    for j, part in enumerate(valid["parts"]):
        if not part["enabled"] or part["strength"] == 0:
            continue
        permitted |= effective[j]
        source = rgb[effective[j]].astype(np.float64)
        output[effective[j]] = np.floor(source * (1-part["strength"]) +
                                       np.asarray(part["color"]) * part["strength"]).clip(0, 255).astype(np.uint8)
    return output, permitted


def _completed_report(run_root, stage):
    """Completion barrier checked before reading masks or rendering a scene."""
    path = run_root / stage / "results.json"
    if not path.is_file():
        raise RuntimeError(f"Need the completed {stage}/results.json before building a demo.")
    report = json.loads(path.read_text())
    from run_parent_experiment import configuration
    if report.get("frozen_config") != configuration():
        raise RuntimeError("Completed report does not match the frozen experiment source, model and policy.")
    seeds = renderer.TEST_SEEDS if stage == "test" else renderer.SMOKE_SEEDS
    expected = {f"{stage}_{condition}_{seed}" for condition, items in seeds.items() for seed in items}
    clips = report.get("clips", [])
    if report.get("stage") != stage or len(clips) != len(expected) or {c["scene"] for c in clips} != expected:
        raise RuntimeError("Report is not the complete fixed seed set.")
    if report["arms"]["raw_part"]["effective_edit"]["overall"]["scored_part_frames"] != len(expected) * 63 * 4:
        raise RuntimeError("Report denominator does not match all completed clips.")
    for name in expected:
        for relative in ("clip_manifest.json", "children/sam2_masks.npz", "parents/sam2_masks.npz"):
            if not (run_root / "clips" / name / relative).is_file():
                raise RuntimeError(f"Download the complete {stage} cache first; missing {name}/{relative}.")
    return report, path


def _short_summary(report):
    result = {"stage": report["stage"], "clips": len(report["clips"]), "passed": report["success"]["passed"]}
    for arm in ("raw_part", "parent_intersection"):
        values = report["arms"][arm]["effective_edit"]["overall"]
        result[arm] = {"wrong_car_pixels": values["wrong_car_pixels"],
                       "visible_mean_iou": values["mean_iou_given_visible"]["rate"],
                       "visible_recall": values["visible_micro_pixel_recall"]["rate"]}
    return result


def build(run_root, output, scene_name=TEST_SCENE, fps=12):
    run_root, output = Path(run_root), Path(output)
    if scene_name not in (TEST_SCENE, SMOKE_SCENE):
        raise ValueError("Only the preselected test_crossing_6200 or smoke_crossing_5200 scene is allowed.")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("FPS must be positive and finite.")
    stage = scene_name.split("_", 1)[0]
    report, report_path = _completed_report(run_root, stage)
    folder = run_root / "clips" / scene_name
    manifest_path = folder / "clip_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    record = next(c for c in report["clips"] if c["scene"] == scene_name)
    if record["clip_manifest_sha256"] != sha256(manifest_path):
        raise RuntimeError("Selected clip manifest differs from the completed report.")
    arrays, cache_paths = {}, {}
    for role, ids in (("children", [1, 2, 3, 4]), ("parents", [101, 102])):
        path = folder / role / "sam2_masks.npz"
        cache_paths[role] = path
        if manifest["cache_sha256"][role] != sha256(path):
            raise RuntimeError(f"Changed {role} prediction cache.")
        with np.load(path, allow_pickle=False) as data:
            if data["ids"].tolist() != ids or data["masks"].dtype != bool:
                raise ValueError(f"Unexpected {role} cache IDs or mask dtype.")
            arrays[role] = data["masks"]
    children, parents = arrays["children"], arrays["parents"]
    if children.shape != (64, 4, 384, 384) or parents.shape != (64, 2, 384, 384):
        raise ValueError("Unexpected native Day5 mask dimensions.")
    # This first scene render happens ONLY after the complete-result barrier.
    scene, owners = renderer.generate_scene(record["seed"], record["condition"], scene_name)
    if hashlib.sha256(scene.frames.tobytes()).hexdigest() != manifest["identity"]["rgb_sha256"]:
        raise RuntimeError("Regenerated scene RGB differs from the cached run.")
    initial_children = np.stack([scene.masks[0] == i for i in (1, 2, 3, 4)])
    initial_parents = np.stack([owners[0] == i for i in (1, 2)])
    for role, initial in (("children", initial_children), ("parents", initial_parents)):
        if hashlib.sha256(initial.tobytes()).hexdigest() != manifest["identity"][role+"0_sha256"]:
            raise RuntimeError(f"Changed first-frame {role} prompts.")
    # Never serialize future part/owner truth into the editor or registry.
    del owners
    encoded = {role: [[encode_rle(mask) for mask in frame] for frame in values]
               for role, values in arrays.items()}
    for role, values in arrays.items():
        for t, frame in enumerate(values):
            for j, mask in enumerate(frame):
                if not np.array_equal(decode_rle(encoded[role][t][j], (384, 384)), mask):
                    raise AssertionError("Mask RLE roundtrip changed a prediction.")
    with tempfile.TemporaryDirectory(prefix="hierarchy-demo-") as temporary:
        temp = Path(temporary)
        for t, frame in enumerate(scene.frames):
            Image.fromarray(frame).save(temp / f"{t:05d}.jpg", format="JPEG", quality=100, subsampling=0)
        video = temp / "source.mp4"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(fps),
                        "-start_number", "0", "-i", str(temp / "%05d.jpg"), "-frames:v", "64", "-c:v", "libx264",
                        "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(video)], check=True)
        video_bytes = video.read_bytes()
    provenance = {"clip_manifest_sha256": sha256(manifest_path), "report_sha256": sha256(report_path),
                  "prediction_archives": {role: {"path": str(path.relative_to(run_root)), "sha256": sha256(path)} for role, path in cache_paths.items()},
                  "rgb_sha256": manifest["identity"]["rgb_sha256"], "source_digest": manifest["identity"]["source_digest"],
                  "checkpoint_sha256": manifest["identity"]["checkpoint_sha256"], "model_revision": manifest["identity"]["model_revision"],
                  "builder_sha256": sha256(__file__), "embedded_video_sha256": hashlib.sha256(video_bytes).hexdigest()}
    registry = {"schema": "vjepa.scene-part-hierarchy", "version": 1, "sequence": scene_name,
                "width": 384, "height": 384, "frames": 64,
                "cars": [{"id": car, "parent_mask_id": pid, "parts": [p["id"] for p in PARTS if p["car"] == car]}
                         for car, pid in (("car_A", 101), ("car_B", 102))],
                "parts": copy.deepcopy(PARTS), "provenance": provenance,
                "annotations": {"frames": [0], "source": "Exact synthetic renderer labels at frame zero only",
                    "rle_encoding": "Base64 unsigned-LEB128 foreground gap,length pairs; row-major 384x384",
                    "children": [{"id": p["id"], "mask_id": p["mask_id"], "pixels": int(mask.sum()), "mask_rle": encode_rle(mask)} for p, mask in zip(PARTS, initial_children)],
                    "parents": [{"id": car, "mask_id": pid, "pixels": int(mask.sum()), "mask_rle": encode_rle(mask)} for car, pid, mask in zip(("car_A", "car_B"), (101, 102), initial_parents)]},
                "policy": {"raw_effective": "M_i minus union(all other RAW child masks), including disabled parts",
                           "contained_effective": "raw_effective_i intersect independently predicted parent_i",
                           "default_containment": False, "extra_parent_prompts": 2},
                "scope": "Synthetic two-dimensional scene; identities are assigned at frame0, not discovered. No faces or generation."}
    recipe = default_recipe(scene_name)
    audit = {"sequence": scene_name, "provenance": provenance, "frames_checked": 64, "rle_exact_roundtrips": 384,
             "unit": "RGB pixels before lossy encoding; includes annotation frame0", "modes": {}}
    baseline, gated = effective_masks(children, parents), effective_masks(children, parents, True)
    if np.any(gated & ~baseline):
        raise AssertionError("Containment released forbidden pixels.")
    overlap = children.sum(axis=1) > 1
    for mode, flag in (("raw", False), ("contained", True)):
        counts = {"changed_outside_permitted_pixels": 0, "changed_raw_overlap_pixels": 0,
                  "changed_outside_own_predicted_parent_pixels": 0 if flag else None,
                  "enabled_part_frames": 64 * 4}
        candidate = {**recipe, "containment": flag}
        for t, frame in enumerate(scene.frames):
            edited, allowed = composite_frame(frame, children[t], parents[t], candidate)
            changed = np.any(edited != frame, axis=-1)
            counts["changed_outside_permitted_pixels"] += int((changed & ~allowed).sum())
            counts["changed_raw_overlap_pixels"] += int((changed & overlap[t]).sum())
            if flag:
                for j in range(4):
                    counts["changed_outside_own_predicted_parent_pixels"] += int((changed & gated[t, j] & ~parents[t, PARENT_INDEX[j]]).sum())
        audit["modes"][mode] = counts
    audit["contained_is_subset_of_raw_for_every_part_frame"] = True
    payload = {"editorId": "hierarchy-"+scene_name+"-"+provenance["prediction_archives"]["children"]["sha256"][:10],
               "sequence": scene_name, "width": 384, "height": 384, "count": 64, "fps": fps,
               "parts": PARTS, "parentIndex": PARENT_INDEX, "children": encoded["children"], "parents": encoded["parents"],
               "defaultRecipe": recipe, "summary": _short_summary(report), "provenance": provenance}
    html = HTML.replace("__DATA__", json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c"))
    html = html.replace("__VIDEO__", base64.b64encode(video_bytes).decode("ascii"))
    html = re.sub(r'id="([^"]+)"', lambda m: f'id="{payload["editorId"]}"' if m[1] == "__ROOT__" else f'id="{payload["editorId"]}-{m[1]}" data-control="{m[1]}"', html)
    html = re.sub(r'for="([^"]+)"', lambda m: f'for="{payload["editorId"]}-{m[1]}"', html)
    html = html.replace("__SCOPE__", "#"+payload["editorId"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    for name, obj in (("scene_registry.json", registry), ("hierarchy_edit_invariants.json", audit), ("default_hierarchy.recipe.json", recipe)):
        (output.parent / name).write_text(json.dumps(obj, indent=2)+"\n")
    return {"output": str(output), "bytes": output.stat().st_size, "sequence": scene_name,
            "registry": str(output.parent / "scene_registry.json"), "invariants": str(output.parent / "hierarchy_edit_invariants.json"),
            "frames": 64, "rle_exact_roundtrips": 384, "default_containment": False}


HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Two cars, four persistent parts</title><style>
__SCOPE__{--bg:#111922;--card:#1b2835;--line:#35475a;--text:#edf5fa;--muted:#abc0d0;--accent:#8ee1c4;color-scheme:dark;max-width:1200px;margin:auto;padding:28px;background:var(--bg);color:var(--text);font:14px/1.5 system-ui,sans-serif;box-sizing:border-box}
__SCOPE__ *{box-sizing:border-box}__SCOPE__ h1{font-size:30px;line-height:1.15;letter-spacing:-.025em;margin:8px 0}__SCOPE__ p{margin:0;color:var(--muted)}__SCOPE__ .eyebrow{color:var(--accent);font-size:11px;font-weight:700;letter-spacing:.1em;text-transform:uppercase}
__SCOPE__ .views{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:22px}__SCOPE__ figure{margin:0;overflow:hidden;border:1px solid var(--line);border-radius:10px;background:#080f16}__SCOPE__ figcaption{padding:10px 13px;background:var(--card);display:flex;justify-content:space-between;font-size:12px}__SCOPE__ figcaption span{color:var(--muted)}__SCOPE__ video,__SCOPE__ canvas{display:block;width:100%;aspect-ratio:1;height:auto}
__SCOPE__ .panel{padding:17px;margin-top:16px;background:var(--card);border:1px solid var(--line);border-radius:10px}__SCOPE__ .controls{display:grid;grid-template-columns:1fr 1.4fr .7fr 1fr;gap:20px;align-items:start}__SCOPE__ label{display:block;color:var(--muted);font-size:12px;margin-bottom:6px}__SCOPE__ select,__SCOPE__ button,__SCOPE__ input[type=color]{font:inherit;background:#22374a;color:var(--text);border:1px solid #51677c;border-radius:6px;min-height:36px}__SCOPE__ select{width:100%;padding:6px 10px}__SCOPE__ button{padding:6px 12px;cursor:pointer}__SCOPE__ button:hover{background:#36516b}__SCOPE__ input[type=color]{width:100%;padding:3px}__SCOPE__ input[type=range]{width:100%;accent-color:var(--accent)}__SCOPE__ input[type=checkbox]{accent-color:var(--accent)}__SCOPE__ .check{display:flex;align-items:center;gap:8px;color:var(--text);margin:8px 0}__SCOPE__ .contain{border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:11px 0;margin:15px 0}__SCOPE__ .contain p{font-size:12px;max-width:850px}
__SCOPE__ button:focus-visible,__SCOPE__ select:focus-visible,__SCOPE__ input:focus-visible,__SCOPE__ textarea:focus-visible,__SCOPE__ summary:focus-visible{outline:2px solid var(--accent);outline-offset:3px}__SCOPE__ .transport{display:flex;gap:12px;align-items:center;margin-top:13px}__SCOPE__ .transport input{flex:1;min-width:50px}__SCOPE__ output{font-size:12px;white-space:nowrap}__SCOPE__ .row{display:flex;flex-wrap:wrap;gap:10px;align-items:center}__SCOPE__ .row span,__SCOPE__ .status{font-size:12px;color:var(--muted)}__SCOPE__ .status{margin-top:11px}__SCOPE__ .status b{color:var(--accent);font-weight:500}__SCOPE__ details{margin-top:12px}__SCOPE__ summary{cursor:pointer;color:var(--text);width:fit-content}__SCOPE__ textarea{display:block;width:100%;min-height:140px;resize:vertical;margin:7px 0 10px;padding:10px;background:#101d2b;color:var(--text);border:1px solid #51677c;border-radius:6px;font:12px/1.5 ui-monospace,monospace}__SCOPE__ .error{background:#543429;color:#ffe4d8;border-radius:7px;padding:10px;margin-top:12px}__SCOPE__ .error:empty{display:none}__SCOPE__ footer{margin-top:18px;color:var(--muted);font-size:12px;display:grid;grid-template-columns:1fr 1fr;gap:20px}__SCOPE__ footer p+p{margin-top:8px}__SCOPE__ footer b{color:var(--text)}__SCOPE__ .sr-only{position:absolute;width:1px;height:1px;clip:rect(0,0,0,0);overflow:hidden}
@media(max-width:760px){__SCOPE__{padding:18px 12px}__SCOPE__ .views{gap:8px}__SCOPE__ .controls{grid-template-columns:1fr 1fr;gap:13px}__SCOPE__ footer{grid-template-columns:1fr}__SCOPE__ h1{font-size:25px}__SCOPE__ .transport{gap:7px}__SCOPE__ .transport button{padding:5px 9px}}
</style></head><body><main id="__ROOT__"><header><div class="eyebrow">VJEPA research / Day 5 · synthetic scene</div><h1>Two cars. Four persistent part controls.</h1><p>Choose a car, then its door or window. Each part keeps its own color and strength across the clip.</p></header>
<section class="views" aria-label="Original and edited synthetic video"><figure><figcaption>Original <span id="sequence"></span></figcaption><video id="source" muted playsinline preload="auto" aria-label="Original synthetic cars" src="data:video/mp4;base64,__VIDEO__"></video></figure><figure><figcaption>Editable preview <span id="active">4 active edits</span></figcaption><canvas id="preview" width="384" height="384" aria-label="Four-part edited video"></canvas></figure></section>
<section class="panel" aria-label="Hierarchical part editing controls"><div class="controls"><div><label for="car">Car owner</label><select id="car"><option value="car_A">Car A</option><option value="car_B">Car B</option></select></div><div><label for="part">Part within this car</label><select id="part"></select><label class="check"><input id="enabled" type="checkbox" checked>Enable this part's edit</label></div><div><label for="color">Part color</label><input id="color" type="color" value="#2a89f0"></div><div><label for="strength">Color strength · <output id="strength-value">65%</output></label><input id="strength" type="range" min="0" max="100" step="1" value="65"></div></div>
<div class="contain"><label class="check"><input id="containment" type="checkbox">Contain edits inside tracked car</label><p>Off by default. Turning this on intersects each editable part with its separately tracked parent car. It uses <b>two extra whole-car prompts</b> at frame zero and an independent parent-tracking pass. It can remove useful part pixels and cannot restore a lost identity.</p></div>
<div class="row"><label class="check"><input id="tags" type="checkbox" checked>Show stable car.part tags</label><button id="reset" type="button">Reset all edits</button><button id="export" type="button">Export recipe</button><span id="recipe-status" aria-live="polite">Four settings remain attached to stable part IDs.</span></div>
<details><summary>Paste a recipe</summary><label for="recipe-text">Recipe JSON for this clip, including containment setting</label><textarea id="recipe-text" rows="6" spellcheck="false" placeholder="Paste an exported hierarchy recipe"></textarea><button id="apply" type="button">Apply recipe</button></details><div class="error" id="error" role="alert"></div>
<div class="transport"><button id="play" type="button">Play</button><button id="previous" type="button" aria-label="Previous frame">←</button><button id="next" type="button" aria-label="Next frame">→</button><label for="scrub" class="sr-only">Video frame</label><input id="scrub" type="range" min="0" max="63" step="1" value="0"><output id="clock">00 / 63</output><button id="save" type="button">Save frame</button></div><div class="status"><b id="selection"></b><div id="mask-status"></div></div></section>
<footer><div><p><b>Completed experiment, not a new model run.</b> <span id="results"></span></p><p>Scores cover every fixed clip in the completed stage, all four parts active and frame zero excluded. This single preselected crossing scene illustrates controls; it is not the entire benchmark.</p></div><div><p><b>Research boundary.</b> These are simple 2-D synthetic cars. Four child masks and two whole-car masks are initialized only at frame zero. Assigned tags are not automatic semantic discovery. No real-video, face-identity, 3-D or generative-editing claim is supported.</p><p>Every raw child overlap stays protected, even if a part is disabled or containment removes its mask. Containment can only remove editable pixels. Predicted masks can still identify the wrong car or anatomy. The offline preview blends colors; recipes save controls only.</p></div></footer>
</main><script>
'use strict';
(()=>{
const D=__DATA__;
const root=document.getElementById(D.editorId),$=id=>root.querySelector(`[data-control="${id}"]`);
const video=$('source'),canvas=$('preview'),ctx=canvas.getContext('2d',{willReadFrequently:true});
const clean=document.createElement('canvas');clean.width=D.width;clean.height=D.height;const cleanCtx=clean.getContext('2d');
canvas.width=D.width;canvas.height=D.height;
const SCHEMA='vjepa.hierarchical-part-edit-recipe';
function validateRecipe(recipe){
 const exact=(obj,keys)=>obj!==null&&typeof obj==='object'&&!Array.isArray(obj)&&Object.keys(obj).sort().join(',')===keys.slice().sort().join(',');
 if(!exact(recipe,['schema','version','sequence','containment','parts']))throw Error('Recipe needs exactly schema, version, sequence, containment and parts.');
 if(recipe.schema!==SCHEMA||recipe.version!==1)throw Error('Unsupported recipe schema/version.');
 if(recipe.sequence!==D.sequence)throw Error('Recipe belongs to a different video sequence.');
 if(typeof recipe.containment!=='boolean')throw Error('containment must be true or false.');
 if(!Array.isArray(recipe.parts)||recipe.parts.length!==D.parts.length)throw Error('Recipe must contain all four registered parts exactly once.');
 const known=new Set(D.parts.map(p=>p.id)),found=new Map();
 for(const p of recipe.parts){
  if(!exact(p,['id','enabled','color','strength']))throw Error('Unexpected or missing part controls.');
  if(typeof p.id!=='string'||!known.has(p.id)||found.has(p.id))throw Error('Unknown or duplicate stable part ID.');
  if(typeof p.enabled!=='boolean')throw Error('enabled must be true or false.');
  if(!Array.isArray(p.color)||p.color.length!==3||p.color.some(c=>!Number.isInteger(c)||c<0||c>255))throw Error('Color needs three integer RGB channels from 0 to 255.');
  if(typeof p.strength!=='number'||!Number.isFinite(p.strength)||p.strength<0||p.strength>1)throw Error('Strength must be a finite number from 0 to 1.');
  found.set(p.id,{id:p.id,enabled:p.enabled,color:p.color.slice(),strength:p.strength});
 }
 return {schema:SCHEMA,version:1,sequence:D.sequence,containment:recipe.containment,parts:D.parts.map(p=>found.get(p.id))};
}
function parseRecipe(text){if(new TextEncoder().encode(text).byteLength>65536)throw Error('Recipe exceeds the 64 KB controls-only limit.');let candidate;try{candidate=JSON.parse(text);}catch(error){throw Error('Recipe is not valid JSON. Paste the complete exported recipe.');}return validateRecipe(candidate);}
function effectiveMasks(children,parents,containment){
 const size=children[0].length,counts=new Uint8Array(size);for(const mask of children)for(let p=0;p<size;p++)counts[p]+=mask[p];
 return children.map((mask,j)=>{const out=new Uint8Array(size);for(let p=0;p<size;p++)out[p]=mask[p]&&counts[p]===1&&(!containment||parents[D.parentIndex[j]][p])?1:0;return out;});
}
function composite(pixels,children,parents,recipe){
 const masks=effectiveMasks(children,parents,recipe.containment);let permitted=0,active=0;
 for(let j=0;j<masks.length;j++){const part=recipe.parts[j];if(!part.enabled||part.strength===0)continue;active++;
  for(let p=0;p<masks[j].length;p++){if(!masks[j][p])continue;permitted++;const i=4*p;for(let c=0;c<3;c++)pixels[i+c]=Math.floor((1-part.strength)*pixels[i+c]+part.strength*part.color[c]);}}
 return {masks,permitted,active};
}
function decodeRLE(encoded){const bytes=atob(encoded),out=new Uint8Array(D.width*D.height);let cursor=0,previous=0;function read(){let n=0,shift=0;for(;;){if(cursor>=bytes.length)throw Error('Truncated mask');const b=bytes.charCodeAt(cursor++);n|=(b&127)<<shift;if(!(b&128))return n;shift+=7;if(shift>28)throw Error('Mask integer overflow');}}while(cursor<bytes.length){const start=previous+read(),length=read(),end=start+length;if(length<=0||end>out.length)throw Error('Invalid mask span');out.fill(1,start,end);previous=end;}return out;}
let recipe=validateRecipe(D.defaultRecipe),shownFrame=0,lastTime=0;
const frameIndex=t=>Math.max(0,Math.min(D.count-1,Math.floor(Math.max(0,t)*D.fps+1e-5)));
const selected=()=>recipe.parts.find(p=>p.id===$('part').value);
function updateLabels(){for(const option of $('part').options){const part=recipe.parts.find(p=>p.id===option.value),meta=D.parts.find(p=>p.id===option.value);option.textContent=`${meta.label} · ${part.enabled?'on':'off'}`;}}
function loadControls(){const part=selected();$('enabled').checked=part.enabled;$('color').value='#'+part.color.map(c=>c.toString(16).padStart(2,'0')).join('');$('strength').value=part.strength*100;$('strength-value').textContent=`${Number((100*part.strength).toFixed(2))}%`;$('containment').checked=recipe.containment;updateLabels();}
function chooseCar(){const kind=$('part').value.split('.').pop();$('part').replaceChildren();for(const part of D.parts.filter(p=>p.car===$('car').value)){const option=document.createElement('option');option.value=part.id;$('part').append(option);}const next=D.parts.find(p=>p.car===$('car').value&&p.id.endsWith('.'+kind));if(next)$('part').value=next.id;loadControls();}
$('car').value='car_A';chooseCar();$('sequence').textContent=D.sequence;
const cache=new Map();function masksAt(frame){if(!cache.has(frame)){cache.set(frame,{children:D.children[frame].map(decodeRLE),parents:D.parents[frame].map(decodeRLE)});if(cache.size>2)cache.delete(cache.keys().next().value);}return cache.get(frame);}
function drawTags(masks){if(!$('tags').checked)return;for(let j=0;j<masks.length;j++){let n=0,xsum=0,ymin=D.height;for(let p=0;p<masks[j].length;p++)if(masks[j][p]){n++;xsum+=p%D.width;ymin=Math.min(ymin,Math.floor(p/D.width));}if(!n)continue;const text=D.parts[j].id;ctx.font='10px system-ui';const width=ctx.measureText(text).width+10,x=Math.max(3,Math.min(D.width-width-3,xsum/n-width/2)),y=Math.max(3,ymin-18);ctx.fillStyle='rgba(8,17,24,.87)';ctx.fillRect(x,y,width,17);ctx.fillStyle='#'+recipe.parts[j].color.map(c=>c.toString(16).padStart(2,'0')).join('');ctx.fillText(text,x+5,y+12);}}
function showError(error){$('error').textContent=String(error.message||error);}function clearError(){$('error').textContent='';}
function render(time=lastTime){if(video.readyState<2||video.seeking)return;lastTime=time;shownFrame=frameIndex(time);const masks=masksAt(shownFrame);ctx.drawImage(video,0,0,D.width,D.height);const frame=ctx.getImageData(0,0,D.width,D.height),stats=composite(frame.data,masks.children,masks.parents,recipe);ctx.putImageData(frame,0,0);cleanCtx.putImageData(frame,0,0);drawTags(stats.masks);$('scrub').value=shownFrame;$('clock').textContent=`${String(shownFrame).padStart(2,'0')} / ${D.count-1}`;$('active').textContent=`${stats.active} active ${stats.active===1?'edit':'edits'}`;$('selection').textContent=`Adjusting ${$('part').value} · containment ${recipe.containment?'on':'off'}`;$('mask-status').textContent=`${stats.permitted.toLocaleString()} permitted edit pixels · frame ${shownFrame} · all raw child overlaps protected`;}
function safeRender(time=lastTime){try{render(time);}catch(error){video.pause();showError(error);}}
function seek(frame){video.pause();video.currentTime=(Math.max(0,Math.min(D.count-1,frame))+.15)/D.fps;}
$('car').addEventListener('change',()=>{chooseCar();safeRender();});$('part').addEventListener('change',()=>{loadControls();safeRender();});
for(const id of ['enabled','color','strength'])$(id).addEventListener('input',()=>{const p=selected();if(id==='enabled')p.enabled=$('enabled').checked;if(id==='color')p.color=[1,3,5].map(i=>parseInt($('color').value.slice(i,i+2),16));if(id==='strength')p.strength=Number($('strength').value)/100;loadControls();clearError();$('recipe-status').textContent='Four settings remain attached to stable part IDs.';safeRender();});
$('containment').addEventListener('input',()=>{recipe.containment=$('containment').checked;clearError();safeRender();});$('tags').addEventListener('input',()=>safeRender());
$('reset').addEventListener('click',()=>{recipe=validateRecipe(D.defaultRecipe);recipe.parts.forEach(p=>p.enabled=false);recipe.containment=false;loadControls();clearError();$('recipe-status').textContent='All four edits disabled; containment off.';safeRender();});
function download(blob,name){const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;root.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('export').addEventListener('click',()=>{const text=JSON.stringify(validateRecipe(recipe),null,2)+'\n';$('recipe-text').value=text;download(new Blob([text],{type:'application/json'}),`${D.sequence}_hierarchy.recipe.json`);$('recipe-status').textContent='Recipe exported and copied into the paste field.';});
$('apply').addEventListener('click',()=>{try{const candidate=parseRecipe($('recipe-text').value);recipe=candidate;loadControls();clearError();$('recipe-status').textContent='Recipe restored by stable car.part IDs.';safeRender();}catch(error){showError(error);$('recipe-status').textContent='Import rejected; previous settings retained.';}});
$('scrub').max=D.count-1;$('scrub').addEventListener('input',()=>seek(Number($('scrub').value)));$('previous').addEventListener('click',()=>seek(shownFrame-1));$('next').addEventListener('click',()=>seek(shownFrame+1));
$('play').addEventListener('click',async()=>{try{if(video.paused){if(video.ended||shownFrame===D.count-1)video.currentTime=0;await video.play();}else video.pause();}catch(error){showError(error);}});
video.addEventListener('play',()=>{$('play').textContent='Pause';});video.addEventListener('pause',()=>{$('play').textContent=video.ended?'Replay':'Play';});video.addEventListener('ended',()=>{$('play').textContent='Replay';safeRender(video.currentTime);});video.addEventListener('loadeddata',()=>safeRender(video.currentTime));video.addEventListener('seeked',()=>safeRender(video.currentTime));video.addEventListener('error',()=>showError('Cannot decode the embedded MP4. Use a browser with H.264 support.'));
if(video.readyState>=2)safeRender(video.currentTime);
if('requestVideoFrameCallback' in video){const callback=(_,meta)=>{safeRender(meta.mediaTime);video.requestVideoFrameCallback(callback);};video.requestVideoFrameCallback(callback);}else{const tick=()=>{if(!video.paused&&!video.seeking)safeRender(video.currentTime);requestAnimationFrame(tick);};requestAnimationFrame(tick);}
$('save').addEventListener('click',()=>{if(video.readyState<2)return;clean.toBlob(blob=>{if(blob)download(blob,`${D.sequence}_frame_${String(shownFrame).padStart(3,'0')}.png`);},'image/png');});
const summary=D.summary,pct=x=>(100*x).toFixed(2)+'%',raw=summary.raw_part,gated=summary.parent_intersection;
$('results').textContent=`${summary.clips} ${summary.stage} clips: wrong-car effective pixels ${raw.wrong_car_pixels.toLocaleString()} → ${gated.wrong_car_pixels.toLocaleString()} with containment. Visible mean IoU ${pct(raw.visible_mean_iou)} → ${pct(gated.visible_mean_iou)}; recall ${pct(raw.visible_recall)} → ${pct(gated.visible_recall)}. Fixed overall criterion ${summary.passed?'passed':'did not pass'}.`;
window.hierarchyEditorTests=window.hierarchyEditorTests||{};window.hierarchyEditorTests[D.editorId]={validateRecipe,parseRecipe,effectiveMasks,composite,decodeRLE,frameIndex};
})();
</script></body></html>'''


def self_test():
    """Fabricated arrays only. Never render smoke or held-out scene seeds."""
    children, parents = np.zeros((4, 8, 8), bool), np.zeros((2, 8, 8), bool)
    children[0, 1:5, 1:5] = True
    children[1, 4:7, 1:4] = True
    children[2, 2:6, 4:7] = True
    children[3, 6:8, 5:8] = True
    parents[0, :, :5] = True
    parents[1, :, 5:] = True
    raw, gated = effective_masks(children, parents), effective_masks(children, parents, True)
    overlap = children.sum(axis=0) > 1
    assert not raw[:, overlap].any() and not gated[:, overlap].any()
    assert not (gated & ~raw).any()
    assert raw.sum() > gated.sum() > 0
    # A tempting but incorrect implementation would recompute overlap after
    # parent clipping, freeing an overlapping pixel that the protocol protects.
    clipped = children & parents[PARENT_INDEX]
    incorrect = clipped & (clipped.sum(axis=0, keepdims=True) == 1)
    assert (incorrect & overlap).any() and not (gated & overlap).any()
    rgb = np.arange(8*8*3, dtype=np.uint8).reshape(8, 8, 3)
    recipe = default_recipe("fabricated_check_only")
    recipe["parts"][2]["enabled"] = False
    for flag in (False, True):
        recipe["containment"] = flag
        output, permitted = composite_frame(rgb, children, parents, recipe)
        assert np.array_equal(output[~permitted], rgb[~permitted])
        assert np.array_equal(output[overlap], rgb[overlap])
    assert validate_recipe({**recipe, "parts": recipe["parts"][::-1]}, recipe["sequence"]) == recipe
    bad = []
    for field, value in (("sequence", "wrong"), ("containment", 1), ("schema", "wrong"), ("version", True)):
        bad.append({**copy.deepcopy(recipe), field: value})
    for field, value in (("id", "unknown.door"), ("enabled", 1), ("color", [True, 2, 3]), ("strength", float("nan")), ("strength", True)):
        item = copy.deepcopy(recipe);item["parts"][-1][field] = value;bad.append(item)
    duplicate = copy.deepcopy(recipe);duplicate["parts"][-1] = copy.deepcopy(duplicate["parts"][0]);bad.append(duplicate)
    for candidate in bad:
        try:
            validate_recipe(candidate, recipe["sequence"])
        except ValueError:
            pass
        else:
            raise AssertionError("Accepted malformed recipe")
    numeric_recipe = copy.deepcopy(recipe)
    numeric_recipe["version"] = 1.0
    for part in numeric_recipe["parts"]:
        part["color"] = [float(c) for c in part["color"]]
    numeric_json = json.dumps(numeric_recipe)
    canonical = validate_recipe(json.loads(numeric_json), recipe["sequence"])
    assert canonical == recipe and type(canonical["version"]) is int
    assert all(type(channel) is int for part in canonical["parts"] for channel in part["color"])
    invalid_numeric_json = []
    for target, value in (("version", 1.5), ("version", True), ("color", 42.5), ("color", True)):
        candidate = copy.deepcopy(numeric_recipe)
        if target == "version":
            candidate["version"] = value
        else:
            candidate["parts"][-1]["color"][0] = value
        text = json.dumps(candidate)
        invalid_numeric_json.append(text)
        try:
            validate_recipe(json.loads(text), recipe["sequence"])
        except ValueError:
            pass
        else:
            raise AssertionError("Accepted a noninteger or boolean numeric literal")
    # Javascript helper checks execute the actual embedded pure functions.
    script = HTML.split("<script>", 1)[1].split("</script>", 1)[0]
    start, end = script.index("const SCHEMA="), script.index("function decodeRLE")
    pure = script[start:end]
    data = {"sequence": recipe["sequence"], "parts": PARTS, "parentIndex": PARENT_INDEX}
    fixture = {"children": children.reshape(4, -1).astype(int).tolist(), "parents": parents.reshape(2, -1).astype(int).tolist(),
               "raw": raw.reshape(4, -1).astype(int).tolist(), "gated": gated.reshape(4, -1).astype(int).tolist(), "recipe": recipe,
               "pixels": np.concatenate([rgb, np.full((8, 8, 1), 255, np.uint8)], axis=-1).ravel().tolist(),
               "integer_numeric_json": numeric_json, "invalid_numeric_json": invalid_numeric_json}
    expected_rgb, _ = composite_frame(rgb, children, parents, recipe)
    fixture["expected_pixels"] = np.concatenate([expected_rgb, np.full((8, 8, 1), 255, np.uint8)], axis=-1).ravel().tolist()
    javascript = "const assert=require('assert/strict');const D="+json.dumps(data)+";\n"+pure+"\nconst F="+json.dumps(fixture)+";\n"+r'''
const child=F.children.map(x=>Uint8Array.from(x)),parent=F.parents.map(x=>Uint8Array.from(x));
assert.deepEqual(effectiveMasks(child,parent,false).map(x=>Array.from(x)),F.raw);
assert.deepEqual(effectiveMasks(child,parent,true).map(x=>Array.from(x)),F.gated);
const original=Uint8ClampedArray.from(F.pixels),edited=original.slice(),recipe=validateRecipe(F.recipe);
const result=composite(edited,child,parent,recipe);
assert.deepEqual(Array.from(edited),F.expected_pixels);
for(let p=0;p<child[0].length;p++){const allowed=result.masks.some((m,j)=>m[p]&&recipe.parts[j].enabled&&recipe.parts[j].strength>0);if(!allowed)assert.deepEqual(edited.slice(4*p,4*p+4),original.slice(4*p,4*p+4));}
assert.deepEqual(validateRecipe({...recipe,parts:recipe.parts.slice().reverse()}),recipe);
assert.deepEqual(parseRecipe(F.integer_numeric_json),recipe);
for(const text of F.invalid_numeric_json)assert.throws(()=>parseRecipe(text));
for(const text of ['{',JSON.stringify({...recipe,containment:1}),JSON.stringify({...recipe,sequence:'wrong'}),'é'.repeat(32769)])assert.throws(()=>parseRecipe(text));
console.log('JS: raw-overlap veto, exact Python mask/pixel parity, subset, disabled protection, recipe ordering/rejections passed');
'''
    with tempfile.TemporaryDirectory(prefix="hierarchy-cpu-check-") as directory:
        path = Path(directory) / "check.js";path.write_text(javascript)
        subprocess.run(["node", str(path)], check=True)
        # Complete UI source syntax, replacing only its data placeholder.
        path.write_text(script.replace("__DATA__", json.dumps(data)))
        subprocess.run(["node", "--check", str(path)], check=True)
        stage_dir = Path(directory) / "test"
        stage_dir.mkdir()
        (stage_dir / "results.json").write_text(json.dumps({"frozen_config": {"source_digest": "fabricated_mismatch"}}))
        try:
            _completed_report(Path(directory), "test")
        except RuntimeError as error:
            assert "frozen experiment source" in str(error)
        else:
            raise AssertionError("Accepted a report from a different frozen experiment")
    return {"fixture": "fabricated 8x8 arrays; no scene seed accessed", "raw_overlap_veto": "passed",
            "containment_subset": "passed", "outside_mask_preservation": "passed",
            "disabled_other_part_still_protects_overlap": "passed", "rejected_python_recipes": len(bad),
            "javascript_python_mask_parity": "passed", "javascript_python_pixel_parity": "passed",
            "same_json_numeric_literal_validation": "passed; integral floats canonicalized, 4 fractional/boolean cases rejected",
            "frozen_report_source_barrier": "passed", "javascript_source_syntax": "passed"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root")
    parser.add_argument("--scene", default=TEST_SCENE, choices=[TEST_SCENE, SMOKE_SCENE])
    parser.add_argument("--out")
    parser.add_argument("--fps", type=float, default=12)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
    else:
        if not args.run_root or not args.out:
            parser.error("--run-root and --out are required unless using --self-test")
        result = build(args.run_root, args.out, args.scene, args.fps)
    print(json.dumps(result, indent=2))
