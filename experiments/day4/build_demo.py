"""Build an offline interactive named-part recoloring demo from predicted masks.

Requires numpy, OpenCV and ffmpeg at build time; the resulting HTML needs only a
modern browser. No server, model, external assets or network calls are used by
the demo. Mask RLE round trips are checked for every part in every frame.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


def encode_rle(mask: np.ndarray) -> str:
    """Base64 unsigned-LEB128 foreground (gap, length) pairs in row-major order."""
    flat = np.asarray(mask, dtype=np.uint8).ravel()
    changes = np.diff(np.r_[np.uint8(0), flat, np.uint8(0)].astype(np.int16))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    result = bytearray()
    previous_end = 0
    for start, end in zip(starts, ends):
        for number in (int(start) - previous_end, int(end - start)):
            while number >= 128:
                result.append((number & 127) | 128)
                number >>= 7
            result.append(number)
        previous_end = int(end)
    return base64.b64encode(result).decode("ascii")


def decode_rle(encoded: str, shape: tuple[int, int]) -> np.ndarray:
    data = base64.b64decode(encoded, validate=True)
    values, number, shift = [], 0, 0
    for byte in data:
        number |= (byte & 127) << shift
        if byte & 128:
            shift += 7
            if shift > 28:
                raise ValueError("RLE integer overflow")
        else:
            values.append(number)
            number, shift = 0, 0
    if shift or len(values) % 2:
        raise ValueError("Truncated RLE")
    flat = np.zeros(int(np.prod(shape)), dtype=bool)
    end = 0
    for gap, length in zip(values[::2], values[1::2]):
        start = end + gap
        end = start + length
        if not length or end > len(flat):
            raise ValueError("Invalid RLE span")
        flat[start:end] = True
    return flat.reshape(shape)


def frame_index(seconds: float, fps: float, count: int) -> int:
    return max(0, min(count - 1, int(np.floor(max(0.0, seconds) * fps))))


def build_demo(frames_dir, masks_file, registry_file, output, fps=12, results_file=None):
    frames_dir, masks_file = Path(frames_dir), Path(masks_file)
    registry_file, output = Path(registry_file), Path(output)
    frames = sorted(frames_dir.glob("*.jpg"))
    with np.load(masks_file, allow_pickle=False) as saved:
        masks = np.asarray(saved["masks"], dtype=bool)
        ids = list(map(int, saved["ids"]))
    registry = json.loads(registry_file.read_text())
    parts_by_id = {int(p["id"]): p for p in registry["parts"]}
    if masks.ndim != 4 or len(frames) != len(masks) or masks.shape[1] != len(ids):
        raise ValueError("Expected one predicted mask set per frame")
    if not frames or fps <= 0 or len(set(ids)) != len(ids):
        raise ValueError("Invalid frames, frame rate or duplicate IDs")
    if set(ids) != set(parts_by_id):
        raise ValueError("Predicted mask IDs and registry IDs differ")
    if [p.name for p in frames] != [f"{i:05d}.jpg" for i in range(len(frames))]:
        raise ValueError("Frame names must be contiguous 00000.jpg through final frame")
    count, _, height, width = masks.shape
    for path in frames:
        rgb = cv2.imread(str(path))
        if rgb is None or rgb.shape[:2] != (height, width):
            raise ValueError(f"Frame/mask size mismatch: {path.name}")
    rle = [[encode_rle(mask) for mask in frame] for frame in masks]
    for t in range(count):
        for j in range(len(ids)):
            if not np.array_equal(decode_rle(rle[t][j], (height, width)), masks[t, j]):
                raise AssertionError(f"Lossy RLE at frame {t}, part {ids[j]}")
    for t in range(count):
        assert frame_index((t + .15) / fps, fps, count) == t
    assert frame_index(-1, fps, count) == 0
    assert frame_index(count / fps, fps, count) == count - 1
    assert frame_index(1e9, fps, count) == count - 1
    metrics = {}
    if results_file is not None:
        results = json.loads(Path(results_file).read_text())
        metrics = {name: {k: v for k, v in result.items() if k != "rows"}
                   for name, result in results.items()
                   if isinstance(result, dict) and "mean_iou" in result}
    with tempfile.TemporaryDirectory(prefix="part-editor-build-") as temporary:
        video = Path(temporary) / "source.mp4"
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(fps),
                   "-start_number", "0", "-i", str(frames_dir / "%05d.jpg"), "-frames:v", str(count),
                   "-c:v", "libx264", "-preset", "medium", "-crf", "22", "-pix_fmt", "yuv420p",
                   "-movflags", "+faststart", "-an", str(video)]
        subprocess.run(command, check=True)
        video_bytes = video.read_bytes()
    payload = {
        "width": width, "height": height, "fps": fps, "count": count, "ids": ids,
        "parts": [parts_by_id[i] for i in ids], "rle": rle, "metrics": metrics,
        "registry": registry,
        "provenance": {"masks_sha256": hashlib.sha256(masks_file.read_bytes()).hexdigest(),
                       "registry_sha256": hashlib.sha256(registry_file.read_bytes()).hexdigest(),
                       "embedded_video_sha256": hashlib.sha256(video_bytes).hexdigest()},
        "validation": {"rle_exact_roundtrips": count * len(ids), "frame_index_bounds": "passed"},
    }
    # '<' is escaped so a user-supplied registry name cannot close the JSON script.
    html = HTML.replace("__DATA__", json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c"))
    html = html.replace("__VIDEO__", base64.b64encode(video_bytes).decode("ascii"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    return {"output": str(output), "bytes": output.stat().st_size,
            "embedded_video_bytes": len(video_bytes), "width": width, "height": height,
            "frames": count, "fps": fps, **payload["validation"]}


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Persistent part editor · VJEPA research</title>
<style>
:root{color-scheme:dark;--bg:#111720;--card:#192330;--line:#2b394b;--text:#edf3fa;--muted:#a6b7cc;--accent:#8ccebc}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,-apple-system,sans-serif}
main{max-width:1420px;margin:0 auto;padding:28px 30px 34px}header{display:flex;justify-content:space-between;gap:20px;align-items:start;margin-bottom:24px}
.eyebrow{color:var(--accent);font-size:12px;font-weight:700;letter-spacing:.11em;text-transform:uppercase}h1{font-size:29px;line-height:1.2;margin:7px 0 8px;letter-spacing:-.025em}
p{margin:0;color:var(--muted)}.badge{padding:6px 11px;border:1px solid var(--line);border-radius:30px;font-size:12px;white-space:nowrap;color:var(--muted)}
.views{display:grid;grid-template-columns:1fr 1fr;gap:18px}.view{margin:0;border:1px solid var(--line);border-radius:12px;overflow:hidden;background:#080d13}
figcaption{padding:10px 14px;font-size:13px;display:flex;justify-content:space-between;background:var(--card);border-bottom:1px solid var(--line)}figcaption span{color:var(--muted)}
video,canvas{width:100%;height:auto;display:block;aspect-ratio:854/480}.panel{background:var(--card);border:1px solid var(--line);border-radius:12px;margin-top:18px;padding:18px}
.controls{display:grid;grid-template-columns:1.15fr .65fr 1fr 1fr;gap:24px;align-items:center}label,.control-title{display:block;font-size:12px;color:var(--muted);margin-bottom:7px}
select,button,input[type=color]{font:inherit;color:var(--text);background:#222f3f;border:1px solid #40516a;border-radius:7px;min-height:38px}select{padding:7px 12px;width:100%}button{padding:7px 14px;cursor:pointer}button:hover{background:#304258}button:focus-visible,input:focus-visible,select:focus-visible{outline:2px solid var(--accent);outline-offset:3px}
input[type=color]{padding:3px;width:100%;max-width:116px;cursor:pointer}input[type=range]{accent-color:var(--accent);width:100%;cursor:pointer}.strength-label{display:flex;justify-content:space-between}.toggles label{display:flex;gap:9px;align-items:center;margin:3px 0;font-size:13px;color:var(--text)}.toggles input{accent-color:var(--accent)}
.transport{display:flex;gap:14px;align-items:center;border-top:1px solid var(--line);margin-top:17px;padding-top:17px}.transport>input{flex:1;min-width:80px}.transport output{font:12px ui-monospace,monospace;min-width:103px;color:var(--muted)}#play{background:#345e55;border-color:#54887b;min-width:83px}.step{padding:6px 11px}.status{display:flex;justify-content:space-between;gap:16px;margin-top:13px;font-size:12px;color:var(--muted)}#selection{color:var(--accent)}
footer{margin-top:22px;font-size:12px;color:var(--muted);display:grid;grid-template-columns:1fr 1fr;gap:28px}footer b{font-weight:650;color:#d7e3f1}footer p+p{margin-top:7px}a{color:#b2d7ee}#error{display:none;background:#56362b;color:#ffe7df;padding:12px;margin-top:15px;border-radius:8px}.metrics{display:flex;gap:18px;flex-wrap:wrap;margin-bottom:8px}.metric strong{color:var(--text);font-size:20px;display:block;font-variant-numeric:tabular-nums}.metric span{font-size:11px}.sr-only{position:absolute;clip:rect(0,0,0,0);width:1px;height:1px;overflow:hidden}
@media(max-width:850px){main{padding:20px 15px}.views{grid-template-columns:1fr}.controls{grid-template-columns:1fr 1fr;gap:16px}footer{grid-template-columns:1fr;gap:16px}header{display:block}.badge{display:inline-block;margin-top:12px}.transport{gap:8px}.transport output{min-width:78px;font-size:11px}.step{display:none}.status{display:block}.status span{display:block}} 
</style></head><body><main>
<header><div><div class="eyebrow">VJEPA research / Day 4</div><h1>Choose a part. Keep the edit.</h1><p>Change the door or window once, then follow the same tagged part through the clip.</p></div><span class="badge">Offline research prototype · SAM 2.1 masks</span></header>
<section class="views" aria-label="Original and edited video comparison">
<figure class="view"><figcaption>Original <span id="sequence-label">DAVIS</span></figcaption><video id="source" muted playsinline preload="auto" aria-label="Original car video" src="data:video/mp4;base64,__VIDEO__"></video></figure>
<figure class="view"><figcaption>Editable preview <span id="preview-label">car_1 / front_door_panel</span></figcaption><canvas id="preview" width="854" height="480" aria-label="Recolored video preview"></canvas></figure>
</section>
<section class="panel" aria-label="Part editing controls">
<div class="controls"><div><label for="part">Tagged part · car_1</label><select id="part"></select></div><div><label for="color">Edit color</label><input id="color" type="color" value="#238cf0"></div><div><label class="strength-label" for="strength"><span>Strength</span><output id="strength-value">75%</output></label><input id="strength" type="range" min="0" max="100" value="75"></div><div class="toggles"><label><input id="outlines" type="checkbox">Show predicted outlines</label><label><input id="tags" type="checkbox">Show stable part tags</label></div></div>
<div class="transport"><button id="play" type="button">Play</button><button id="previous" class="step" type="button" aria-label="Previous frame">←</button><button id="next" class="step" type="button" aria-label="Next frame">→</button><label for="scrub" class="sr-only">Video frame</label><input id="scrub" type="range" min="0" max="63" value="0" step="1"><output id="clock">00 / 63</output><button id="save" type="button">Save frame</button></div>
<div class="status"><span id="selection" aria-live="polite">Preparing predicted masks…</span><span id="frame-status">64 frames · 12 fps · native 854 × 480</span></div>
</section>
<div id="error" role="alert"></div>
<footer><div><div class="metrics" id="metrics"></div><p><b>Scope of this result.</b> <span id="score-scope">Sparse scores compare predicted parts with approximate assistant-authored polygons.</span> The initial frame alone prompts tracking; there are no later corrections. Masks can include nearby pixels. SAM 2 may have seen DAVIS in training.</p><p>The selected predicted mask is edited only after subtracting every unselected predicted mask. This protects predicted regions, not anatomical truth. This is deterministic recoloring, not generative video editing or a face-identity result.</p></div><div><p><b>Provenance.</b> Video: DAVIS authors, <a id="sequence-credit" href="https://davischallenge.org/" target="_blank" rel="noopener">DAVIS car sequence</a>, using <a href="https://creativecommons.org/licenses/by-nc/4.0/" target="_blank" rel="noopener">CC BY-NC 4.0</a> terms. Preview modifications: MP4 compression, new part tags and optional recoloring. No endorsement is implied. Perazzi et al., CVPR 2016.</p><p>Predictions: <a href="https://github.com/facebookresearch/sam2" target="_blank" rel="noopener">SAM 2.1 Hiera Tiny</a> (Apache 2.0; Ravi et al., 2024). All frames and masks are embedded. Controls require no model rerun or internet. The PNG saves the edited frame without diagnostic overlays.</p></div></footer>
</main><script>
'use strict';
// Colab may remove inert application/json script elements from displayed HTML.
// Inline, HTML-escaped JSON keeps the same payload available offline and there.
const D=__DATA__, $=id=>document.getElementById(id);
const video=$('source'), canvas=$('preview'), ctx=canvas.getContext('2d',{willReadFrequently:true});
canvas.width=D.width;canvas.height=D.height;video.width=D.width;video.height=D.height;
const clean=document.createElement('canvas');clean.width=D.width;clean.height=D.height;
const cleanCtx=clean.getContext('2d');let shownFrame=0,lastMediaTime=0;
// Browser media timestamps may be rounded to microseconds. The tiny tolerance
// keeps a frame timestamp such as 0.083333 mapped to frame 1 at 12 fps.
const clampFrame=seconds=>Math.max(0,Math.min(D.count-1,Math.floor(Math.max(0,seconds)*D.fps+1e-5)));
const niceName=p=>p.name.replaceAll('_',' ');
for(const p of D.parts){const option=document.createElement('option');option.value=p.id;option.textContent=`${niceName(p)} · ID ${p.id}`;$('part').append(option);}
$('scrub').max=D.count-1;
$('clock').textContent=`00 / ${D.count-1}`;
$('frame-status').textContent=`${D.count} frames · ${D.fps} fps · native ${D.width} × ${D.height}`;
$('sequence-label').textContent=`DAVIS · ${D.registry.sequence}`;
$('sequence-credit').textContent=D.registry.sequence;
const scored=D.metrics['SAM2.1 tiny']?.scored_part_frames;
if(Number.isInteger(scored))$('score-scope').textContent=`Sparse scores use ${scored} part–frame pairs on this clip, against approximate assistant-authored polygons.`;
for(const [name,score] of Object.entries(D.metrics)){
 const box=document.createElement('div');box.className='metric';const value=document.createElement('strong'),label=document.createElement('span');
 value.textContent=`${(100*score.mean_iou).toFixed(1)}%`;label.textContent=`${name} · mean IoU`;box.append(value,label);$('metrics').append(box);
}
function decodeRLE(encoded){
 const bytes=atob(encoded),out=new Uint8Array(D.width*D.height);let cursor=0,previousEnd=0;
 function read(){let result=0,shift=0;for(;;){if(cursor>=bytes.length)throw Error('Truncated mask');const n=bytes.charCodeAt(cursor++);result|=(n&127)<<shift;if(!(n&128))return result;shift+=7;if(shift>28)throw Error('Mask integer overflow');}}
 while(cursor<bytes.length){const start=previousEnd+read(),length=read(),end=start+length;if(length<=0||end>out.length)throw Error('Invalid mask span');out.fill(1,start,end);previousEnd=end;}
 return out;
}
// Cache only two frame sets: enough for scrubbing without keeping 52 MB decoded.
const maskCache=new Map();
function masksAt(frame){if(!maskCache.has(frame)){maskCache.set(frame,D.rle[frame].map(decodeRLE));if(maskCache.size>2)maskCache.delete(maskCache.keys().next().value);}return maskCache.get(frame);}
function compositePixels(pixels,masks,selected,color,strength){
 const target=masks[selected];let changed=0,protectedCount=0;
 for(let p=0;p<target.length;p++){
   if(!target[p])continue;let protectedPixel=false;
   for(let j=0;j<masks.length;j++){if(j!==selected&&masks[j][p]){protectedPixel=true;break;}}
   if(protectedPixel){protectedCount++;continue;}
   const i=p*4,gray=Math.round(.299*pixels[i]+.587*pixels[i+1]+.114*pixels[i+2])/255,shade=.35+.65*gray;
   for(let c=0;c<3;c++)pixels[i+c]=Math.floor(Math.max(0,Math.min(255,(1-strength)*pixels[i+c]+strength*color[c]*shade)));
   changed++;
 }
 return {permitted:changed,protectedOverlap:protectedCount};
}
function drawOverlays(masks,selected){
 if(!$('outlines').checked&&!$('tags').checked)return;
 const colors=['#83dbba','#ffc46c'];
 masks.forEach((mask,j)=>{
  ctx.fillStyle=colors[j%colors.length];let count=0,sumX=0,minY=D.height;
  for(let p=0;p<mask.length;p++){if(!mask[p])continue;const x=p%D.width,y=Math.floor(p/D.width);count++;sumX+=x;minY=Math.min(minY,y);
   if($('outlines').checked&&(x===0||x===D.width-1||y===0||y===D.height-1||!mask[p-1]||!mask[p+1]||!mask[p-D.width]||!mask[p+D.width]))ctx.fillRect(x,y,1,1);
  }
  if($('tags').checked&&count){const label=`${D.parts[j].parent} / ${D.parts[j].name} #${D.ids[j]}`;ctx.font='12px system-ui';const w=ctx.measureText(label).width+14,x=Math.max(4,Math.min(D.width-w-4,sumX/count-w/2)),y=Math.max(4,minY-25);ctx.fillStyle='rgba(10,18,27,.86)';ctx.fillRect(x,y,w,21);ctx.fillStyle=colors[j%colors.length];ctx.fillText(label,x+7,y+15);}
 });
}
function render(mediaTime=lastMediaTime){
 if(video.readyState<2||video.seeking)return;
 lastMediaTime=mediaTime;shownFrame=clampFrame(mediaTime);
 const selected=D.ids.indexOf(Number($('part').value)),masks=masksAt(shownFrame),hex=$('color').value;
 const color=[1,3,5].map(i=>parseInt(hex.slice(i,i+2),16)),strength=Number($('strength').value)/100;
 ctx.drawImage(video,0,0,D.width,D.height);const frame=ctx.getImageData(0,0,D.width,D.height);
 const stats=compositePixels(frame.data,masks,selected,color,strength);ctx.putImageData(frame,0,0);cleanCtx.putImageData(frame,0,0);
 drawOverlays(masks,selected);$('scrub').value=shownFrame;$('clock').textContent=`${String(shownFrame).padStart(2,'0')} / ${D.count-1}`;
 const p=D.parts[selected];$('preview-label').textContent=`${p.parent} / ${p.name}`;$('strength-value').textContent=`${Math.round(strength*100)}%`;
 const other=D.parts.filter((_,j)=>j!==selected).map(niceName).join(', ');
 $('selection').textContent=`Editing ${niceName(p)} · protecting ${other}`;
 $('frame-status').textContent=`${stats.permitted.toLocaleString()} permitted pixels · frame ${shownFrame} · ${D.fps} fps`;
}
function showError(error){$('error').textContent=String(error.message||error);$('error').style.display='block';}
function safeRender(time){try{render(time);}catch(error){video.pause();showError(error);}}
function seek(frame){video.pause();video.currentTime=(Math.max(0,Math.min(D.count-1,frame))+.15)/D.fps;}
$('part').addEventListener('change',()=>safeRender(lastMediaTime));
for(const id of ['color','strength','outlines','tags'])$(id).addEventListener('input',()=>safeRender(lastMediaTime));
$('scrub').addEventListener('input',()=>seek(Number($('scrub').value)));
$('previous').addEventListener('click',()=>seek(shownFrame-1));$('next').addEventListener('click',()=>seek(shownFrame+1));
$('play').addEventListener('click',async()=>{try{if(video.paused){if(video.ended||shownFrame===D.count-1)video.currentTime=0;await video.play();}else video.pause();}catch(error){showError(error);}});
video.addEventListener('play',()=>{$('play').textContent='Pause';});video.addEventListener('pause',()=>{$('play').textContent=video.ended?'Replay':'Play';});video.addEventListener('ended',()=>{$('play').textContent='Replay';safeRender(video.currentTime);});
video.addEventListener('loadeddata',()=>safeRender(video.currentTime));video.addEventListener('seeked',()=>safeRender(video.currentTime));video.addEventListener('error',()=>showError('The embedded MP4 could not be decoded. Try a browser with H.264 support.'));
if('requestVideoFrameCallback' in video){
 const onFrame=(_,metadata)=>{safeRender(metadata.mediaTime);video.requestVideoFrameCallback(onFrame);};video.requestVideoFrameCallback(onFrame);
}else{
 const tick=()=>{if(!video.paused&&!video.seeking)safeRender(video.currentTime);requestAnimationFrame(tick);};requestAnimationFrame(tick);
}
$('save').addEventListener('click',()=>{if(video.readyState<2)return;const link=document.createElement('a');link.download=`car_1_${D.parts[D.ids.indexOf(Number($('part').value))].name}_frame_${String(shownFrame).padStart(3,'0')}.png`;link.href=clean.toDataURL('image/png');link.click();});
// Pure functions exposed only for local verification; no network or model calls.
window.partEditorTest={decodeRLE,clampFrame,compositePixels};
</script></body></html>'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fps", type=float, default=12)
    parser.add_argument("--results")
    args = parser.parse_args()
    print(json.dumps(build_demo(args.frames_dir, args.masks, args.registry, args.out, args.fps, args.results), indent=2))
