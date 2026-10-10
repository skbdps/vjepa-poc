"""Replay a named-part edit on saved predicted masks, without rerunning a model.

Pixel edits are deterministic compositing, not generative video synthesis.
Unselected predicted parts are protected even when predicted masks overlap.
The safety guarantee is on RGB arrays before lossy video encoding, and is only
as anatomically accurate as the predicted masks. Ground truth is never an input.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def apply_recolor(rgb, part_masks, ids, selected_id, color_rgb=(35,140,240), strength=.75):
    rgb=np.asarray(rgb)
    part_masks=np.asarray(part_masks,dtype=bool)
    ids=list(map(int,ids))
    if selected_id not in ids: raise ValueError(f'Unknown part id {selected_id}; available {ids}')
    if rgb.ndim!=3 or rgb.shape[-1]!=3 or rgb.dtype!=np.uint8: raise ValueError('Expected RGB uint8 image')
    if part_masks.shape!=(len(ids),*rgb.shape[:2]): raise ValueError('Mask and image sizes differ')
    if not 0<=strength<=1: raise ValueError('strength must be in [0,1]')
    color=np.asarray(color_rgb,dtype=np.float32)
    if color.shape!=(3,) or np.any(color<0) or np.any(color>255): raise ValueError('Color must be three0..255 RGB values')
    target_index=ids.index(int(selected_id))
    other=[j for j in range(len(ids)) if j!=target_index]
    protected=part_masks[other].any(0) if other else np.zeros(rgb.shape[:2],bool)
    effective=part_masks[target_index]&~protected
    luminance=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255.
    tint=color[None,None]*(.35+.65*luminance[...,None])
    output=rgb.copy()
    output[effective]=np.clip((1-strength)*rgb[effective]+strength*tint[effective],0,255).astype(np.uint8)
    changed=np.any(output!=rgb,axis=-1)
    if (changed&~effective).any() or (changed&protected).any(): raise AssertionError('Edit escaped selected permitted mask')
    return output,effective,protected


def edit_masks(predictions,ids,selected_id):
    ids=list(map(int,ids));j=ids.index(int(selected_id))
    others=[i for i in range(len(ids)) if i!=j]
    masks=np.asarray(predictions,dtype=bool)
    protected=masks[:,others].any(1) if others else np.zeros(masks.shape[0:1]+masks.shape[-2:],bool)
    return masks[:,j]&~protected


def replay(frames_dir,masks_file,output,target_id=1,color_rgb=(35,140,240),strength=.75,fps=12):
    from benchmark import write_video
    frames=sorted(Path(frames_dir).glob('*.jpg'))
    saved=np.load(masks_file);masks=saved['masks'];ids=saved['ids'].tolist()
    if len(frames)!=len(masks): raise ValueError('Frame/mask count mismatch')
    rendered=[];checks=[]
    for t,path in enumerate(frames):
        rgb=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
        edited,effective,protected=apply_recolor(rgb,masks[t],ids,target_id,color_rgb,strength)
        changed=np.any(edited!=rgb,axis=-1)
        checks.append({'frame':t,'changed_pixels':int(changed.sum()),'effective_mask_pixels':int(effective.sum()),
                       'changes_outside_selected_mask':int((changed&~effective).sum()),
                       'changes_inside_protected_predicted_parts':int((changed&protected).sum())})
        rendered.append(edited)
    write_video(output,rendered,fps=fps)
    report={'target_id':int(target_id),'color_rgb':list(color_rgb),'strength':strength,
            'mask_source':str(masks_file),'operation':'shading-preserving deterministic recolor',
            'guarantee_scope':'array pixels before lossy encoding; predicted-part protection, not ground-truth accuracy',
            'frames':checks}
    Path(output).with_suffix('.json').write_text(json.dumps(report,indent=2))
    return report


def self_test():
    rgb=np.full((16,16,3),100,np.uint8);masks=np.zeros((2,16,16),bool)
    masks[0,2:12,2:12]=True;masks[1,8:14,8:14]=True
    a,e,p=apply_recolor(rgb,masks,[11,29],11)
    assert np.any(a[e]!=rgb[e]) and np.array_equal(a[p],rgb[p]) and np.array_equal(a[~e],rgb[~e])
    b,_,_=apply_recolor(rgb,masks[[1,0]],[29,11],11)
    assert np.array_equal(a,b),'IDs must not depend on array order'
    return {'named_id_stable':True,'overlapping_unselected_parts_protected':True,'outside_pixels_unchanged_before_encoding':True}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--frames-dir',required=True);p.add_argument('--masks',required=True)
    p.add_argument('--out',required=True);p.add_argument('--target-id',type=int,default=1)
    p.add_argument('--color',type=int,nargs=3,default=[35,140,240]);p.add_argument('--strength',type=float,default=.75)
    a=p.parse_args();print(json.dumps(replay(a.frames_dir,a.masks,a.out,a.target_id,a.color,a.strength),indent=2))
