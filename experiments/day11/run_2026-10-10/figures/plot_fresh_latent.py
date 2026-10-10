"""Reproduce the descriptive fresh latent plot from the saved independent audit."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path(__file__).resolve().parent.parent
a=json.loads((root/'audits/fresh/latent_metric_audit.json').read_text())
rows=[r for r in a['paired_comparisons'] if r['translator']=='cnn']
assert len(rows)==8
def balanced(r,key):return .5*(r[key]['source_hole']['ratio']+r[key]['destination']['ratio'])
c=np.array([balanced(r,'primary') for r in rows]);n=np.array([balanced(r,'no_F') for r in rows])
linear=[balanced(r,'primary') for r in a['paired_comparisons'] if r['translator']=='linear']
fig,(ax,bx)=plt.subplots(1,2,figsize=(11.5,4.5),gridspec_kw={'width_ratios':[1,2.3]})
colors=['#637280','#1867a8','#a16b37']
vals=[float(n.mean()),float(c.mean()),float(np.mean(linear))]
ax.bar(['Copy + fill','CNN + transport','Linear + transport'],vals,color=colors,width=.62)
ax.set_ylim(0,.11);ax.set_ylabel('Balanced regional latent error ratio')
ax.tick_params(axis='x',labelrotation=23,labelsize=9)
for i,v in enumerate(vals):ax.text(i,v+.002,f'{v:.4f}',ha='center',fontsize=10)
ax.set_title('Mean across the same eight scenes',fontsize=11)
x=np.arange(8)
bx.bar(x-.18,n,width=.34,label='Copy + fill, no learned F',color=colors[0])
bx.bar(x+.18,c,width=.34,label='CNN + transported residual',color=colors[1])
bx.set_xticks(x,[r['scene'].replace('test_','').replace('_dx',' / ') for r in rows],rotation=30,ha='right',fontsize=9)
bx.set_ylim(0,max(n.max(),c.max())*1.19);bx.set_title('Per-scene comparison: CNN wins 5 of 8',fontsize=11)
bx.legend(frameon=False,fontsize=9,loc='upper left')
for ar in (ax,bx):
 ar.spines[['top','right']].set_visible(False);ar.set_axisbelow(True);ar.grid(axis='y',color='#e6e9ed',linewidth=.7)
fig.suptitle('Fresh latent test: a small, mixed learned advantage',x=.05,ha='left',fontsize=15,fontweight='bold')
fig.text(.05,.025,'Lower is better. Ratio 1 = unedited source. Average of source-hole and destination ratios; each region also assessed separately.\nFrozen eight-scene image pilot. These are conditioning-latent measurements, not generated-image quality.',fontsize=9,color='#424b55')
fig.subplots_adjust(left=.07,right=.99,top=.81,bottom=.27,wspace=.25)
fig.savefig(root/'figures/fresh_latent_comparison.png',dpi=180,facecolor='white')
fig.savefig(root/'figures/fresh_latent_comparison.pdf',facecolor='white')
print(json.dumps({'means':vals,'cnn_paired_wins':int((c<n).sum())}))
