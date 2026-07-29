"""KaryoFlow architecture sketch — v2."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.lines import Line2D

C_BG   = "#F5F5F5"; C_GREY = "#888888"; C_WHITE = "#FFFFFF"
C_BLACK= "#222222"; C_RF   = "#4C72B0"; C_DDPM  = "#DD8452"
C_NOISE= "#64B5F6"; C_GT   = "#E53935"; C_DPM   = "#55A868"
C_HEAD = "#937860"; C_ATTN = "#8E6AB3"; C_FFN   = "#F9A825"
C_ROI  = "#5D8AA8"; C_CLS  = "#66BB6A"; C_REG   = "#FFA726"
C_OT   = "#7B3FA0"; C_RENEW = "#EF5350"

def rb(ax, x, y, w, h, fc=C_WHITE, ec=C_GREY, lw=0.8, ls="-", z=2):
    ax.add_patch(FancyBboxPatch((x,y), w, h, boxstyle="round,pad=0.06",
                               fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))

def tb(ax, x, y, w, h, txt, fc=C_WHITE, ec=C_GREY, tc="#333", sz=7, wgt="normal", z=3):
    rb(ax, x, y, w, h, fc=fc, ec=ec, z=z)
    ax.text(x+w/2, y+h/2, txt, ha="center", va="center", fontsize=sz, color=tc, weight=wgt, zorder=z+1)

def ah(ax, x1, x2, y, c=C_GREY, lw=0.8, z=3):
    ax.annotate("", xy=(x2,y), xytext=(x1,y), arrowprops=dict(arrowstyle="-|>", color=c, lw=lw), zorder=z)

def av(ax, x, y1, y2, c=C_GREY, lw=0.8, z=3):
    ax.annotate("", xy=(x,y2), xytext=(x,y1), arrowprops=dict(arrowstyle="->", color=c, lw=lw), zorder=z)

def curved_arrow(ax, x1, y1, x2, y2, c=C_GREY, lw=0.6, ls="-", z=2):
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle="->", color=c, lw=lw, ls=ls,
                connectionstyle="arc3,rad=0.3"), zorder=z)
def curved_arrow_rev(ax, x1, y1, x2, y2, c=C_GREY, lw=0.6, ls="-", z=2):
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle="->", color=c, lw=lw, ls=ls,
                connectionstyle="arc3,rad=-0.3"), zorder=z)

# ── Figure ──
fig = plt.figure(figsize=(15, 8.5))
ax = fig.add_axes([0.02,0.02,0.96,0.96])
ax.set_xlim(0,20); ax.set_ylim(0,12); ax.set_facecolor(C_BG)
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_visible(False)
ax.text(10,12,"KaryoFlow Architecture",fontsize=14,ha="center",weight="bold",color=C_BLACK)


# ═══════════════════════════════════════════════════════════════
# ZONE 1: OVERALL PIPELINE (top, y=8.5-11)
# ═══════════════════════════════════════════════════════════════
ax.text(0.2,11.3,"1. Pipeline",fontsize=10,color=C_BLACK,weight="bold")
Y1=9.0

tb(ax,0.4,Y1,1.0,1.2,"Image\n1333x800",sz=6)
tb(ax,1.9,Y1+0.1,1.2,1.0,"ResNet-50\n+ FPN\nP2-P5",fc="#E8EAF6",ec=C_GREY,sz=6)
ah(ax,1.4,1.9,Y1+0.6)

# Detection head box
dhx,dhy,dhw,dhh = 3.5,Y1-0.3,6.2,1.8
rb(ax,dhx,dhy,dhw,dhh,fc="#FAFAFA",ec=C_GREY,lw=0.8,ls="--")
ax.text(dhx+dhw/2,dhy+dhh+0.08,"DiffusionDetHead",fontsize=7,ha="center",color=C_GREY,weight="bold")

ah(ax,3.1,3.5,Y1+0.6)

# Inside the head
cy = Y1+0.6

# Noise circle
c=Circle((dhx+0.5,cy),0.22,fc=C_NOISE,ec=C_RF,lw=1.2,zorder=5); ax.add_patch(c)
ax.text(dhx+0.5,cy,"x1",fontsize=6,ha="center",va="center",color="white",weight="bold")

# t embed (simplified, no AdaLN detail)
tb(ax,dhx+1.0,cy-0.3,0.7,0.6,"t\nembed",fc="#FFF3E0",ec=C_GREY,sz=6)

# 6 cascade heads
for i in range(6):
    cx=dhx+2.0+i*0.48; is_main=(i==5)
    tb(ax,cx,cy-0.4,0.44,0.8,
       f"H{i}" if not is_main else "H5",
       fc="#EDE7F6" if is_main else C_WHITE,
       ec=C_HEAD if is_main else C_GREY, sz=5.5,
       wgt="bold" if is_main else "normal")

# "x4" annotation below cascade
ax.text(dhx+2.0+2.5*0.48,cy-0.6,"x 4 solver steps",fontsize=6,color=C_GREY,style="italic",ha="center")

# Output diamond
ox=dhx+dhw+0.45; d=np.array([[ox,cy+0.3],[ox+0.3,cy],[ox,cy-0.3],[ox-0.3,cy]])
ax.fill(d[:,0],d[:,1],fc=C_WHITE,ec=C_GREY,lw=0.8,zorder=5)
ax.text(ox,cy,"24\ncls",fontsize=5.5,ha="center",va="center",color=C_GREY,weight="bold")
ah(ax,dhx+dhw-0.1,ox-0.3,cy)


# ═══════════════════════════════════════════════════════════════
# ZONE 2: RF + OT Coupling (bottom-left, y=3-8.5)
# ═══════════════════════════════════════════════════════════════
ax.text(0.2,8.2,"2. Rectified Flow & OT Coupling",fontsize=10,color=C_RF,weight="bold")

# RF trajectory
cx0,cy0=2.5,6.2; sc=1.5
x0=np.array([cx0-1.2*sc,cy0-0.8*sc]); x1=np.array([cx0+0.8*sc,cy0+0.7*sc])
ax.plot([x0[0],x1[0]],[x0[1],x1[1]],color=C_RF,lw=2.5,zorder=5)
# DDPM ghost
ts=np.linspace(0,1,100)
bx=x0[0]+ts*(x1[0]-x0[0]); by=x0[1]+ts*(x1[1]-x0[1])
perp=np.array([-0.5,0.6]); perp=perp/np.linalg.norm(perp)
ax.plot(bx+0.3*np.sin(np.pi*ts)**1.3*perp[0], by+0.3*np.sin(np.pi*ts)**1.3*perp[1],
        color=C_DDPM,lw=1.8,ls="--",zorder=4)
# dots on line
for fr in [0.0,0.33,0.67,1.0]:
    ax.scatter(x0[0]+fr*(x1[0]-x0[0]),x0[1]+fr*(x1[1]-x0[1]),s=25,color=C_RF,edgecolor="white",lw=1.2,zorder=7)
ax.scatter(*x0,s=80,color=C_NOISE,edgecolor=C_BLACK,lw=1.5,zorder=6)
ax.scatter(*x1,s=80,color=C_GT,edgecolor=C_BLACK,lw=1.5,zorder=6)
ax.text(x0[0]-0.25,x0[1]-0.25,"x1",fontsize=7,ha="center",color=C_NOISE,weight="bold")
ax.text(x1[0]+0.25,x1[1]+0.15,"x0",fontsize=7,ha="center",color=C_GT,weight="bold")
ax.text(cx0-0.3,cy0+0.4,"x_t=(1-t)x0+tx1",fontsize=6.5,color=C_RF,ha="center",weight="bold")

# Legend
ax.legend([Line2D([],[],color=C_RF,lw=2.5),Line2D([],[],color=C_DDPM,lw=1.8,ls="--")],
          ["RF (straight)","DDPM (curved)"],fontsize=5.5,loc="upper left",
          bbox_to_anchor=(0.15,7.8),frameon=True).get_frame().set_lw(0.5)

# ── OT Coupling (between noise and GT, below the trajectory) ──
# Mini coupling diagram
ot_y = 4.2
ax.text(2.5,5.0,"OT Coupling",fontsize=7,color=C_OT,ha="center",weight="bold")
rng=np.random.default_rng(42)
n_pts=5
cl_ot,cr_ot=1.0,4.0
yn=np.linspace(ot_y-0.25,ot_y+0.25,n_pts)
yg=np.linspace(ot_y-0.2,ot_y+0.2,n_pts)+rng.normal(0,0.02,n_pts)

# Hard OT dashed (faint)
for i in range(n_pts):
    j=np.argmin(np.abs(yn[i]-yg))
    ax.plot([cl_ot,cr_ot],[yn[i],yg[j]],color=C_OT,lw=0.5,alpha=0.2,ls="--",zorder=3)
# Stochastic OT solid
perm=rng.permutation(n_pts)
for i in range(n_pts):
    ax.plot([cl_ot,cr_ot],[yn[i],yg[perm[i]]],color=C_OT,lw=1.0,alpha=0.7,zorder=4)

ax.scatter([cl_ot]*n_pts,yn,s=22,color=C_NOISE,edgecolor=C_OT,lw=0.8,zorder=5)
ax.scatter([cr_ot]*n_pts,yg,s=22,color=C_GT,edgecolor=C_OT,lw=0.8,zorder=5,marker="s")
ax.text(cl_ot,ot_y-0.55,"noise",fontsize=6,ha="center",color=C_NOISE)
ax.text(cr_ot,ot_y-0.55,"GT",fontsize=6,ha="center",color=C_GT)
ax.text(2.5,ot_y+0.55,"Sinkhorn(e=5) multinomial sample",fontsize=5.5,ha="center",color=C_OT)

# Arrow from OT coupling up to RF path
ax.annotate("",xy=(cx0,cy0-0.5),xytext=(2.5,ot_y+0.7),
            arrowprops=dict(arrowstyle="->",lw=0.6,color=C_OT,ls=":"),zorder=2)


# ═══════════════════════════════════════════════════════════════
# ZONE 3: DPM-Solver++ (bottom-center, y=3-8)
# ═══════════════════════════════════════════════════════════════
ax.text(5.5,8.2,"3. DPM-Solver++ (Inference)",fontsize=10,color=C_DPM,weight="bold")

dpm_y=6.2; dpm_x=5.5
# 4 state boxes
for i in range(4):
    sx=dpm_x+i*1.8; sy=dpm_y
    tb(ax,sx-0.35,sy-0.35,0.7,0.7,f"x({i})\nx0({i})",fc="#E8F5E9",ec=C_DPM,sz=5.5,wgt="bold")
    if i<3: ah(ax,sx+0.35,sx+1.45,sy,c=C_DPM,lw=1.3)

# Formula box
rb(ax,dpm_x,dpm_y-1.8,5.4,1.5,fc="#F1F8E9",ec=C_DPM,lw=0.8)
ax.text(dpm_x+2.7,dpm_y-1.35,
        "2nd-order multistep (1 NFE/step):\n"
        "  D1 = (x0_n - x0_{n-1}) / (t_n - t_{n-1})\n"
        "  phi1 = t_next*log(t_n/t_next) - t_n + t_next\n"
        "  x_next = (t_next/t_n)*x + (1-t_next/t_n)*x0 + phi1*D1",
        fontsize=6,color=C_BLACK,ha="center",va="center")

# Speed annotation
ax.text(dpm_x+4.5,dpm_y-2.1,"4 NFE vs 7 NFE (Heun)\n=> 1.71x faster\n13.3 FPS (A6000)",
        fontsize=6,color=C_DPM,ha="center",weight="bold")

# Top-K pruning
rb(ax,dpm_x+0.5,dpm_y-2.6,2.0,0.4,fc="#E8F5E9",ec=C_DPM,lw=0.6)
ax.text(dpm_x+1.5,dpm_y-2.4,"Top-K: 500 -> 200",fontsize=6,color=C_DPM,ha="center")

# ── Arrow: DPM-Solver++ output -> Cascade input (interaction) ──
# DPM solver at step i produces x_next, which becomes the input to cascade for step i+1
ax.annotate("",xy=(dhx+0.5,cy-0.2),xytext=(dpm_x+4.5,dpm_y-0.5),
            arrowprops=dict(arrowstyle="->",lw=0.8,color=C_DPM,ls="-",connectionstyle="arc3,rad=0.4"),zorder=2)
ax.text(dpm_x+6.0,cy-2.0,"x -> cascade\ninput",fontsize=5.5,color=C_DPM,ha="center")


# ═══════════════════════════════════════════════════════════════
# ZONE 4: Cascade Head detail (right)
# ═══════════════════════════════════════════════════════════════
ax.text(12.0,11.3,"4. Single Cascade Head (x6, shared weights)",fontsize=10,color=C_HEAD,weight="bold")
chx,chy=12.0,2.8; chw=3.4
rb(ax,chx,chy,chw,7.2,fc="#FDF8F0",ec=C_HEAD,lw=1.5)

items = [
    (chy+6.4,0.6,"bboxes [bs,500,4]  xyxy","#FFF3E0",C_GREY),
    (chy+5.4,1.0,"RoIAlign  7x7\nfeatures [N,256,7,7]\nmultiscale FPN","#E3F2FD",C_ROI),
    (chy+4.4,0.9,"Self-Attention\nMultihead(256,8 heads)\nSDPA","#F3E5F5",C_ATTN),
    (chy+3.4,0.9,"Instance Interaction\nDynamicConv\n(proposal-conditioned)","#FFF8E1",C_GREY),
    (chy+2.4,0.9,"FFN\nLinear(256->2048)\nReLU -> Linear(2048->256)","#FFF3E0",C_FFN),
    (chy+1.4,0.9,"Time Condition\nscale-shift: h = h*(1+scale)+shift\n(from t embed)","#F3E5F5",C_GREY),
]
for y_item,h_item,txt_item,fc_item,ec_item in items:
    tb(ax,chx+0.4,y_item,2.6,h_item,txt_item,fc=fc_item,ec=ec_item,sz=5.5)

# Prediction
tb(ax,chx+0.4,chy+0.2,1.3,1.1,"Cls Head\nLinear(256->24)\nFocal Loss",fc="#E8F5E9",ec=C_CLS,sz=5.5)
tb(ax,chx+1.9,chy+0.2,1.3,1.1,"Reg Head\nLinear(256->256)x3\n-> Linear(4)\napply_deltas",fc="#FFF3E0",ec=C_REG,sz=5.5)


# ═══════════════════════════════════════════════════════════════
# ZONE 5: box_renewal loop (above the cascade, curved arrow)
# ═══════════════════════════════════════════════════════════════
renew_x = dhx+4.5; renew_y = dhy+dhh+1.2
rb(ax,renew_x-1.0,renew_y,2.0,0.5,fc="#FFEBEE",ec=C_RENEW,lw=0.8)
ax.text(renew_x,renew_y+0.25,"box_renewal",fontsize=7,color=C_RENEW,ha="center",weight="bold")
ax.text(renew_x,renew_y-0.08+0.25,"low-score proposals\n-> random noise",fontsize=5.5,color=C_RENEW,ha="center")

# Curved arrow: cascade output (H5) loops back to input via renewal
# From H5 output area, looping back around to noise area
ax.annotate("",xy=(renew_x-0.8,renew_y+0.25),
            xytext=(dhx+4.5,cy+0.7),
            arrowprops=dict(arrowstyle="<-",lw=0.8,color=C_RENEW,connectionstyle="arc3,rad=0.3"),zorder=2)

ax.annotate("",xy=(dhx+0.5,cy+0.4),
            xytext=(renew_x-0.8,renew_y+0.25),
            arrowprops=dict(arrowstyle="->",lw=0.8,color=C_RENEW,connectionstyle="arc3,rad=0.2"),zorder=2)

ax.text(renew_x-0.3,renew_y+1.0,"per solver step: replace",fontsize=5.5,color=C_RENEW,ha="center")


# ═══════════════════════════════════════════════════════════════
# Interactions between zones
# ═══════════════════════════════════════════════════════════════

# RF -> DPM: the RF velocity field defines the ODE that DPM solves
ax.plot([4.5,5.2],[5.0,dpm_y],":",color=C_RF,lw=0.5,zorder=1)
ax.text(4.85,5.6,"v = x1-x0\ndefines ODE",fontsize=5.5,color=C_RF,ha="center")

# DPM -> Cascade: x from DPM step feeds into cascade head
# Already drawn above (curved arrow ZONE3->ZONE1)

# Cascade output -> OT: the predicted x0 is used to evaluate cost
ax.annotate("",xy=(chx,chy+7.0),xytext=(dhx+dhw,cy-0.3),
            arrowprops=dict(arrowstyle="->",lw=0.5,color=C_OT,ls=":"),zorder=1)

plt.savefig("karyoflow_arch_v2.png",dpi=200,bbox_inches="tight",facecolor=C_BG)
plt.savefig("karyoflow_arch_v2.pdf",bbox_inches="tight",facecolor=C_BG)
plt.close()
print("Saved: karyoflow_arch_v2.png + .pdf")
