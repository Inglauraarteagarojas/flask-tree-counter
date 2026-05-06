
"""
CONTADOR DE ÁRBOLES — UMNG Cajicá
Flask + NumPy + SciPy | Detección + Modo Manual
Laura Mercedes Arteaga Rojas — UMNG — Mayo 2026
"""
import os
import os,uuid,json,base64
from io import BytesIO
from flask import Flask,render_template,request,jsonify,Response
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import (uniform_filter,label,find_objects,
    binary_closing,binary_opening,binary_erosion,binary_dilation,
    gaussian_filter,maximum_filter,distance_transform_edt)

app=Flask(__name__)
app.config['UPLOAD_FOLDER']=os.path.join(os.path.dirname(__file__),'static','uploads')
app.config['RESULTS_FOLDER']=os.path.join(os.path.dirname(__file__),'static','results')
app.config['MAX_CONTENT_LENGTH']=300*1024*1024
ALLOWED={'png','jpg','jpeg','tif','tiff','bmp'}
os.makedirs(app.config['UPLOAD_FOLDER'],exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'],exist_ok=True)

def allowed_file(fn):
    return '.' in fn and fn.rsplit('.',1)[1].lower() in ALLOWED

def detect_trees(path,params=None):
    p=dict(max_dim=2200,hue_min=35,hue_max=170,sat_min=8,val_max=62,exg_min=4,
           texture_thr=4,erosion_iter=1,peak_spacing=22,min_radius=4,gauss_sigma=2)
    if params:
        for k,v in params.items():
            if k in p:p[k]=type(p[k])(v)
    img=Image.open(path).convert('RGB');ow,oh=img.size;scale=1.0
    if max(ow,oh)>p['max_dim']:
        scale=p['max_dim']/max(ow,oh);img=img.resize((int(ow*scale),int(oh*scale)),Image.LANCZOS)
    arr=np.array(img).astype(float);h,w=arr.shape[:2]
    r,g,b=arr[:,:,0],arr[:,:,1],arr[:,:,2]
    exg=2*g-r-b;mx=np.maximum(np.maximum(r,g),b);delta=mx-np.minimum(np.minimum(r,g),b)
    with np.errstate(invalid='ignore',divide='ignore'):sat=np.where(mx>0,(delta/mx)*100,0)
    val=(mx/255)*100;hue=np.zeros_like(r)
    mr=(mx==r)&(delta>0);mg=(mx==g)&(delta>0)&~mr;mb=(mx==b)&(delta>0)&~mr&~mg
    with np.errstate(invalid='ignore',divide='ignore'):
        hue[mr]=60*(((g[mr]-b[mr])/delta[mr])%6);hue[mg]=60*((b[mg]-r[mg])/delta[mg]+2);hue[mb]=60*((r[mb]-g[mb])/delta[mb]+4)
    hue[hue<0]+=360
    gm=uniform_filter(g,size=7);gs=np.sqrt(np.maximum(uniform_filter(g*g,size=7)-gm**2,0))
    cm=((hue>=p['hue_min'])&(hue<=p['hue_max'])&(sat>=p['sat_min'])&(val>=5)&(val<=p['val_max'])&
        (exg>=p['exg_min'])&(g>r*0.8)&(g>b*0.95)&(gs>p['texture_thr']))
    # El césped suele ser homogéneo: mucho brillo y poca textura.
    # Se elimina para dejar preferiblemente copas con textura/sombra.
    grass=(val>46)&(gs<4.2)&(sat<38);cm=cm&~grass
    ce=binary_erosion(cm,structure=np.ones((3,3)),iterations=p['erosion_iter'])
    ce=binary_opening(ce,structure=np.ones((3,3)),iterations=1)
    ce=binary_dilation(ce,structure=np.ones((2,2)),iterations=1)
    dist=distance_transform_edt(ce);ds=gaussian_filter(dist,sigma=p['gauss_sigma'])
    lm=maximum_filter(ds,size=p['peak_spacing']);peaks=(ds==lm)&(ds>=max(2.0,p['min_radius']*0.45))
    pl,_=label(peaks);ps=find_objects(pl)
    trees=[]
    for i,sl in enumerate(ps):
        if sl is None:continue
        pk=pl[sl]==(i+1);pys,pxs=np.where(pk)
        pcy=int(sl[0].start+pys.mean());pcx=int(sl[1].start+pxs.mean())
        radii=[]
        for ai in range(16):
            ang=2*np.pi*ai/16;bd=0
            for d in range(2,55):
                px=int(round(pcx+np.cos(ang)*d));py=int(round(pcy+np.sin(ang)*d))
                if 0<=px<w and 0<=py<h and cm[py,px]:bd=d
                elif d>bd+3:break
            radii.append(bd)
        if not radii:continue
        mr_=float(np.median(radii))
        if mr_<p['min_radius']:continue
        br=mr_*1.15;bx0=max(0,int(pcx-br));by0=max(0,int(pcy-br))
        bx1=min(w-1,int(pcx+br));by1=min(h-1,int(pcy+br))
        bw_=bx1-bx0;bh_=by1-by0
        if bw_<6 or bh_<6:continue
        mi=cm[by0:by1,bx0:bx1];fill=mi.sum()/max(1,bw_*bh_)
        if fill<0.06:continue
        me=float(exg[by0:by1,bx0:bx1][mi].mean()) if mi.sum()>0 else 0
        ms_=float(sat[by0:by1,bx0:bx1][mi].mean()) if mi.sum()>0 else 0
        health='Healthy' if me>12 and ms_>16 else ('Moderate' if me>4 else 'Dry')
        trees.append(dict(cx=pcx,cy=pcy,x0=bx0,y0=by0,x1=bx1,y1=by1,
                         radius=round(mr_,1),exg=round(me,1),health=health,source='auto'))
    ts=sorted(trees,key=lambda t:t['radius'],reverse=True);keep=[]
    for t in ts:
        ok=True
        for k in keep:
            ix0=max(t['x0'],k['x0']);iy0=max(t['y0'],k['y0']);ix1=min(t['x1'],k['x1']);iy1=min(t['y1'],k['y1'])
            if ix0<ix1 and iy0<iy1:
                inter=(ix1-ix0)*(iy1-iy0);a1=(t['x1']-t['x0'])*(t['y1']-t['y0']);a2=(k['x1']-k['x0'])*(k['y1']-k['y0'])
                if inter/(a1+a2-inter)>0.2:ok=False;break
        if ok:keep.append(t)
    trees=keep;trees.sort(key=lambda t:(t['cy']//35,t['cx']))
    for i,t in enumerate(trees):t['id']=i+1;t['label']=f"a{i+1}"
    result=draw_det(img,trees)
    hc={};
    for t in trees:hc[t['health']]=hc.get(t['health'],0)+1
    stats=dict(original_size=f"{ow}x{oh}",processed_size=f"{w}x{h}",scale=round(scale*100,1),
               total=len(trees),healthy=hc.get('Healthy',0),moderate=hc.get('Moderate',0),dry=hc.get('Dry',0),params=p)
    return trees,result,stats

def draw_det(img,trees):
    r=img.copy();d=ImageDraw.Draw(r,'RGBA')
    C={'Healthy':(102,255,51),'Moderate':(255,215,0),'Dry':(255,51,51)}
    for t in trees:
        c=C.get(t['health'],C['Healthy']);x0,y0,x1,y1=t['x0'],t['y0'],t['x1'],t['y1']
        d.rectangle([x0,y0,x1,y1],outline=c,width=2)
        hs=5
        for hx,hy in [(x0,y0),(x1,y0),(x0,y1),(x1,y1),((x0+x1)//2,y0),((x0+x1)//2,y1),(x0,(y0+y1)//2),(x1,(y0+y1)//2)]:
            d.rectangle([hx-hs//2,hy-hs//2,hx+hs//2,hy+hs//2],fill=(255,255,255),outline=c)
        cl=min(8,(x1-x0)//3)
        d.line([(t['cx']-cl,t['cy']),(t['cx']+cl,t['cy'])],fill=(255,34,34),width=1)
        d.line([(t['cx'],t['cy']-cl),(t['cx'],t['cy']+cl)],fill=(255,34,34),width=1)
        txt=f"{t['label']} {t['health']}";bb=d.textbbox((0,0),txt);tw=bb[2]-bb[0];th=bb[3]-bb[1]
        tc=(0,0,0) if t['health']!='Dry' else (255,255,255)
        d.rectangle([x0,y0-th-6,x0+tw+8,y0-1],fill=c+(220,))
        d.text((x0+4,y0-th-4),txt,fill=tc)
    return r

def to_b64(im):buf=BytesIO();im.save(buf,format='JPEG',quality=85);return base64.b64encode(buf.getvalue()).decode()

@app.route('/')
def index():return render_template('index.html')

@app.route('/upload',methods=['POST'])
def upload():
    if 'file' not in request.files:return jsonify(error='No file'),400
    f=request.files['file']
    if not f.filename or not allowed_file(f.filename):return jsonify(error='Formato no válido'),400
    fid=str(uuid.uuid4())[:8];fp=os.path.join(app.config['UPLOAD_FOLDER'],f"{fid}.{f.filename.rsplit('.',1)[1].lower()}")
    f.save(fp)
    params={k:float(v) if '.' in v else int(v) for k in
            ['peak_spacing','val_max','texture_thr','exg_min','erosion_iter','min_radius','gauss_sigma']
            if (v:=request.form.get(k))}
    try:trees,result,stats=detect_trees(fp,params)
    except Exception as e:return jsonify(error=str(e)),500
    result.save(os.path.join(app.config['RESULTS_FOLDER'],f"{fid}_result.jpg"),quality=90)
    with open(os.path.join(app.config['RESULTS_FOLDER'],f"{fid}_trees.json"),'w') as jf:json.dump(dict(trees=trees,stats=stats),jf)
    orig=Image.open(fp).convert('RGB')
    if max(orig.size)>1800:sc=1800/max(orig.size);orig=orig.resize((int(orig.width*sc),int(orig.height*sc)),Image.LANCZOS)
    return jsonify(ok=True,fid=fid,trees=trees,stats=stats,orig_b64=to_b64(orig),result_b64=to_b64(result))

@app.route('/csv/<fid>')
def csv_dl(fid):
    jp=os.path.join(app.config['RESULTS_FOLDER'],f"{fid}_trees.json")
    if not os.path.exists(jp):return "Not found",404
    with open(jp) as f:data=json.load(f)
    rows=['ID,Label,Source,CentroX,CentroY,Radio,BboxW,BboxH,ExG,Health']
    for t in data['trees']:
        rows.append(f"{t['id']},{t['label']},{t.get('source','auto')},{t['cx']},{t['cy']},{t['radius']},"
                     f"{t['x1']-t['x0']},{t['y1']-t['y0']},{t['exg']},{t['health']}")
    return Response('\n'.join(rows),mimetype='text/csv',headers={'Content-Disposition':f'attachment;filename=arboles_{fid}.csv'})



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
