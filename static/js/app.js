let img = new Image();
let trees = [];
let stats = null;
let fid = null;
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
const drop = document.getElementById('drop');
const fIn = document.getElementById('fIn');
const ov = document.getElementById('ov');
const pf = document.getElementById('pf');
const tip = document.getElementById('tip');

function $(id){return document.getElementById(id)}
['Spacing','ValMax','Texture','Exg','Erosion','Radius'].forEach(k=>{
  const input = $('p'+k), out = $('v'+k);
  if(input && out) input.addEventListener('input',()=>out.textContent=input.value);
});

drop.addEventListener('click',()=>fIn.click());
drop.addEventListener('dragover',e=>{e.preventDefault();drop.style.borderColor='#bcff7a'});
drop.addEventListener('dragleave',()=>drop.style.borderColor='#59ff40');
drop.addEventListener('drop',e=>{e.preventDefault();drop.style.borderColor='#59ff40'; if(e.dataTransfer.files.length) upload(e.dataTransfer.files[0]);});
fIn.addEventListener('change',()=>{if(fIn.files.length) upload(fIn.files[0]);});

function showLoad(msg){ov.classList.add('show');$('ovMsg').textContent=msg;pf.style.width='15%';setTimeout(()=>pf.style.width='65%',300)}
function hideLoad(){pf.style.width='100%';setTimeout(()=>ov.classList.remove('show'),250)}

async function upload(file){
  const fd = new FormData(); fd.append('file', file);
  fd.append('peak_spacing', $('pSpacing').value);
  fd.append('val_max', $('pValMax').value);
  fd.append('texture_thr', $('pTexture').value);
  fd.append('exg_min', $('pExg').value);
  fd.append('erosion_iter', $('pErosion').value);
  fd.append('min_radius', $('pRadius').value);
  fd.append('gauss_sigma', 2);
  showLoad('Detectando copas de árboles...');
  try{
    const res = await fetch('/upload',{method:'POST',body:fd});
    const data = await res.json();
    if(!res.ok || !data.ok) throw new Error(data.error || 'No se pudo procesar la imagen');
    fid=data.fid; trees=data.trees; stats=data.stats;
    img.onload=()=>{cv.width=img.width;cv.height=img.height;draw();showResults();hideLoad();};
    img.src='data:image/jpeg;base64,'+data.orig_b64;
  }catch(err){hideLoad();alert('Error: '+err.message);}
}
function showResults(){
  $('uploadSec').style.display='none'; $('toolbar').style.display='flex'; $('wrap').style.display='block'; $('foot').style.display='flex'; $('hdrStats').style.display='flex';
  $('sDim').textContent=stats.processed_size; $('sH').textContent=stats.healthy; $('sM').textContent=stats.moderate; $('sD').textContent=stats.dry; $('sT').textContent=stats.total+' árboles';
  $('btnCsv').onclick=()=>{ if(fid) location.href='/csv/'+fid; };
}
function colorFor(h){return h==='Dry'?'#ff3333':(h==='Moderate'?'#ffd84b':'#66ff33')}
function draw(){
  if(!img.complete) return;
  ctx.clearRect(0,0,cv.width,cv.height); ctx.drawImage(img,0,0);
  const showBox=$('chkBox').checked, showLabel=$('chkLabel').checked, showHealth=$('chkHealth').checked, showHandles=$('chkHandles').checked, showCross=$('chkCross').checked;
  ctx.font='bold 15px JetBrains Mono, monospace'; ctx.lineWidth=3;
  trees.forEach((t,i)=>{
    const c=colorFor(t.health); const x=t.x0,y=t.y0,w=t.x1-t.x0,h=t.y1-t.y0;
    if(showBox){ctx.strokeStyle=c;ctx.strokeRect(x,y,w,h)}
    if(showHandles){drawHandles(x,y,w,h,c)}
    if(showCross){ctx.strokeStyle='#ff2727';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(t.cx-10,t.cy);ctx.lineTo(t.cx+10,t.cy);ctx.moveTo(t.cx,t.cy-10);ctx.lineTo(t.cx,t.cy+10);ctx.stroke();ctx.lineWidth=3}
    if(showLabel){
      const text=(t.label||('a'+(i+1)))+(showHealth?' '+t.health:'');
      const m=ctx.measureText(text); const th=19; const ty=Math.max(0,y-th-4);
      ctx.fillStyle=c; ctx.fillRect(x,ty,m.width+12,th+3);
      ctx.fillStyle=t.health==='Dry'?'#fff':'#061006'; ctx.fillText(text,x+6,ty+16);
    }
  });
}
function drawHandles(x,y,w,h,c){
  ctx.fillStyle='#fff';ctx.strokeStyle=c;ctx.lineWidth=2; const pts=[[x,y],[x+w,y],[x,y+h],[x+w,y+h],[x+w/2,y],[x+w/2,y+h],[x,y+h/2],[x+w,y+h/2]];
  pts.forEach(p=>{ctx.fillRect(p[0]-4,p[1]-4,8,8);ctx.strokeRect(p[0]-4,p[1]-4,8,8)}); ctx.lineWidth=3;
}
cv.addEventListener('mousemove',e=>{
  const r=cv.getBoundingClientRect(); const sx=cv.width/r.width, sy=cv.height/r.height; const x=(e.clientX-r.left)*sx, y=(e.clientY-r.top)*sy;
  const t=trees.find(a=>x>=a.x0 && x<=a.x1 && y>=a.y0 && y<=a.y1);
  if(!t){tip.style.display='none';return}
  tip.style.display='block'; tip.style.left=e.clientX+12+'px'; tip.style.top=e.clientY+12+'px';
  tip.innerHTML=`${t.label} · ${t.health}<br>Radio: ${t.radius}px · ExG: ${t.exg}`;
});
cv.addEventListener('mouseleave',()=>tip.style.display='none');
function dlPNG(){const a=document.createElement('a');a.download='conteo_arboles.png';a.href=cv.toDataURL('image/png');a.click();}
function dlJSON(){const blob=new Blob([JSON.stringify({stats,trees},null,2)],{type:'application/json'});const a=document.createElement('a');a.download='arboles.json';a.href=URL.createObjectURL(blob);a.click();URL.revokeObjectURL(a.href)}
function reset(){location.reload()}
