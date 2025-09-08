import React, { useRef, useState } from 'react'
import NeonButton from './NeonButton'

export default function LineThreshold({initial=0.8,onSave,onCancel}:{initial?:number,onSave:(r:number)=>void,onCancel:()=>void}){
  const [r,setR]=useState(initial)
  const wrap=useRef<HTMLDivElement|null>(null)
  const dragging=useRef(false)
  const move=(e:React.MouseEvent)=>{
    if(!dragging.current||!wrap.current) return
    const rc=wrap.current.getBoundingClientRect()
    const y=Math.min(rc.bottom,Math.max(rc.top,e.clientY))
    const rel=(y-rc.top)/rc.height
    setR(Math.max(0.5,Math.min(0.98,rel)))
  }
  return (
    <div className="section" onMouseMove={move} onMouseUp={()=>dragging.current=false}>
      <h2>Линия-порог (fallback)</h2>
      <p className="chip" style={{marginTop:6}}>Если стол не размечен, кулак считается «на столе» ниже этой линии</p>
      <div className="card" style={{overflow:'hidden', position:'relative', marginTop:12}}>
        <div ref={wrap} style={{position:'relative'}}>
          <img className="video" src="http://127.0.0.1:8000/video/mjpeg" alt="camera" />
          <div style={{position:'absolute',left:0,right:0,top:`${r*100}%`,height:0,borderTop:'2px dashed rgba(123,255,234,.9)'}} onMouseDown={()=>dragging.current=true}/>
        </div>
      </div>
      <div style={{display:'flex',gap:10,alignItems:'center',marginTop:12}}>
        <input type="range" min={0.5} max={0.98} step={0.005} value={r} onChange={e=>setR(parseFloat(e.target.value))}/>
        <code>{r.toFixed(3)}</code>
        <NeonButton onClick={()=>onSave(r)}>Сохранить</NeonButton>
        <NeonButton className="ghost" onClick={onCancel}>Отмена</NeonButton>
      </div>
    </div>
  )
}
