import React, { useEffect, useRef, useState } from 'react'
import NeonButton from './NeonButton'

type Pt = { x:number, y:number }
type Props = {
  initial?: [number,number][],
  onSave: (poly:[number,number][])=>void,
  onCancel: ()=>void
}

export default function TablePolyDrawer({ initial, onSave, onCancel }: Props){
  const [pts, setPts] = useState<Pt[]>(()=> initial ? initial.map(([x,y])=>({x,y})) : [])
  const [closed, setClosed] = useState(false)
  const [dragIdx, setDragIdx] = useState<number | null>(null)
  const wrapRef = useRef<HTMLDivElement|null>(null)

  useEffect(() => {
    if (initial && initial.length >= 3) {
      setPts(initial.map(([x, y]) => ({ x, y })))
      setClosed(true)
    }
  }, [initial])

  const toRel = (clientX:number, clientY:number) => {
    const r = wrapRef.current!.getBoundingClientRect()
    return {
      x: Math.min(1, Math.max(0, (clientX - r.left) / r.width)),
      y: Math.min(1, Math.max(0, (clientY - r.top) / r.height))
    }
  }

  const onClick = (e: React.MouseEvent) => {
    if (closed) return
    const p = toRel(e.clientX, e.clientY)
    setPts(prev => [...prev, p])
  }

  const onDblClick = () => {
    if (pts.length >= 3) setClosed(true)
  }

  const onMouseMove = (e: React.MouseEvent) => {
    if (dragIdx==null) return
    const p = toRel(e.clientX, e.clientY)
    setPts(prev => prev.map((q,i)=> i===dragIdx ? p : q))
  }

  const startDrag = (i:number) => (e: React.MouseEvent) => {
    e.stopPropagation()
    setDragIdx(i)
  }
  const stopDrag = () => setDragIdx(null)

  const save = () => {
    if (!closed || pts.length < 3) return
    onSave(pts.map(p=>[p.x, p.y]))
  }

  return (
    <div className="neon-card" style={{padding:12, marginTop:12}}>
      <h2>Ручная обводка стола (многоугольник)</h2>
      <p className="neon-sub">Кликайте по контуру стола, двойной клик — замкнуть. Точки можно перетаскивать.</p>

      <div className="neon-card" style={{overflow:'hidden', maxWidth:'100%', position:'relative'}}>
        <div
          ref={wrapRef}
          style={{position:'relative'}}
          onClick={onClick}
          onDoubleClick={onDblClick}
          onMouseMove={onMouseMove}
          onMouseUp={stopDrag}
          onMouseLeave={stopDrag}
        >
          <img src="http://127.0.0.1:8000/video/mjpeg" alt="camera" style={{display:'block', width:'100%', height:'auto'}} />

          {/* линии / заливка */}
          <svg style={{position:'absolute', left:0, top:0, width:'100%', height:'100%', pointerEvents:'none'}}>
            {pts.length >= 2 && (
              <polyline
                points={pts.map(p=>`${p.x*100}%,${p.y*100}%`).join(' ')}
                fill={closed ? 'rgba(40,200,255,0.12)' : 'none'}
                stroke="rgba(255,200,50,0.9)"
                strokeWidth="2"
              />
            )}
            {closed && pts.length>=3 && (
              <line
                x1={`${pts[pts.length-1].x*100}%`} y1={`${pts[pts.length-1].y*100}%`}
                x2={`${pts[0].x*100}%`} y2={`${pts[0].y*100}%`}
                stroke="rgba(255,200,50,0.9)"
                strokeWidth="2"
              />
            )}
          </svg>

          {/* вершины */}
          {pts.map((p,i)=>(
            <div key={i}
              onMouseDown={startDrag(i)}
              style={{
                position:'absolute',
                left:`calc(${p.x*100}% - 6px)`,
                top: `calc(${p.y*100}% - 6px)`,
                width:12, height:12, borderRadius:12,
                background:'#7bffea', boxShadow:'0 0 8px rgba(123,255,234,0.8)',
                cursor:'grab', pointerEvents:'auto'
              }}
              title={`vertex ${i+1}`}
            />
          ))}
        </div>
      </div>

      <div className="flex" style={{gap:12, marginTop:12, flexWrap:'wrap'}}>
        <button className="neon-btn" onClick={()=>{ setPts([]); setClosed(false) }}>Сбросить</button>
        <button className="neon-btn" onClick={()=> setClosed(false)} disabled={!closed}>Редактировать</button>
        <button className="neon-btn" onClick={save} disabled={!closed || pts.length<3}>Сохранить</button>
        <button className="neon-btn" onClick={onCancel}>Отмена</button>
        <span className="chip">вершин: {pts.length} {closed ? '(замкнуто)' : ''}</span>
      </div>
    </div>
  )
}
