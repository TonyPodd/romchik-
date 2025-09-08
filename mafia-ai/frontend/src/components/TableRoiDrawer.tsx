import React, { useEffect, useRef, useState } from 'react'

type P = { onSave:(poly:[number,number][])=>void, onCancel:()=>void, initial?: [number,number][] }

export default function TableRoiDrawer({ onSave, onCancel, initial }: P){
  // Храним прямоугольник в относительных координатах [0..1]
  const [rect, setRect] = useState<{x1:number,y1:number,x2:number,y2:number} | null>(() => {
    if (initial && initial.length === 4) {
      const xs = initial.map(p=>p[0]), ys = initial.map(p=>p[1])
      return { x1: Math.min(...xs), y1: Math.min(...ys), x2: Math.max(...xs), y2: Math.max(...ys) }
    }
    return null
  })
  const wrapRef = useRef<HTMLDivElement|null>(null)
  const drawing = useRef(false)

  const toRel = (clientX:number, clientY:number) => {
    const el = wrapRef.current!
    const r = el.getBoundingClientRect()
    const rx = Math.min(1, Math.max(0, (clientX - r.left) / r.width))
    const ry = Math.min(1, Math.max(0, (clientY - r.top) / r.height))
    return { rx, ry }
  }

  const onDown = (e: React.MouseEvent) => {
    if (!wrapRef.current) return
    drawing.current = true
    const { rx, ry } = toRel(e.clientX, e.clientY)
    setRect({ x1: rx, y1: ry, x2: rx, y2: ry })
  }
  const onMove = (e: React.MouseEvent) => {
    if (!drawing.current || !rect) return
    const { rx, ry } = toRel(e.clientX, e.clientY)
    setRect({ x1: rect.x1, y1: rect.y1, x2: rx, y2: ry })
  }
  const onUp = () => { drawing.current = false }

  const polyFromRect = (): [number,number][] | null => {
    if (!rect) return null
    const x1 = Math.min(rect.x1, rect.x2), x2 = Math.max(rect.x1, rect.x2)
    const y1 = Math.min(rect.y1, rect.y2), y2 = Math.max(rect.y1, rect.y2)
    return [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
  }

  return (
    <div className="neon-card" style={{padding:12, marginTop:12}}>
      <h2>Ручная разметка стола</h2>
      <p className="neon-sub">Потяните мышью по видео, чтобы выделить область стола. Сохраните — и контур появится в потоке.</p>

      <div className="neon-card" style={{overflow:'hidden', maxWidth:'100%', position:'relative'}}>
        <div
          ref={wrapRef}
          style={{position:'relative'}}
          onMouseDown={onDown} onMouseMove={onMove} onMouseUp={onUp} onMouseLeave={onUp}
        >
          <img src="http://127.0.0.1:8000/video/mjpeg" alt="camera" style={{display:'block', width:'100%', height:'auto'}} />
          {rect && (
            <div
              style={{
                position:'absolute',
                left: `${Math.min(rect.x1, rect.x2)*100}%`,
                top: `${Math.min(rect.y1, rect.y2)*100}%`,
                width: `${Math.abs(rect.x2-rect.x1)*100}%`,
                height: `${Math.abs(rect.y2-rect.y1)*100}%`,
                border:'2px dashed #7bffea',
                boxShadow:'0 0 12px rgba(123,255,234,0.5) inset',
                pointerEvents:'none'
              }}
            />
          )}
        </div>
      </div>

      <div className="flex" style={{gap:12, marginTop:12}}>
        <button className="neon-btn" onClick={()=>{
          const p = polyFromRect()
          if (p) onSave(p)
        }} disabled={!rect}>Сохранить</button>
        <button className="neon-btn" onClick={onCancel}>Отмена</button>
      </div>
    </div>
  )
}
