import React, { useEffect, useRef, useState } from 'react'
import CameraPanel from './CameraPanel'
import NeonButton from './NeonButton'

export default function CalibratePanel(
  {
    initialRatio = 0.8,
    onSave, onCancel
  }: {
    initialRatio?: number,
    onSave: (ratio:number)=>void,
    onCancel: ()=>void
  }
){
  const [ratio, setRatio] = useState(initialRatio)
  const wrapRef = useRef<HTMLDivElement | null>(null)
  const lineRef = useRef<HTMLDivElement | null>(null)
  const dragging = useRef(false)

  useEffect(()=>{
    const onMove = (e: MouseEvent) => {
      if(!dragging.current || !wrapRef.current) return
      const rect = wrapRef.current.getBoundingClientRect()
      const y = Math.min(rect.bottom, Math.max(rect.top, e.clientY))
      const rel = (y - rect.top) / rect.height
      setRatio(Math.max(0.5, Math.min(0.98, rel)))
    }
    const onUp = () => { dragging.current = false }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    return () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
  }, [])

  return (
    <div className="neon-card" style={{padding:16, marginTop:12}}>
      <h2>Калибровка стола</h2>
      <p className="neon-sub">Потяните линию или используйте ползунок. Чем ниже линия — тем выше порог «кулак на столе».</p>

      <div ref={wrapRef}>
        <CameraPanel>
          <div
            ref={lineRef}
            className="overlay-line"
            style={{ top: `${ratio*100}%` }}
            onMouseDown={()=> (dragging.current = true)}
            title="Перетащите, чтобы изменить линию стола"
          />
        </CameraPanel>
      </div>

      <div className="slider-row" style={{marginTop:12}}>
        <span className="chip">ratio</span>
        <input type="range" min={0.5} max={0.98} step={0.005}
               value={ratio}
               onChange={e=> setRatio(parseFloat(e.target.value))}/>
        <code>{ratio.toFixed(3)}</code>
      </div>

      <div className="flex" style={{gap:12, marginTop:12}}>
        <NeonButton onClick={()=> onSave(ratio)}>Сохранить</NeonButton>
        <NeonButton onClick={onCancel}>Отмена</NeonButton>
      </div>
    </div>
  )
}
