import React, { useEffect, useRef, useState } from 'react'
import './index.css'
import CameraPanel from './components/CameraPanel'
import NeonButton from './components/NeonButton'
import Modal from './components/Modal'
import TablePolyDrawer from './components/TablePolyDrawer'
import LineThreshold from './components/LineThreshold'
import EnrollPanel from './components/EnrollPanel'
import { getHealth, startVideo, stopVideo, autoDetectTable, setTableROI } from './api'

export default function App(){
  const wsRef=useRef<WebSocket|null>(null)
  const [wsOk,setWsOk]=useState(false)
  const [videoRunning,setVideoRunning]=useState(false)
  const [lastGesture,setLastGesture]=useState('—')
  const [tableRatio,setTableRatio]=useState(0.80)

  const [polyModal,setPolyModal]=useState(false)
  const [lineModal,setLineModal]=useState(false)
  const [enrollModal,setEnrollModal]=useState(false)
  const [toast,setToast]=useState<string|null>(null)

  useEffect(()=>{
    if(wsRef.current) return
    const ws=new WebSocket('ws://127.0.0.1:8000/ws'); wsRef.current=ws
    ws.addEventListener('open',()=>setWsOk(true))
    ws.addEventListener('close',()=>setWsOk(false))
    ws.addEventListener('message',(e)=>{
      try{
        const m=JSON.parse(e.data as string)
        if(m.type==='gesture'){
          const hand = m.hands?.find?.(()=>true)
          setLastGesture(`digit=${m.digit??'·'} fist=${m.fist_on_table?1:0}`)
        }
      }catch{}
    })
    return ()=>{ ws.close(); wsRef.current=null }
  },[])

  useEffect(()=>{
    let stop=false
    const loop=async()=>{ try{ const h=await getHealth(); if(!stop) setVideoRunning(!!h.video_running) }catch{}; if(!stop) setTimeout(loop,1500) }
    loop(); return ()=>{ stop=true }
  },[])

  const startCam=async()=>{ await startVideo({fps:30,table_y_ratio:tableRatio}); const h=await getHealth(); setVideoRunning(!!h.video_running) }

  const calibrate=async()=>{
    setToast('Авто-поиск стола…')
    const r=await autoDetectTable()
    if(r.ok){ setToast('Стол найден ✓'); setTimeout(()=>setToast(null),1200); return }
    setToast('Обведите область стола'); setPolyModal(true)
  }
  const onPolySave=async(poly:[number,number][])=>{
    setPolyModal(false); setToast('Сохраняем контур…')
    const ok = (await setTableROI(poly)).ok
    if(ok){ setToast('Контур сохранён ✓'); setTimeout(()=>setToast(null),1200) }
    else { setToast('Не удалось — используйте линию-порог'); setLineModal(true) }
  }
  const onPolySkip=()=>{ setPolyModal(false); setLineModal(true) }
  const onLineSave=async(r:number)=>{
    setTableRatio(r); setLineModal(false); setToast('Перезапуск видео…')
    await stopVideo(); await startVideo({fps:30,table_y_ratio:r})
    const h=await getHealth(); setVideoRunning(!!h.video_running)
    setToast('Порог сохранён ✓'); setTimeout(()=>setToast(null),1200)
  }

  return (
    <div className="app-wrap">
      <div className="topbar">
        <div className="brand">Mafia AI</div>
        <div className="chips">
          <div className="chip">WS: <b style={{color:wsOk?'var(--ok)':'var(--bad)'}}>{wsOk?'connected':'offline'}</b></div>
          <div className="chip">Camera: <b style={{color:videoRunning?'var(--ok)':'var(--warn)'}}>{videoRunning?'on':'off'}</b></div>
          <div className="chip">Gesture: <b>{lastGesture}</b></div>
        </div>
        <div style={{display:'flex',gap:8,flexWrap:'wrap'}}>
          <NeonButton className="min" onClick={startCam}>Включить камеру</NeonButton>
          <NeonButton className="min" onClick={()=>setEnrollModal(true)} disabled={!videoRunning}>Авторизация</NeonButton>
          <NeonButton className="min" onClick={calibrate} disabled={!videoRunning}>Калибровка стола</NeonButton>
        </div>
      </div>

      <CameraPanel>
        <div className="hud">
          <span>FPS≈30</span>
          <span className="chip">Порог: <b>{tableRatio.toFixed(3)}</b></span>
        </div>
      </CameraPanel>

      {toast && <div style={{position:'fixed',left:16,bottom:16,zIndex:60}}><div className="card section" style={{padding:'10px 14px'}}>{toast}</div></div>}

      <Modal open={polyModal} onClose={onPolySkip}><TablePolyDrawer onSave={onPolySave} onCancel={onPolySkip}/></Modal>
      <Modal open={lineModal} onClose={()=>setLineModal(false)}><LineThreshold initial={tableRatio} onSave={onLineSave} onCancel={()=>setLineModal(false)}/></Modal>
      <Modal open={enrollModal} onClose={()=>setEnrollModal(false)}><EnrollPanel onClose={()=>setEnrollModal(false)}/></Modal>
    </div>
  )
}
