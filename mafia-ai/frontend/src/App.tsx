import React, { useEffect, useRef, useState } from 'react'
import Table from './components/Table'
import Timer from './components/Timer'
import CameraPanel from './components/CameraPanel'
import StartScreen from './components/StartScreen'
import CalibratePanel from './components/CalibratePanel'
import NeonButton from './components/NeonButton'
import TableRoiDrawer from './components/TableRoiDrawer'
import { getHealth, startVideo, stopVideo } from './api'
// import { autoDetectTable, clearTableROI, setTableROI } from './api'
import TablePolyDrawer from './components/TablePolyDrawer'
import { autoDetectTable, clearTableROI, setTableROI, getTableStatus } from './api'



type TickMap = Record<number, number>
type Phase = 'intro' | 'calibrate' | 'game'

export default function App(){
  const wsRef = useRef<WebSocket | null>(null)
  const [ticks, setTicks] = useState<TickMap>({})
  const [log, setLog] = useState<string[]>([])
  const [wsOk, setWsOk] = useState(false)
  const [videoRunning, setVideoRunning] = useState(false)
  const [phase, setPhase] = useState<Phase>('intro')
  const [lastGesture, setLastGesture] = useState<string>('—')
  const [tableRatio, setTableRatio] = useState<number>(0.80)

  // WS init
  useEffect(() => {
    if (wsRef.current) return
    const ws = new WebSocket('ws://127.0.0.1:8000/ws')
    wsRef.current = ws

    const onOpen = () => { setWsOk(true); setLog(l => ['WS connected', ...l].slice(0, 14)) }
    const onClose = () => { setWsOk(false); setLog(l => ['WS disconnected', ...l].slice(0, 14)) }
    const onMessage = (e: MessageEvent) => {
      try {
        const m = JSON.parse(e.data as string)
        if (m.type === 'timer.tick') {
          setTicks(prev => ({ ...prev, [m.seat]: m.msLeft }))
        } else if (m.type === 'timer.end') {
          setTicks(prev => ({ ...prev, [m.seat]: 0 }))
        } else if (m.type === 'gesture') {
          const s = `gesture: digit=${m.digit ?? '·'} fist=${m.fist_on_table?1:0} pistol=${m.pistol?1:0}`
          setLastGesture(s)
          setLog(l => [s, ...l].slice(0, 14))
        }
      } catch {}
    }

    ws.addEventListener('open', onOpen)
    ws.addEventListener('close', onClose)
    ws.addEventListener('message', onMessage)
    return () => {
      ws.removeEventListener('open', onOpen)
      ws.removeEventListener('close', onClose)
      ws.removeEventListener('message', onMessage)
      ws.close()
      wsRef.current = null
    }
  }, [])

  // poll /health для отметки videoRunning
  useEffect(()=>{
    let stop = false
    const tick = async () => {
      try {
        const h = await getHealth()
        if (!stop) setVideoRunning(!!h.video_running)
      } catch {}
      if (!stop) setTimeout(tick, 1500)
    }
    tick()
    return ()=> { stop = true }
  }, [])

  const startTimer = (seat:number, ms:number=60000) =>
    wsRef.current?.send(JSON.stringify({ type:'timer.start', seat, ms }))

  // StartScreen actions
  const onStartCamera = async () => {
    await startVideo({ table_y_ratio: tableRatio })
    const h = await getHealth()
    setVideoRunning(!!h.video_running)
  }
  const onCalibrate = () => setPhase('calibrate')
  const onStartGame = () => setPhase('game')

  // Calibration save
  const saveCalibration = async (ratio:number) => {
    setTableRatio(ratio)
    await stopVideo()
    await startVideo({ table_y_ratio: ratio })
    const h = await getHealth()
    setVideoRunning(!!h.video_running)
    setPhase('intro') // вернёмся на интро (или сразу 'game' — решай сам)
  }

  return (
    <div style={{fontFamily:'Inter, system-ui, sans-serif', padding:16, maxWidth: 1200, margin: '0 auto'}}>
      {phase === 'intro' && (
        <>
          <StartScreen
            wsOk={wsOk}
            videoRunning={videoRunning}
            onStartCamera={onStartCamera}
            onCalibrate={onCalibrate}
            onStartGame={onStartGame}
          />
          {videoRunning && (
            <div className="neon-card" style={{padding:16}}>
              <div className="flex" style={{gap:12, alignItems:'center', marginBottom:8}}>
                <span className="chip">Last gesture</span>
                <code>{lastGesture}</code>
                <span className="chip">Table ratio</span>
                <code>{tableRatio.toFixed(3)}</code>
                <NeonButton onClick={()=> setPhase('calibrate')}>Калибровка</NeonButton>
              </div>
              <CameraPanel />
            </div>
          )}
        </>
      )}

     {phase === 'calibrate' && (
        <>
          <div className="flex" style={{gap:12, alignItems:'center', marginBottom:8}}>
            <NeonButton onClick={()=> setPhase('intro')}>← Назад</NeonButton>
            <span className="chip">Camera: <b style={{color: videoRunning ? '#56ff9c' : '#ffdd66'}}>{videoRunning ? 'on' : 'off'}</b></span>
          </div>

          {/* Авто-детект */}
          <div className="neon-card" style={{padding:16}}>
            <h2>Авто-детект стола</h2>
            <p className="neon-sub">Ищем самый крупный прямоугольный контур в кадре.</p>
            <div className="flex" style={{gap:12, marginTop:8}}>
              <NeonButton onClick={async ()=>{
                const r = await autoDetectTable()
                if (!r.ok) { alert('Не удалось найти прямоугольник'); return }
                setLog(l => [`table: autodetect OK`, ...l].slice(0,14))
              }} disabled={!videoRunning}>Авто-детект</NeonButton>
              <NeonButton onClick={async ()=>{
                await clearTableROI()
                setLog(l => [`table: cleared`, ...l].slice(0,14))
              }} disabled={!videoRunning}>Сбросить ROI</NeonButton>
            </div>
            <CameraPanel />
          </div>

          {/* Ручная разметка */}
          <div style={{marginTop:16}}>
            <TablePolyDrawer
              onSave={async (poly)=>{
                const r = await setTableROI(poly)
                if (!r.ok) { alert('Ошибка сохранения ROI'); return }
                setLog(l => [`table: polygon ROI saved (${poly.length} pts)`, ...l].slice(0,14))
              }}
              onCancel={()=> setPhase('intro')}
              initial={undefined /* можно подставить сохранённый poly_norm */}
            />
          </div>


          {/* Линия-порог (оставим пока как резерв) */}
          <CalibratePanel
            initialRatio={tableRatio}
            onSave={async (ratio)=>{
              setTableRatio(ratio)
              await stopVideo()
              await startVideo({ table_y_ratio: ratio })
              const h = await getHealth()
              setVideoRunning(!!h.video_running)
              setLog(l => [`line ratio set ${ratio.toFixed(3)}`, ...l].slice(0,14))
            }}
            onCancel={()=> setPhase('intro')}
          />
        </>
      )}


      {phase === 'game' && (
        <>
          <div className="flex" style={{gap:12, alignItems:'center', marginBottom:8}}>
            <NeonButton onClick={()=> setPhase('intro')}>← В меню</NeonButton>
            <span className="chip">WS: <b style={{color: wsOk ? '#56ff9c' : '#ff6a6a'}}>{wsOk ? 'connected' : 'offline'}</b></span>
            <span className="chip">Camera: <b style={{color: videoRunning ? '#56ff9c' : '#ffdd66'}}>{videoRunning ? 'on' : 'off'}</b></span>
            <span className="chip">Table ratio</span><code>{tableRatio.toFixed(3)}</code>
            <span className="chip">Last gesture</span><code>{lastGesture}</code>
          </div>

          <div className="neon-card" style={{padding:16}}>
            <CameraPanel />
          </div>

          <div className="neon-card" style={{padding:16, marginTop:16}}>
            <h2>Стол</h2>
            <Table onStart={(seat)=> startTimer(seat)} ticks={ticks} />
          </div>

          <div className="neon-card" style={{padding:16, marginTop:16}}>
            <h3>Лог</h3>
            <ul>{log.map((s,i)=><li key={i}>{s}</li>)}</ul>
          </div>
        </>
      )}
    </div>
  )
}
