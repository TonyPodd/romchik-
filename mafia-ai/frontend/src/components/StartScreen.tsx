import React from 'react'
import NeonButton from './NeonButton'

export default function StartScreen(
  {
    wsOk, videoRunning, onStartCamera, onCalibrate, onStartGame
  }: {
    wsOk: boolean,
    videoRunning: boolean,
    onStartCamera: ()=>void,
    onCalibrate: ()=>void,
    onStartGame: ()=>void
  }
){
  return (
    <div className="center" style={{minHeight:'68vh'}}>
      <div className="neon-card" style={{padding:24, width: 'min(900px, 92vw)'}}>
        <div style={{display:'flex', justifyContent:'space-between', gap:12, alignItems:'center'}}>
          <div>
            <div className="neon-title">Mafia AI</div>
            <div className="neon-sub">Футуристичный ведущий. Включите камеру и скорректируйте линию стола перед игрой.</div>
          </div>
          <div className="flex" style={{gap:8}}>
            <div className="chip">WS: <b style={{color: wsOk ? '#56ff9c' : '#ff6a6a'}}>{wsOk ? 'connected' : 'offline'}</b></div>
            <div className="chip">Camera: <b style={{color: videoRunning ? '#56ff9c' : '#ffdd66'}}>{videoRunning ? 'on' : 'off'}</b></div>
          </div>
        </div>

        <div className="flex" style={{gap:12, marginTop:18, flexWrap:'wrap'}}>
          <NeonButton onClick={onStartCamera}>Включить камеру</NeonButton>
          <NeonButton onClick={onCalibrate} disabled={!videoRunning}>Калибровка стола</NeonButton>
          <NeonButton onClick={onStartGame} disabled={!videoRunning}>Старт игры</NeonButton>
        </div>

        <div style={{marginTop:18}}>
          <small className="neon-sub">Подсказка: линия «стола» определяет зону, где кулак считается «на столе» во время голосования.</small>
        </div>
      </div>
    </div>
  )
}
