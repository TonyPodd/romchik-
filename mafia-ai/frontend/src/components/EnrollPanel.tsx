import React, { useEffect, useState } from 'react'
import NeonButton from './NeonButton'
import { enrollPlayer, listPlayers, resetPlayers, setPlayerName, deletePlayer } from '../api'

type Player = { id:number, name:string, thumb:string }

export default function EnrollPanel({onClose}:{onClose:()=>void}){
  const [players,setPlayers]=useState<Player[]>([])
  const [busy,setBusy]=useState(false)
  const [name,setName]=useState('')

  const refresh = async()=> {
    const r = await listPlayers(); setPlayers(r.players||[])
  }
  useEffect(()=>{ refresh() },[])

  const add = async()=>{
    setBusy(true)
    const r = await enrollPlayer(name.trim()||undefined)
    setBusy(false)
    if(!r.ok){ alert(r.error||'Не удалось зафиксировать лицо. Встаньте фронтально, ровный свет.'); return }
    setName(''); refresh()
  }

  const rename = async(id:number, n:string)=>{
    await setPlayerName(id, n); refresh()
  }
  const del = async(id:number)=>{
    await deletePlayer(id); refresh()
  }
  const reset = async()=>{
    if(!confirm('Сбросить всех игроков?')) return
    await resetPlayers(); refresh()
  }

  return (
    <div className="section">
      <h2>Авторизация игроков</h2>
      <p className="chip" style={{marginTop:6}}>Подойдите по одному в кадр. Нажмите «Добавить игрока», чтобы зафиксировать лицо. Порядок = номера.</p>

      <div className="card" style={{padding:12, marginTop:12}}>
        <div style={{display:'flex', gap:8, flexWrap:'wrap', alignItems:'center'}}>
          <input value={name} onChange={e=>setName(e.target.value)} placeholder="Имя (необязательно)" style={{padding:'10px 12px', borderRadius:10, border:'1px solid rgba(120,180,255,.35)', background:'transparent', color:'var(--fg)'}}/>
          <NeonButton onClick={add} disabled={busy}>Добавить игрока</NeonButton>
          <NeonButton className="ghost" onClick={reset}>Сбросить</NeonButton>
          <div className="chip">Всего: <b>{players.length}</b></div>
        </div>
      </div>

      <div className="card" style={{padding:12, marginTop:12}}>
        <div style={{display:'grid', gridTemplateColumns:'repeat(auto-fill, minmax(200px,1fr))', gap:12}}>
          {players.map(p=>(
            <div key={p.id} className="card" style={{padding:10}}>
              <div style={{display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:8}}>
                <strong>#{p.id}</strong>
                <NeonButton className="min ghost" onClick={()=>del(p.id)}>Удалить</NeonButton>
              </div>
              <div style={{borderRadius:10, overflow:'hidden', border:'1px solid rgba(120,180,255,.2)'}}>
                <img src={`http://127.0.0.1:8000/static/${p.thumb}`} alt={`player ${p.id}`} style={{display:'block', width:'100%', height:160, objectFit:'cover'}}/>
              </div>
              <div style={{marginTop:8, display:'flex', gap:8}}>
                <input defaultValue={p.name||''} onBlur={e=>rename(p.id, e.target.value)} placeholder="Имя" style={{flex:1, padding:'8px 10px', borderRadius:10, border:'1px solid rgba(120,180,255,.35)', background:'transparent', color:'var(--fg)'}}/>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div style={{display:'flex', gap:10, marginTop:12}}>
        <NeonButton onClick={onClose}>Готово</NeonButton>
      </div>
    </div>
  )
}
