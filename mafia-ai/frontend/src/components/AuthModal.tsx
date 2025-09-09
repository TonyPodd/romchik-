// import React, { useEffect, useState } from 'react'
// import Modal from './Modal'
// import NeonButton from './NeonButton'
// import { enrollCapture, listPlayers, resetPlayers } from '../api'

// export default function AuthModal({open,onClose}:{open:boolean,onClose:()=>void}){
//   const [players,setPlayers]=useState<{pid:number,label:string}[]>([])
//   const [label,setLabel]=useState('')
//   const [busy,setBusy]=useState(false)
//   const refresh = async()=>{ const r=await listPlayers(); setPlayers(r.players) }
//   useEffect(()=>{ if(open) refresh() },[open])

//   const nextLabel = `P${players.length+1}`
//   useEffect(()=>{ setLabel(nextLabel) },[players.length])

//   const capture = async()=>{
//     setBusy(true)
//     const r = await enrollCapture(label||nextLabel, 1200) // ~1 сек усреднения
//     setBusy(false)
//     if(!r.ok){ alert(r.error||'Нет лица в кадре'); return }
//     await refresh()
//   }

//   const reset = async()=>{
//     if(!confirm('Сбросить список игроков?')) return
//     await resetPlayers(); await refresh()
//   }

//   return (
//     <Modal open={open} onClose={onClose}>
//       <div className="section">
//         <h2>Авторизация игроков</h2>
//         <p className="chip" style={{marginTop:6}}>Ставьте игрока к камере и жмите «Добавить игрока». Метки появятся над головами.</p>
//         <div className="card section" style={{marginTop:12}}>
//           <div style={{display:'flex', gap:12, alignItems:'center', flexWrap:'wrap'}}>
//             <input
//               placeholder="Метка (необязательно)"
//               value={label} onChange={e=>setLabel(e.target.value)}
//               style={{padding:'10px 12px', borderRadius:10, border:'1px solid rgba(120,180,255,.25)', background:'transparent', color:'var(--fg)'}}
//             />
//             <NeonButton onClick={capture} disabled={busy}>Добавить игрока</NeonButton>
//             <NeonButton className="ghost" onClick={reset}>Сбросить</NeonButton>
//             <div className="chip">Всего: <b>{players.length}</b></div>
//           </div>
//         </div>
//         <div className="card section" style={{marginTop:12}}>
//           <h3>Список</h3>
//           <div style={{display:'flex', gap:8, flexWrap:'wrap', marginTop:8}}>
//             {players.map(p=>(
//               <div key={p.pid} className="chip">P{p.pid} — {p.label}</div>
//             ))}
//             {players.length===0 && <span className="neon-sub">ещё никого</span>}
//           </div>
//         </div>
//         <div style={{display:'flex', gap:10, marginTop:12, justifyContent:'flex-end'}}>
//           <NeonButton className="ghost" onClick={onClose}>Готово</NeonButton>
//         </div>
//       </div>
//     </Modal>
//   )
// }
