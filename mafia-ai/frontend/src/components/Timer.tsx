import React from 'react'

export default function Timer({ seat, ms }:{ seat:number, ms:number }){
  const s = Math.floor(ms/1000)
  const d = Math.floor((ms%1000)/100)
  return <div style={{fontFamily:'monospace', fontSize:24}}>{s}.{d}s</div>
}
