// src/components/Table.tsx
import React from 'react'
import Timer from './Timer'

export default function Table({
  onStart, ticks
}:{ onStart:(seat:number)=>void, ticks: Record<number, number> }){
  const seats = Array.from({length:10}, (_,i)=>i+1)
  return (
    <div style={{display:'grid', gridTemplateColumns:'repeat(5, 1fr)', gap:12}}>
      {seats.map(seat=>(
        <div key={seat} style={{border:'1px solid #ddd', borderRadius:12, padding:12}}>
          <div style={{display:'flex', justifyContent:'space-between', alignItems:'center'}}>
            <strong>#{seat}</strong>
            <button onClick={()=> onStart(seat)} style={{padding:'4px 8px'}}>Start 60s</button>
          </div>
          <Timer seat={seat} ms={ticks[seat] ?? 0} />
        </div>
      ))}
    </div>
  )
}
