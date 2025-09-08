import React from 'react'

export default function NeonButton(
  { children, onClick, disabled }: { children: React.ReactNode, onClick?: ()=>void, disabled?: boolean }
){
  return (
    <button className="neon-btn" onClick={onClick} disabled={disabled} style={{
      opacity: disabled ? .5 : 1,
      pointerEvents: disabled ? 'none' : 'auto'
    }}>
      {children}
    </button>
  )
}
