import React from 'react'
export default function Modal({open,onClose,children}:{open:boolean,onClose:()=>void,children:React.ReactNode}){
  if(!open) return null
  return (
    <div className="modal-back" onClick={onClose}>
      <div className="card modal" onClick={e=>e.stopPropagation()}>
        {children}
      </div>
    </div>
  )
}
