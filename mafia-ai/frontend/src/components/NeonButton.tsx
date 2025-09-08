import React from 'react'
export default function NeonButton({children,onClick,disabled,className}:{children:React.ReactNode,onClick?:()=>void,disabled?:boolean,className?:string}){
  return <button className={`btn ${className||''}`} onClick={onClick} disabled={disabled} style={{opacity:disabled?.5:1,pointerEvents:disabled?'none':'auto'}}>{children}</button>
}
