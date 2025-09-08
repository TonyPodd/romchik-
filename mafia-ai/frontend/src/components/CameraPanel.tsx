import React from 'react'

export default function CameraPanel({ children }:{ children?: React.ReactNode }){
  return (
    <div style={{marginTop:16}}>
      <h3>Camera (server overlay)</h3>
      <div className="neon-card" style={{overflow:'hidden', maxWidth: '100%'}}>
        <div style={{position:'relative'}}>
          <img
            src="http://127.0.0.1:8000/video/mjpeg"
            alt="camera stream"
            style={{display:'block', width:'100%', height:'auto'}}
          />
          {children}
        </div>
      </div>
      <p style={{fontSize:12, color:'#8aa0c8', marginTop:6}}>
        Если изображения нет — проверьте, что бэкенд запущен и <code>/video/status</code> даёт <b>running: true</b>.
      </p>
    </div>
  )
}
