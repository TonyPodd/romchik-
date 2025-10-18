# Mafia AI - Architecture Document

**Version**: 2.0
**Date**: 2025-10-19
**Status**: Design Phase

---

## 1. Overview

AI-powered host for competitive Mafia game. System detects players via camera, tracks speech, enforces rules, detects fouls, and manages game flow.

### Key Requirements

- **Real-time detection**: faces, gestures, speech (30 FPS minimum)
- **High accuracy**: priority over performance, but must be fast enough
- **Scalability**: no hard limit on number of players (typically 8-10)
- **Modularity**: easy to swap detection algorithms
- **Future-proof**: easy migration from JSON to SQL database

---

## 2. System Architecture

### 2.1 Architecture Style

**Clean Architecture** with **Event-Driven Design**

```
┌─────────────────────────────────────────────┐
│            Presentation Layer               │
│  (React Frontend + WebSocket Client)        │
└──────────────────┬──────────────────────────┘
                   │ WebSocket/HTTP
┌──────────────────▼──────────────────────────┐
│              API Layer                      │
│  (FastAPI + WebSocket + REST endpoints)     │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│         Application Layer                   │
│  (Use Cases, DTOs, Event Bus)              │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│           Domain Layer                      │
│  (Entities, Value Objects, Services)        │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│       Infrastructure Layer                  │
│  (Detection, Audio, Storage, Video)         │
└─────────────────────────────────────────────┘
```

### 2.2 Core Principles

1. **Dependency Inversion**: Domain depends on abstractions, not implementations
2. **Single Responsibility**: Each module has one job
3. **Open/Closed**: Open for extension, closed for modification
4. **Interface Segregation**: Small, focused interfaces
5. **Dependency Injection**: Loose coupling via DI container

---

## 3. Technology Stack

### 3.1 Backend

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **Framework** | FastAPI 0.115+ | High performance, native async, WebSocket support |
| **State Machine** | python-statemachine | Clean FSM implementation |
| **Face Detection** | YOLOv8-face | Fast, accurate, no face count limit |
| **Face Recognition** | InsightFace / ArcFace ONNX | 512D embeddings, high accuracy |
| **Hand Detection** | YOLOv8-hand | Real-time, robust |
| **Hand Landmarks** | MediaPipe Hands | Precise finger tracking |
| **VAD** | Silero VAD | CPU-only, fast, accurate |
| **ASR** | Faster-Whisper | Optimized Whisper, 3x faster |
| **Speaker ID** | Audio-Video Correlation | Match speech to visible face |
| **Storage** | JSON + Pydantic | Easy to migrate to SQLAlchemy later |
| **Async** | asyncio + uvloop | High concurrency |

### 3.2 Frontend

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **Framework** | React 18 + TypeScript | Modern, type-safe |
| **Build Tool** | Vite | Fast HMR, optimized builds |
| **State** | Zustand | Lightweight, simple API |
| **Styling** | TailwindCSS | Utility-first, customizable |
| **Animations** | Framer Motion | Smooth, declarative animations |
| **UI Kit** | Headless UI + Custom | Accessible, flexible |
| **Icons** | Lucide React | Clean, consistent |
| **API Client** | Native fetch + WebSocket | Simple, no heavy deps |

---

## 4. Domain Model

### 4.1 Core Entities

**Player**
```python
class Player:
    id: PlayerId
    name: str
    face_embedding: np.ndarray  # 512D vector
    voice_profile: Optional[VoiceProfile]
    seat: int
    role: Optional[Role]  # Mafia, Civilian, Sheriff, etc.
    is_alive: bool
    fouls: List[Foul]
```

**Game**
```python
class Game:
    id: GameId
    players: List[Player]
    phase: GamePhase
    current_speaker: Optional[PlayerId]
    day_number: int
    start_time: datetime
    events: List[GameEvent]
```

**Turn**
```python
class Turn:
    player_id: PlayerId
    phase: GamePhase
    start_time: datetime
    duration_seconds: int
    speech_text: Optional[str]
    fouls_detected: List[Foul]
```

### 4.2 Value Objects

**GamePhase** (Enum)
- IDLE
- SETUP
- ROLE_ASSIGNMENT
- NIGHT_0 (mafia introduction)
- DAY_DISCUSSION
- NOMINATIONS
- VOTING
- LAST_WORD
- NIGHT
- GAME_END

**Role** (Enum)
- CIVILIAN
- MAFIA
- SHERIFF
- DON

**FoulType** (Enum)
- SPEAKING_OUT_OF_TURN
- GESTURING_DURING_VOTE
- TIMEOUT_EXCEEDED

---

## 5. Application Layer

### 5.1 Use Cases

1. **Enroll Player**
   - Input: video frames
   - Output: Player with face embedding
   - Side effects: Save to storage

2. **Start Game**
   - Input: list of players
   - Output: Game initialized
   - Side effects: Assign roles, start state machine

3. **Process Day Phase**
   - Input: game state, audio/video streams
   - Output: turn results, fouls
   - Side effects: Update game state, broadcast events

4. **Detect Foul**
   - Input: player action, current phase
   - Output: Foul or None
   - Side effects: Record foul, notify frontend

5. **Process Voting**
   - Input: votes from players
   - Output: eliminated player
   - Side effects: Update game state

### 5.2 Event Bus

Central event bus for loose coupling:

```python
class EventBus:
    async def publish(event: DomainEvent) -> None
    async def subscribe(event_type: Type[Event], handler: Handler) -> None
```

Events:
- `PlayerEnrolled`
- `GameStarted`
- `PhaseChanged`
- `TurnStarted`
- `TurnEnded`
- `FoulDetected`
- `PlayerEliminated`
- `GameEnded`

---

## 6. Infrastructure Layer

### 6.1 Vision Pipeline

**Architecture:**
```
Camera
  ↓
Frame Buffer (async queue)
  ↓
Parallel Processing:
  ├─ Face Detection → Recognition → Tracking
  ├─ Hand Detection → Landmarks → Gesture Classification
  └─ Table ROI Detection
  ↓
Event Aggregation
  ↓
Event Bus
```

**Face Detection & Recognition:**
- YOLOv8-face for bounding boxes
- ArcFace for 512D embeddings
- Cosine similarity matching (threshold: 0.4)
- Face tracking with Kalman filter (reduce jitter)

**Gesture Detection:**
- YOLOv8 for hand bounding boxes
- MediaPipe for 21 hand landmarks
- Custom classifier for "finger on table"
- Future: more complex gesture patterns

### 6.2 Audio Pipeline

**Architecture:**
```
Microphone
  ↓
Audio Buffer (16kHz mono)
  ↓
Silero VAD (detect speech segments)
  ↓
Faster-Whisper ASR (transcribe)
  ↓
Audio-Video Correlator (match to player)
  ↓
Event Bus
```

**Speaker Identification:**
- Correlate speech timing with visible faces
- Find player whose mouth moves during speech
- Fallback: use face detection during speech start

### 6.3 Storage Layer

**Repository Pattern:**
```python
class IPlayerRepository(ABC):
    async def add(player: Player) -> PlayerId
    async def get(id: PlayerId) -> Optional[Player]
    async def list() -> List[Player]
    async def update(player: Player) -> bool
    async def delete(id: PlayerId) -> bool
```

**Implementations:**
- `JsonPlayerRepository` (current)
- `SqlPlayerRepository` (future)

---

## 7. Game State Machine

```
┌─────────┐
│  IDLE   │
└────┬────┘
     │ start_setup()
┌────▼────┐
│  SETUP  │ (enroll players, calibrate table)
└────┬────┘
     │ assign_roles()
┌────▼─────────────┐
│ ROLE_ASSIGNMENT  │
└────┬─────────────┘
     │ start_night_0()
┌────▼────────┐
│  NIGHT_0    │ (mafia intro)
└────┬────────┘
     │
┌────▼──────────────────┐
│    GAME LOOP          │
│  ┌────────────────┐   │
│  │ DAY_DISCUSSION │   │ (60s per player)
│  └────┬───────────┘   │
│  ┌────▼────────┐      │
│  │ NOMINATIONS │      │
│  └────┬────────┘      │
│  ┌────▼────────┐      │
│  │   VOTING    │      │ (NO gestures!)
│  └────┬────────┘      │
│  ┌────▼─────────┐     │
│  │  LAST_WORD   │     │ (60s)
│  └────┬─────────┘     │
│  ┌────▼──────┐        │
│  │   NIGHT   │        │ (mafia kills)
│  └────┬──────┘        │
│       │               │
│   [check win]         │
│       │               │
└───────┼───────────────┘
        │
   ┌────▼─────────┐
   │  GAME_END    │
   └──────────────┘
```

**Transitions:**
- Automatic: timer-based (e.g., DAY → NOMINATIONS after all speak)
- Manual: moderator override (future feature)
- Conditional: win condition check

---

## 8. Frontend Architecture

### 8.1 Feature-Based Structure

```
src/
├── features/
│   ├── setup/          # Setup wizard
│   ├── game/           # Active game UI
│   └── analytics/      # Post-game analysis
├── shared/
│   ├── components/ui/  # Design system
│   ├── hooks/          # Reusable hooks
│   └── api/            # API client
└── store/              # Global state (Zustand)
```

### 8.2 State Management

**Zustand Stores:**

1. **gameStore**: Current game state
   - phase, players, current speaker, timer

2. **playersStore**: Enrolled players
   - list, add, update, delete

3. **uiStore**: UI state
   - modals, notifications, theme

### 8.3 WebSocket Integration

```typescript
useWebSocket({
  onMessage: (event) => {
    switch (event.type) {
      case 'phase_changed':
        gameStore.setPhase(event.phase)
        break
      case 'foul_detected':
        uiStore.showFoul(event.foul)
        break
      case 'timer_tick':
        gameStore.updateTimer(event.time_left)
        break
    }
  }
})
```

---

## 9. Design System

### 9.1 Visual Style

**Theme**: Futuristic Cyberpunk with Glassmorphism

**Color Palette** (Dark Mode):
```css
--bg-primary: #0a0e1a
--bg-secondary: #111827
--bg-glass: rgba(17, 24, 39, 0.6)
--accent-primary: #3b82f6    /* blue */
--accent-secondary: #8b5cf6  /* purple */
--accent-success: #10b981    /* green */
--accent-danger: #ef4444     /* red */
--accent-warning: #f59e0b    /* amber */
--text-primary: #f9fafb
--text-secondary: #9ca3af
--border-glass: rgba(255, 255, 255, 0.1)
```

**Glass Effect:**
```css
.glass {
  background: rgba(17, 24, 39, 0.6);
  backdrop-filter: blur(16px);
  border: 1px solid rgba(255, 255, 255, 0.1);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}
```

**Animations:**
- Smooth page transitions (Framer Motion)
- Hover effects on buttons (scale, glow)
- Phase transition animations
- Timer countdown with pulse
- Foul alerts with shake + flash

### 9.2 Key Components

1. **GlassPanel**: Glassmorphism container
2. **NeonButton**: Futuristic button with glow
3. **PlayerSeat**: Player position with avatar + status
4. **PhaseIndicator**: Current game phase with icon
5. **SpeechTimer**: Circular progress timer
6. **FoulAlert**: Animated foul notification

---

## 10. Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Vision FPS | ≥30 FPS | On modern CPU/GPU |
| Face Detection Latency | <50ms | Per frame |
| Gesture Detection Latency | <50ms | Per frame |
| ASR Latency | <1s | Real-time transcription |
| WebSocket Latency | <100ms | Event propagation |
| UI Frame Rate | 60 FPS | Smooth animations |
| Face Recognition Accuracy | >95% | Same-person match |
| Gesture Accuracy | >90% | Finger on table |

---

## 11. Future Enhancements

### Phase 2 (After MVP)
- [ ] Voice profiles for better speaker ID
- [ ] Advanced gesture detection (dynamic gestures)
- [ ] Game replay system
- [ ] Statistics and analytics
- [ ] Multi-language support

### Phase 3 (Long-term)
- [ ] Cloud deployment
- [ ] Multi-camera support
- [ ] Real-time video annotations
- [ ] Mobile app for players
- [ ] Tournament mode

---

## 12. Implementation Plan

### Phase 1: Core Architecture (Week 1-2)
1. Restructure backend (Clean Architecture)
2. Implement domain models
3. Setup event bus
4. Create repository abstractions

### Phase 2: Detection Pipeline (Week 2-3)
1. Integrate YOLOv8-face
2. Integrate ArcFace recognition
3. Implement gesture detection
4. Optimize pipeline performance

### Phase 3: Audio Pipeline (Week 3-4)
1. Integrate Silero VAD
2. Integrate Faster-Whisper
3. Implement speaker identification
4. Test end-to-end audio flow

### Phase 4: Game Engine (Week 4-5)
1. Implement state machine
2. Implement use cases
3. Implement foul detection
4. Test game flow

### Phase 5: Frontend (Week 5-6)
1. Design system components
2. Setup wizard
3. Game UI
4. WebSocket integration

### Phase 6: Integration & Polish (Week 6-7)
1. End-to-end testing
2. Performance optimization
3. Bug fixes
4. Documentation

---

## 13. Dependencies

### Backend Requirements
```
fastapi==0.115.0
uvicorn[standard]==0.30.6
pydantic==2.9.2
python-statemachine==2.3.0
opencv-python==4.10.0
numpy==1.26.4
onnxruntime==1.18.0
insightface==0.7.3
ultralytics==8.3.0
mediapipe==0.10.14
silero-vad==5.1
faster-whisper==1.0.3
pyannote.audio==3.1.1  # optional
```

### Frontend Dependencies
```json
{
  "react": "^18.3.1",
  "react-dom": "^18.3.1",
  "typescript": "^5.6.2",
  "zustand": "^4.5.0",
  "framer-motion": "^12.0.0",
  "tailwindcss": "^3.4.0",
  "@headlessui/react": "^2.0.0",
  "lucide-react": "^0.400.0",
  "clsx": "^2.1.1"
}
```

---

## 14. Testing Strategy

1. **Unit Tests**: Core domain logic, use cases
2. **Integration Tests**: API endpoints, WebSocket
3. **E2E Tests**: Full game flow simulation
4. **Performance Tests**: Detection pipeline benchmarks
5. **Manual Testing**: Real game scenarios

---

## Appendix A: API Specification

### REST Endpoints

```
GET  /health
GET  /api/players
POST /api/players/enroll
POST /api/players/reset
GET  /api/game/status
POST /api/game/start
POST /api/game/stop
POST /api/video/start
POST /api/video/stop
GET  /api/video/mjpeg
POST /api/calibration/table/set
POST /api/calibration/table/auto
```

### WebSocket Events

**Client → Server:**
```json
{"type": "game.start", "players": [...]}
{"type": "game.action", "action": "vote", "target": 1}
```

**Server → Client:**
```json
{"type": "phase.changed", "phase": "DAY_DISCUSSION"}
{"type": "turn.started", "player_id": 1, "duration": 60}
{"type": "foul.detected", "player_id": 2, "type": "SPEAKING_OUT_OF_TURN"}
{"type": "timer.tick", "time_left": 45}
```

---

**End of Document**
