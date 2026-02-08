import { useNavigate } from 'react-router-dom';
import { GlassCard } from '../../components/ui/GlassCard';
import { Button } from '../../components/ui/Button';

const features = [
  { title: 'AI Ведущий', desc: 'Контроль фаз, правил и игровых событий без модератора.' },
  { title: 'Face Detection', desc: 'Быстрая регистрация игроков с хранением профилей лиц.' },
  { title: 'Voice Flow', desc: 'Поток регистрации голоса и подготовка к speaker ID.' },
  { title: 'Realtime Pipeline', desc: 'Обновления состояния игры через API и stream-контур.' },
];

export function WelcomePage() {
  const navigate = useNavigate();

  return (
    <div className="setup-shell">
      <div className="setup-container">
        <div className="setup-wizard">
          <div className="setup-hero">
            <h1 className="setup-title">MAFIA AI</h1>
            <p className="setup-subtitle">
              Конструктор игровой сессии: лица, голос, стол и запуск партии в едином setup-процессе.
            </p>
          </div>

          <div className="feature-grid">
            {features.map((feature) => (
              <GlassCard key={feature.title} className="feature-card">
                <h3 className="feature-card__title">{feature.title}</h3>
                <p className="feature-card__text">{feature.desc}</p>
              </GlassCard>
            ))}
          </div>

          <div className="setup-actions setup-actions--center">
            <Button size="lg" onClick={() => navigate('/setup/players')}>
              Начать настройку
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
