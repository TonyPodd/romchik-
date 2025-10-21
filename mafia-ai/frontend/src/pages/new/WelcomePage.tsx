import { useNavigate } from 'react-router-dom';
import { GlassCard } from '../../components/ui/GlassCard';
import { Button } from '../../components/ui/Button';

export function WelcomePage() {
  const navigate = useNavigate();

  const features = [
    { title: 'AI Ведущий', desc: 'Автоматическое управление игрой и контроль фолов' },
    { title: 'Распознавание лиц', desc: 'Идентификация игроков в реальном времени' },
    { title: 'Голосовой контроль', desc: 'Определение говорящего игрока' },
    { title: 'Контроль времени', desc: 'Точный тайминг для каждого игрока' },
  ];

  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '2rem',
    }}>
      <div style={{ maxWidth: '1000px', width: '100%' }}>
        <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
          <h1 style={{
            fontSize: 'clamp(2.5rem, 6vw, 4rem)',
            fontWeight: 700,
            marginBottom: '1rem',
            color: '#fff',
          }}>
            MAFIA AI
          </h1>
          <p style={{
            fontSize: '1.125rem',
            color: '#94a3b8',
            maxWidth: '600px',
            margin: '0 auto',
          }}>
            Система автоматического ведения игр в спортивную мафию
          </p>
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
          gap: '1.5rem',
          marginBottom: '3rem',
        }}>
          {features.map((f, i) => (
            <GlassCard key={i} style={{ padding: '2rem' }}>
              <h3 style={{ fontSize: '1.125rem', fontWeight: 600, marginBottom: '0.5rem' }}>
                {f.title}
              </h3>
              <p style={{ fontSize: '0.9375rem', color: '#94a3b8', lineHeight: 1.6 }}>
                {f.desc}
              </p>
            </GlassCard>
          ))}
        </div>

        <div style={{ textAlign: 'center' }}>
          <Button size="lg" onClick={() => navigate('/setup/players')} style={{ minWidth: '200px' }}>
            Начать игру
          </Button>
        </div>
      </div>
    </div>
  );
}
