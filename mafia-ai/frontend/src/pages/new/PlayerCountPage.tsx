import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { GlassCard } from '../../components/ui/GlassCard';
import { Button } from '../../components/ui/Button';
import { Input } from '../../components/ui/Input';

export function PlayerCountPage() {
  const navigate = useNavigate();
  const [customCount, setCustomCount] = useState('');
  const [selectedPreset, setSelectedPreset] = useState<number | null>(null);

  const presets = [
    { count: 8, roles: '5 Мирных · 2 Мафия · 1 Дон' },
    { count: 10, roles: '6 Мирных · 2 Мафия · 1 Дон · 1 Шериф' },
  ];

  const handlePresetSelect = (count: number) => {
    setSelectedPreset(count);
    setCustomCount('');
  };

  const handleCustomChange = (value: string) => {
    const num = parseInt(value);
    if (value === '' || (!isNaN(num) && num >= 1 && num <= 50)) {
      setCustomCount(value);
      setSelectedPreset(null);
    }
  };

  const handleContinue = () => {
    const count = selectedPreset || parseInt(customCount);
    if (count && count >= 1) {
      navigate('/setup/face-registration', { state: { playerCount: count } });
    }
  };

  const canContinue = selectedPreset !== null || (customCount !== '' && parseInt(customCount) >= 1);

  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '2rem',
    }}>
      <div style={{
        maxWidth: '800px',
        width: '100%',
        textAlign: 'center',
      }}>
        <h1 style={{
          fontSize: 'clamp(2rem, 5vw, 3rem)',
          fontWeight: 700,
          marginBottom: '0.5rem',
        }}>
          Выберите количество игроков
        </h1>

        <p style={{
          color: '#94a3b8',
          marginBottom: '3rem',
          fontSize: '1.125rem',
        }}>
          Спортивная мафия поддерживает 8 или 10 игроков
        </p>

        {/* Presets */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
          gap: '1.5rem',
          marginBottom: '2rem',
        }}>
          {presets.map((option) => (
            <GlassCard
              key={option.count}
              onClick={() => handlePresetSelect(option.count)}
              style={{
                padding: '2.5rem 2rem',
                cursor: 'pointer',
                position: 'relative',
                background: selectedPreset === option.count
                  ? 'rgba(79, 70, 229, 0.1)'
                  : '#1a1d2e',
                border: selectedPreset === option.count
                  ? '2px solid rgba(79, 70, 229, 0.5)'
                  : '1px solid rgba(255, 255, 255, 0.08)',
                transition: 'all 0.2s ease',
              }}
            >
              <div style={{
                fontSize: '4rem',
                fontWeight: 800,
                color: selectedPreset === option.count ? '#4f46e5' : '#e2e8f0',
                marginBottom: '1rem',
                transition: 'color 0.2s ease',
              }}>
                {option.count}
              </div>

              <div style={{
                fontSize: '1.125rem',
                fontWeight: 600,
                marginBottom: '0.75rem',
                color: selectedPreset === option.count ? '#e2e8f0' : '#94a3b8',
                transition: 'color 0.2s ease',
              }}>
                Игроков
              </div>

              <div style={{
                fontSize: '0.875rem',
                color: '#64748b',
                lineHeight: 1.6,
              }}>
                {option.roles}
              </div>

              {selectedPreset === option.count && (
                <div style={{
                  position: 'absolute',
                  top: '1rem',
                  right: '1rem',
                  width: '2rem',
                  height: '2rem',
                  background: '#4f46e5',
                  borderRadius: '50%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '1rem',
                  color: '#fff',
                }}>
                  ✓
                </div>
              )}
            </GlassCard>
          ))}
        </div>

        {/* Custom input */}
        <GlassCard style={{
          padding: '2rem',
          marginBottom: '3rem',
        }}>
          <h3 style={{
            fontSize: '1.125rem',
            fontWeight: 600,
            marginBottom: '1rem',
            color: '#e2e8f0',
          }}>
            Или введите свое количество (для тестирования)
          </h3>
          <Input
            type="number"
            placeholder="Например: 3"
            value={customCount}
            onChange={(e) => handleCustomChange(e.target.value)}
            style={{
              fontSize: '1.5rem',
              textAlign: 'center',
              fontWeight: 600,
            }}
            min={1}
            max={50}
          />
          <p style={{
            color: '#64748b',
            fontSize: '0.875rem',
            marginTop: '0.75rem',
          }}>
            От 1 до 50 игроков
          </p>
        </GlassCard>

        {/* Actions */}
        <div style={{
          display: 'flex',
          gap: '1rem',
          justifyContent: 'center',
        }}>
          <Button
            variant="secondary"
            size="lg"
            onClick={() => navigate('/')}
          >
            Назад
          </Button>
          <Button
            size="lg"
            disabled={!canContinue}
            onClick={handleContinue}
            style={{ minWidth: '200px' }}
          >
            Продолжить
          </Button>
        </div>
      </div>
    </div>
  );
}
