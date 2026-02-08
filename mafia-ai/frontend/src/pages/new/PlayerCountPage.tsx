import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { GlassCard } from '../../components/ui/GlassCard';
import { Button } from '../../components/ui/Button';
import { Input } from '../../components/ui/Input';

const presets = [
  { count: 8, roles: '5 мирных · 2 мафия · 1 дон' },
  { count: 10, roles: '6 мирных · 2 мафия · 1 дон · 1 шериф' },
];

export function PlayerCountPage() {
  const navigate = useNavigate();
  const [customCount, setCustomCount] = useState('');
  const [selectedPreset, setSelectedPreset] = useState<number | null>(null);

  function handleCustomChange(value: string) {
    const parsed = parseInt(value, 10);
    if (value === '' || (!Number.isNaN(parsed) && parsed >= 1 && parsed <= 50)) {
      setCustomCount(value);
      setSelectedPreset(null);
    }
  }

  function handleContinue() {
    const count = selectedPreset || parseInt(customCount, 10);
    if (count && count >= 1) {
      navigate('/setup/face-registration', { state: { playerCount: count } });
    }
  }

  const canContinue = selectedPreset !== null || (customCount !== '' && parseInt(customCount, 10) >= 1);

  return (
    <div className="setup-shell">
      <div className="setup-container">
        <div className="setup-hero">
          <h1 className="setup-title">Количество игроков</h1>
          <p className="setup-subtitle">
            Выберите стандартный формат или задайте кастомное число для тестового прогона.
          </p>
        </div>

        <div className="preset-grid">
          {presets.map((preset) => (
            <button
              type="button"
              key={preset.count}
              className={`preset-card ${selectedPreset === preset.count ? 'preset-card--active' : ''}`.trim()}
              onClick={() => {
                setSelectedPreset(preset.count);
                setCustomCount('');
              }}
            >
              <div className="preset-card__count">{preset.count}</div>
              <div className="feature-card__title">Игроков</div>
              <div className="preset-card__hint">{preset.roles}</div>
            </button>
          ))}
        </div>

        <GlassCard className="stack panel-pad">
          <div>
            <h3 className="feature-card__title">Кастомное значение</h3>
            <p className="feature-card__text">
              Значение используется для отладки интерфейса и пайплайна регистрации.
            </p>
          </div>
          <Input
            type="number"
            placeholder="1-50"
            value={customCount}
            onChange={(event) => handleCustomChange(event.target.value)}
            min={1}
            max={50}
          />
        </GlassCard>

        <div className="setup-actions">
          <Button variant="secondary" size="lg" onClick={() => navigate('/')}>
            Назад
          </Button>
          <Button size="lg" disabled={!canContinue} onClick={handleContinue}>
            Продолжить
          </Button>
        </div>
      </div>
    </div>
  );
}
