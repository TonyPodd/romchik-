type StageKey = 'faces' | 'voice' | 'table';

interface SetupStageHeaderProps {
  current: StageKey;
  title: string;
  subtitle: string;
}

const stages: Array<{ key: StageKey; label: string; short: string }> = [
  { key: 'faces', label: 'Регистрация лиц', short: 'Лица' },
  { key: 'voice', label: 'Регистрация голосов', short: 'Голос' },
  { key: 'table', label: 'Определение стола', short: 'Стол' },
];

export function SetupStageHeader({ current, title, subtitle }: SetupStageHeaderProps) {
  const currentIndex = stages.findIndex((stage) => stage.key === current);
  const currentStage = stages[currentIndex] || stages[0];

  return (
    <header className="stage-head">
      <div className="stage-head__meta">
        <span>Этап {currentIndex + 1} из {stages.length}</span>
        <strong>{currentStage.label}</strong>
      </div>

      <div className="stage-head__track">
        {stages.map((stage, index) => {
          const isActive = index === currentIndex;
          const isPassed = index < currentIndex;
          return (
            <div
              key={stage.key}
              className={`stage-pill ${isActive ? 'is-active' : ''} ${isPassed ? 'is-passed' : ''}`.trim()}
              title={stage.label}
              aria-current={isActive ? 'step' : undefined}
            >
              <span>{stage.short}</span>
            </div>
          );
        })}
      </div>

      <div className="stage-head__titles">
        <h1>{title}</h1>
        <p>{subtitle}</p>
      </div>
    </header>
  );
}
