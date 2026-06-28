import { useState, useEffect } from 'react';

const API_URL = import.meta.env.PROD ? '' : 'http://localhost:8000';

// Symptom configuration
const SYMPTOMS = [
  { key: 'Wilting',       emoji: '🥀', label: 'Wilting',        help: 'The plant appears droopy or limp' },
  { key: 'WhiteSpots',    emoji: '⚪', label: 'White Spots',    help: 'White/grey spots or fuzzy mold patches' },
  { key: 'YellowLeaves',  emoji: '🍂', label: 'Yellow Leaves',  help: 'Leaves turning yellow, brown, or mottled' },
  { key: 'StuntedGrowth', emoji: '📏', label: 'Stunted Growth', help: 'Plant is smaller or grows slower than expected' },
  { key: 'BlackSpots',    emoji: '⬛', label: 'Black Spots',    help: 'Dark brown or black lesions on foliage' },
  { key: 'LeafCurl',      emoji: '🌀', label: 'Leaf Curl',      help: 'Leaves curling upwards, downwards, or twisted' },
  { key: 'FoulSmell',     emoji: '👃', label: 'Foul Smell',     help: 'Unpleasant odour from rotting tissue or roots' },
];

const SEVERITY_OPTIONS = ['None', 'Mild', 'Severe'];
const ENVIRON_OPTIONS  = ['Low', 'Normal', 'High'];

function SeveritySelector({ value, onChange }) {
  return (
    <div className="severity-group">
      {SEVERITY_OPTIONS.map((opt) => (
        <button
          key={opt}
          type="button"
          className={`severity-btn ${value === opt ? `severity-btn--active-${opt.toLowerCase()}` : ''}`}
          onClick={() => onChange(opt)}
        >
          {opt}
        </button>
      ))}
    </div>
  );
}

function BarChart({ predictions }) {
  const maxProb = Math.max(...predictions.map(p => p.probability), 0.01);

  return (
    <div className="chart card">
      {predictions.map((pred) => (
        <div className="chart-bar" key={pred.disease}>
          <span className="chart-bar__label">
            {pred.emoji} {pred.disease.replace(/_/g, ' ')}
          </span>
          <div className="chart-bar__track">
            <div
              className="chart-bar__fill"
              style={{
                width: `${(pred.probability / Math.max(maxProb, 0.01)) * 100}%`,
                backgroundColor: pred.color,
                opacity: 0.85,
              }}
            />
          </div>
          <span className="chart-bar__value">
            {(pred.probability * 100).toFixed(1)}%
          </span>
        </div>
      ))}
    </div>
  );
}

function ResultCard({ prediction, isTop }) {
  return (
    <div className={`result-card ${isTop ? 'result-card--top' : ''}`}>
      <div className="result-card__info">
        <div className="result-card__name">
          {prediction.emoji} {prediction.disease.replace(/_/g, ' ')}
          {isTop && <span className="result-card__badge">⭐ Most Likely</span>}
        </div>
        <div className="result-card__desc">{prediction.description}</div>
        <div className="result-card__tip">💡 {prediction.tip}</div>
      </div>
      <div className="result-card__prob">
        {(prediction.probability * 100).toFixed(1)}%
      </div>
    </div>
  );
}

function TreatmentPlanner({ crop, disease, diseaseInfo }) {
  const [activeTab, setActiveTab] = useState('organic');
  const [completedTasks, setCompletedTasks] = useState(new Set());

  if (!diseaseInfo) return null;

  const organicTasks = diseaseInfo.organic || [];
  const chemicalTasks = diseaseInfo.chemical || [];
  const preventionTasks = diseaseInfo.prevention || [];
  const allTasks = [...organicTasks, ...chemicalTasks, ...preventionTasks];

  const toggleTask = (task) => {
    setCompletedTasks(prev => {
      const next = new Set(prev);
      if (next.has(task)) {
        next.delete(task);
      } else {
        next.add(task);
      }
      return next;
    });
  };

  const totalCount = allTasks.length;
  const completedCount = allTasks.filter(t => completedTasks.has(t)).length;
  const progressPercent = totalCount > 0 ? Math.round((completedCount / totalCount) * 100) : 0;

  const getProgressMessage = (pct) => {
    if (pct === 0) return '🌱 Recovery plan initialized. Ready to treat!';
    if (pct < 40) return '⚡ Active treatment started. Keep going!';
    if (pct < 80) return '🛡️ Moderate protection achieved. Plant is recovering.';
    if (pct < 100) return '🌿 Excellent care. Almost fully protected!';
    return '🟢 Complete recovery plan applied! Outstanding job.';
  };

  const getActiveTasks = () => {
    if (activeTab === 'organic') return organicTasks;
    if (activeTab === 'chemical') return chemicalTasks;
    return preventionTasks;
  };

  const activeTasks = getActiveTasks();

  return (
    <div className="treatment-planner card" style={{ marginTop: '1.2rem' }}>
      <div className="treatment-planner__header" style={{ marginBottom: '1.25rem' }}>
        <h3 className="treatment-planner__title" style={{ fontSize: '1.15rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          🌱 Actionable Remedy Planner
        </h3>
        <p className="treatment-planner__subtitle" style={{ fontSize: '0.85rem', color: '#aaa', marginTop: '0.2rem' }}>
          Check off treatments you've completed for <strong>{disease.replace(/_/g, ' ')}</strong>
        </p>
      </div>

      {/* Progress Bar */}
      <div className="progress-container" style={{ marginBottom: '1.25rem' }}>
        <div className="progress-header" style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '0.4rem', fontWeight: 600 }}>
          <span className="progress-label" style={{ color: 'var(--text-secondary)' }}>📋 Plan Completion</span>
          <span className="progress-value" style={{ color: 'var(--accent-green-light)' }}>{progressPercent}% ({completedCount}/{totalCount})</span>
        </div>
        <div className="progress-track" style={{ height: '8px', background: 'rgba(255,255,255,0.06)', borderRadius: '4px', overflow: 'hidden', marginBottom: '0.4rem' }}>
          <div
            className="progress-fill"
            style={{
              height: '100%',
              width: `${progressPercent}%`,
              transition: 'width 0.3s ease',
              background: progressPercent === 100 
                ? 'linear-gradient(90deg, #4CAF50, #81C784)' 
                : 'linear-gradient(90deg, var(--accent-green), var(--accent-green-light))'
            }}
          />
        </div>
        <div className="progress-status" style={{ fontSize: '0.8rem', fontStyle: 'italic', color: '#bbb' }}>{getProgressMessage(progressPercent)}</div>
      </div>

      {/* Tabs */}
      <div className="planner-tabs" style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem', borderBottom: '1px solid rgba(255,255,255,0.08)', paddingBottom: '0.75rem' }}>
        <button
          type="button"
          className={`planner-tab-btn ${activeTab === 'organic' ? 'planner-tab-btn--active' : ''}`}
          onClick={() => setActiveTab('organic')}
        >
          🍃 Organic ({organicTasks.length})
        </button>
        <button
          type="button"
          className={`planner-tab-btn ${activeTab === 'chemical' ? 'planner-tab-btn--active' : ''}`}
          onClick={() => setActiveTab('chemical')}
          disabled={chemicalTasks.length === 0}
        >
          🧪 Chemical ({chemicalTasks.length})
        </button>
        <button
          type="button"
          className={`planner-tab-btn ${activeTab === 'prevention' ? 'planner-tab-btn--active' : ''}`}
          onClick={() => setActiveTab('prevention')}
        >
          🛡️ Prevention ({preventionTasks.length})
        </button>
      </div>

      {/* Tasks List */}
      <div className="tasks-list" style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
        {activeTasks.length > 0 ? (
          activeTasks.map((task, idx) => {
            const isChecked = completedTasks.has(task);
            return (
              <label
                key={idx}
                className={`task-item ${isChecked ? 'task-item--checked' : ''}`}
              >
                <input
                  type="checkbox"
                  className="task-item__checkbox"
                  checked={isChecked}
                  onChange={() => toggleTask(task)}
                />
                <span className="task-item__text">{task}</span>
              </label>
            );
          })
        ) : (
          <div className="task-item__empty">
            No specific {activeTab} remedies required.
          </div>
        )}
      </div>
    </div>
  );
}

function App() {
  // State
  const [modelInfo, setModelInfo] = useState(null);
  const [selectedCrop, setSelectedCrop] = useState('tomato');
  const [temperature, setTemperature] = useState('Normal');
  const [humidity, setHumidity]       = useState('Normal');
  const [symptoms, setSymptoms]       = useState(
    Object.fromEntries(SYMPTOMS.map(s => [s.key, 'None']))
  );
  const [results, setResults]     = useState(null);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState(null);

  // Fetch model specifications on mount
  useEffect(() => {
    const fetchModelInfo = async () => {
      try {
        const response = await fetch(`${API_URL}/model-info`);
        if (!response.ok) throw new Error('API server is not responding');
        const data = await response.json();
        setModelInfo(data);
      } catch (err) {
        setError('Could not connect to the backend. Make sure the FastAPI API is running on port 8000.');
        console.error(err);
      }
    };
    fetchModelInfo();
  }, []);

  // Debounced auto-prediction whenever inputs change
  useEffect(() => {
    if (!modelInfo) return;

    setLoading(true);
    const delayDebounceFn = setTimeout(async () => {
      try {
        const response = await fetch(`${API_URL}/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            crop: selectedCrop,
            symptoms,
            environment: { Temperature: temperature, Humidity: humidity },
          }),
        });

        if (!response.ok) throw new Error('Prediction API failed');
        const data = await response.json();
        setResults(data);
        setError(null);
      } catch (err) {
        setError('Connection lost. Please make sure the backend is active.');
        console.error(err);
      } finally {
        setLoading(false);
      }
    }, 200); // 200ms debounce window for fast response

    return () => clearTimeout(delayDebounceFn);
  }, [selectedCrop, symptoms, temperature, humidity, modelInfo]);

  const updateSymptom = (key, value) => {
    setSymptoms(prev => ({ ...prev, [key]: value }));
  };

  // Reset symptoms helper
  const handleResetSymptoms = () => {
    setSymptoms(Object.fromEntries(SYMPTOMS.map(s => [s.key, 'None'])));
  };

  // Compute active symptoms summary
  const activeSymptoms = SYMPTOMS.filter(s => symptoms[s.key] !== 'None');

  return (
    <div className="app-container">
      {/* ---- Header ---- */}
      <header className="header">
        <span className="header__icon">🌿</span>
        <h1 className="header__title">Agriculture Disease Finder</h1>
        <p className="header__subtitle">
          Real-Time Crop Disease Diagnosis using Bayesian Networks
        </p>
        <div className="header__badges">
          <span className="badge">🌱 {modelInfo ? `${modelInfo.crops.length} Crops` : 'Crops'}</span>
          <span className="badge">🦠 {modelInfo ? `${modelInfo.crop_details[selectedCrop]?.diseases.length} Diseases` : 'Diseases'}</span>
          <span className="badge">🩺 7 Symptoms</span>
          <span className="badge">🌡️ Environment AI</span>
        </div>
      </header>

      <hr className="divider" />

      {/* ---- Crop Selection ---- */}
      <section className="section">
        <h2 className="section__title">🌾 Select Your Crop</h2>
        <p className="section__subtitle">
          Choose a crop type to load its specialized disease diagnosis model
        </p>
        {modelInfo ? (
          <div className="crop-grid">
            {modelInfo.crops.map((cropKey) => {
              const crop = modelInfo.crop_details[cropKey];
              const isSelected = selectedCrop === cropKey;
              return (
                <button
                  key={cropKey}
                  type="button"
                  className={`crop-card ${isSelected ? 'crop-card--active' : ''}`}
                  onClick={() => setSelectedCrop(cropKey)}
                >
                  <span className="crop-card__emoji">{crop.emoji}</span>
                  <span className="crop-card__name">{crop.name}</span>
                </button>
              );
            })}
          </div>
        ) : (
          <div className="loading-placeholder">Loading crop models...</div>
        )}
      </section>

      <hr className="divider" />

      {/* ---- Environmental Conditions ---- */}
      <section className="section">
        <h2 className="section__title">🌡️ Environmental Conditions</h2>
        <p className="section__subtitle">
          Set current weather factors (Temperature & Humidity) to calibrate disease priors
        </p>
        <div className="card card--env">
          <div className="env-grid">
            <div className="select-group">
              <label className="select-group__label">🌡️ Temperature</label>
              <select
                className="select-group__input"
                value={temperature}
                onChange={(e) => setTemperature(e.target.value)}
              >
                {ENVIRON_OPTIONS.map(opt => (
                  <option key={opt} value={opt}>{opt}</option>
                ))}
              </select>
            </div>
            <div className="select-group">
              <label className="select-group__label">💧 Humidity</label>
              <select
                className="select-group__input"
                value={humidity}
                onChange={(e) => setHumidity(e.target.value)}
              >
                {ENVIRON_OPTIONS.map(opt => (
                  <option key={opt} value={opt}>{opt}</option>
                ))}
              </select>
            </div>
          </div>
        </div>
      </section>

      <hr className="divider" />

      {/* ---- Symptoms ---- */}
      <section className="section">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
          <h2 className="section__title" style={{ margin: 0 }}>🩺 Observed Symptoms</h2>
          {activeSymptoms.length > 0 && (
            <button type="button" className="reset-link" onClick={handleResetSymptoms}>
              🔄 Reset Symptoms
            </button>
          )}
        </div>
        <p className="section__subtitle">
          Rate each symptom's severity — values propagate to the Bayesian Network in real-time
        </p>
        <div className="symptom-grid">
          {SYMPTOMS.map((symptom) => (
            <div className="symptom-item" key={symptom.key}>
              <div className="symptom-item__label">
                {symptom.emoji} {symptom.label}
              </div>
              <div className="symptom-item__help">{symptom.help}</div>
              <SeveritySelector
                value={symptoms[symptom.key]}
                onChange={(val) => updateSymptom(symptom.key, val)}
              />
            </div>
          ))}
        </div>
      </section>

      <hr className="divider" />

      {/* ---- Live Diagnosis Status ---- */}
      <div className="live-status-bar">
        <span className="live-status-bar__pulse"></span>
        <span className="live-status-bar__text">
          {loading ? '⏳ Updating Bayesian inference...' : '⚡ Live AI Diagnostics Active'}
        </span>
      </div>

      {error && (
        <div className="summary-banner" style={{ borderColor: 'rgba(244,67,54,0.3)', background: 'rgba(244,67,54,0.08)', marginTop: '1.5rem' }}>
          <strong>⚠️ Error:</strong> {error}
        </div>
      )}

      {/* ---- Results ---- */}
      {results && (
        <div className="results" style={{ opacity: loading ? 0.75 : 1, transition: 'opacity 0.2s' }}>
          <h2 className="section__title">📊 Probabilistic Diagnosis</h2>

          {/* Bar Chart */}
          <BarChart predictions={results.predictions} />

          {/* Result Cards */}
          {results.predictions.map((pred, idx) => (
            <ResultCard
              key={pred.disease}
              prediction={pred}
              isTop={idx === 0}
            />
          ))}

          {/* Actionable Remedy Planner */}
          <TreatmentPlanner
            key={`${selectedCrop}-${results.top_disease}`}
            crop={selectedCrop}
            disease={results.top_disease}
            diseaseInfo={modelInfo?.crop_details[selectedCrop]?.disease_info[results.top_disease]}
          />

          <hr className="divider" />

          {/* Summary */}
          <div className="summary-banner">
            <strong>Crop: </strong>
            {modelInfo ? modelInfo.crop_details[selectedCrop]?.name : selectedCrop}
            <br />
            <strong>Symptoms observed: </strong>
            {activeSymptoms.length > 0
              ? activeSymptoms.map(s => `${s.emoji} ${s.label} (${symptoms[s.key]})`).join(', ')
              : 'None (Healthy Baseline)'
            }
            <br />
            <strong>Environment: </strong>
            🌡️ {temperature} temp · 💧 {humidity} humidity
            <br />
            <strong>Most likely diagnosis: </strong>
            {results.top_disease.replace(/_/g, ' ')} ({(results.top_probability * 100).toFixed(1)}%)
          </div>
        </div>
      )}

      {/* ---- Footer ---- */}
      <footer className="footer">
        Agriculture Disease Finder • Multi-Crop Bayesian AI
        • Powered by pgmpy • React + FastAPI
      </footer>
    </div>
  );
}

export default App;
