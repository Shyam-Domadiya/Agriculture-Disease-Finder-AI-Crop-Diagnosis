import { useState } from 'react';

const API_URL = 'http://localhost:8000';

// Symptom configuration
const SYMPTOMS = [
  { key: 'Wilting',       emoji: '🥀', label: 'Wilting',        help: 'The plant appears droopy or limp' },
  { key: 'WhiteSpots',    emoji: '⚪', label: 'White Spots',    help: 'White powdery patches on leaves' },
  { key: 'YellowLeaves',  emoji: '🍂', label: 'Yellow Leaves',  help: 'Leaves turning yellow or brown' },
  { key: 'StuntedGrowth', emoji: '📏', label: 'Stunted Growth', help: 'Plant is smaller than expected' },
  { key: 'BlackSpots',    emoji: '⬛', label: 'Black Spots',    help: 'Black or brown lesions on leaves' },
  { key: 'LeafCurl',      emoji: '🌀', label: 'Leaf Curl',      help: 'Leaves curling or deforming' },
  { key: 'FoulSmell',     emoji: '👃', label: 'Foul Smell',     help: 'Unpleasant odour from roots or stems' },
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

function App() {
  // State
  const [temperature, setTemperature] = useState('Normal');
  const [humidity, setHumidity]       = useState('Normal');
  const [symptoms, setSymptoms]       = useState(
    Object.fromEntries(SYMPTOMS.map(s => [s.key, 'None']))
  );
  const [results, setResults]     = useState(null);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState(null);

  const updateSymptom = (key, value) => {
    setSymptoms(prev => ({ ...prev, [key]: value }));
  };

  const handleDiagnose = async () => {
    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symptoms,
          environment: { Temperature: temperature, Humidity: humidity },
        }),
      });

      if (!response.ok) throw new Error('API request failed');
      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError('Could not connect to the backend. Make sure the API is running on port 8000.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Compute summary
  const activeSymptoms = SYMPTOMS.filter(s => symptoms[s.key] !== 'None');

  return (
    <div className="app-container">
      {/* ---- Header ---- */}
      <header className="header">
        <span className="header__icon">🌿</span>
        <h1 className="header__title">Agriculture Disease Finder</h1>
        <p className="header__subtitle">
          AI-Powered Crop Disease Diagnosis using Bayesian Networks
        </p>
        <div className="header__badges">
          <span className="badge">🦠 7 Diseases</span>
          <span className="badge">🩺 7 Symptoms</span>
          <span className="badge">🌡️ Environment AI</span>
          <span className="badge">📊 Severity Levels</span>
        </div>
      </header>

      <hr className="divider" />

      {/* ---- Environmental Conditions ---- */}
      <section className="section">
        <h2 className="section__title">🌡️ Environmental Conditions</h2>
        <p className="section__subtitle">
          Set the current weather conditions to refine the diagnosis
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
        <h2 className="section__title">🩺 Observed Symptoms</h2>
        <p className="section__subtitle">
          Rate each symptom's severity — leave as None if not observed
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

      {/* ---- Diagnose Button ---- */}
      <button
        className="diagnose-btn"
        onClick={handleDiagnose}
        disabled={loading}
      >
        {loading ? '⏳ Analysing...' : '🔍  Diagnose My Crop'}
      </button>

      <hr className="divider" />

      {/* ---- Results / Placeholder ---- */}
      {loading && (
        <div className="loading">
          <div className="spinner" />
          <span>Analysing symptoms with Bayesian inference...</span>
        </div>
      )}

      {error && (
        <div className="summary-banner" style={{ borderColor: 'rgba(244,67,54,0.3)', background: 'rgba(244,67,54,0.08)' }}>
          <strong>⚠️ Error:</strong> {error}
        </div>
      )}

      {results && !loading && (
        <div className="results">
          <h2 className="section__title">📊 Diagnosis Results</h2>

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

          <hr className="divider" />

          {/* Summary */}
          <div className="summary-banner">
            <strong>Symptoms observed: </strong>
            {activeSymptoms.length > 0
              ? activeSymptoms.map(s => `${s.emoji} ${s.label} (${symptoms[s.key]})`).join(', ')
              : 'None'
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

      {!results && !loading && !error && (
        <div className="placeholder">
          <span className="placeholder__icon">🌱</span>
          <p className="placeholder__text">
            Set environmental conditions, rate symptom severity, and click <strong>Diagnose My Crop</strong>
          </p>
          <p className="placeholder__subtext">
            The expanded AI model analyses 7 symptoms × 7 diseases with environmental context
          </p>
        </div>
      )}

      {/* ---- Footer ---- */}
      <footer className="footer">
        Agriculture Disease Finder • 7 Diseases · 7 Symptoms · Environmental AI
        • Powered by Bayesian Networks (pgmpy) • React + FastAPI
      </footer>
    </div>
  );
}

export default App;
