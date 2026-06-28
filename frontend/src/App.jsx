import { useState, useEffect } from 'react';

const API_URL = 'http://localhost:8000';

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
