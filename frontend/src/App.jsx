import React, { useState, useEffect } from 'react';
import './style.css';

export default function App() {
  const [movies, setMovies] = useState([]);
  const [selectedId, setSelectedId] = useState(1);
  const [graphData, setGraphData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(false);
  const [threshold, setThreshold] = useState(0.15);

  useEffect(() => {
    fetch('http://localhost:8000/movies', { credentials: 'include' })
      .then(res => res.json())
      .then(data => setMovies(data))
      .catch(err => console.error("Error loading movies:", err));
  }, []);

  useEffect(() => {
    setLoading(true);
    setError(false);
    fetch(`http://localhost:8000/graph/${selectedId}?threshold=${threshold}`, { credentials: 'include' })
      .then(res => {
        if (!res.ok) throw new Error("Isolated movie");
        return res.json();
      })
      .then(data => {
        setGraphData(data);
        setLoading(false);
      })
      .catch(() => {
        setError(true);
        setGraphData(null);
        setLoading(false);
      });
  }, [selectedId, threshold]);

  return (
    <div className="container">
      <header>
        <h1>🎬 Movie Similarity Graph</h1>
        <p>Explore relationships through genre overlap and audience behavior.</p>
      </header>

      <div className="controls">
        <label>Select Movie: </label>
        <select value={selectedId} onChange={e => setSelectedId(Number(e.target.value))}>
          {movies.map(m => (
            <option key={m.movieId} value={m.movieId}>{m.title}</option>
          ))}
        </select>

        <label>Filter Weak Connections ({threshold}): </label>
        <input 
          type="range" min="0.0" max="0.5" step="0.05" 
          value={threshold} onChange={e => setThreshold(Number(e.target.value))} 
        />
      </div>

      {loading && <div className="state-msg">Loading relationship graph...</div>}

      {error && (
        <div className="empty-state">
          <h3>No Connections Found</h3>
          <p>This movie is unknown or isolated based on the current threshold filter.</p>
        </div>
      )}

      {graphData && !loading && (
        <div className="graph-view">
          <h2>Selected: {graphData.movie.title}</h2>
          <span className="badge">Genres: {graphData.movie.genres}</span>

          <div className="connection-grid">
            {graphData.connections.map(conn => (
              <div key={conn.movieId} className="card" onClick={() => setSelectedId(conn.movieId)}>
                <h3>{conn.title}</h3>
                <p className="genres">{conn.genres}</p>
                <div className="relations">
                  {conn.relationships.map((rel, idx) => (
                    <div key={idx} className="rel-tag">
                      <strong>{rel.type}</strong> ({Math.round(rel.strength * 100)}%)
                      <small>{rel.explanation}</small>
                    </div>
                  ))}
                </div>
                <button className="navigate-btn">Explore Graph From Here →</button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}