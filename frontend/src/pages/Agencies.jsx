/**
 * Agencies Page
 * Agency management and performance tracking
 */
import { useState, useEffect } from 'react';
import { Building2, Plus, TrendingUp, Users, Shield } from 'lucide-react';
import { getAgencies } from '../services/api';
import './Agencies.css';

export default function Agencies() {
    const [agencies, setAgencies] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        loadAgencies();
    }, []);

    const loadAgencies = async () => {
        try {
            const response = await getAgencies();
            setAgencies(response.data);
        } catch (error) {
            console.error('Failed to load agencies:', error);
        } finally {
            setLoading(false);
        }
    };

    if (loading) {
        return (
            <div className="loading-container">
                <div className="spinner"></div>
                <span>Loading agencies...</span>
            </div>
        );
    }

    return (
        <div className="agencies-page">
            <div className="page-header">
                <div>
                    <h1 className="page-title">Agencies</h1>
                    <p className="page-subtitle">Manage debt collection agencies</p>
                </div>
                <button className="btn btn-primary">
                    <Plus size={16} />
                    Add Agency
                </button>
            </div>

            <div className="agencies-grid">
                {agencies.map((agency) => (
                    <div key={agency.id} className="agency-card">
                        <div className="agency-header">
                            <div className="agency-icon">
                                <Building2 size={24} />
                            </div>
                            <div>
                                <h3 className="agency-name">{agency.name}</h3>
                                <span className={`category-badge ${agency.category}`}>
                                    {agency.category}
                                </span>
                            </div>
                        </div>

                        <div className="agency-stats">
                            <div className="stat">
                                <Users size={16} />
                                <span className="stat-label">Load</span>
                                <span className="stat-value">
                                    {agency.current_load} / {agency.max_capacity}
                                </span>
                                <div className="load-bar">
                                    <div
                                        className="load-fill"
                                        style={{
                                            width: `${(agency.current_load / agency.max_capacity) * 100}%`,
                                            background: agency.current_load / agency.max_capacity > 0.8 ? 'var(--danger)' : 'var(--primary)'
                                        }}
                                    ></div>
                                </div>
                            </div>

                            <div className="stat">
                                <TrendingUp size={16} />
                                <span className="stat-label">Performance</span>
                                <span className="stat-value">{(agency.performance_score * 100).toFixed(0)}%</span>
                            </div>

                            <div className="stat">
                                <Shield size={16} />
                                <span className="stat-label">Compliance</span>
                                <span className="stat-value">{(agency.compliance_score * 100).toFixed(0)}%</span>
                            </div>
                        </div>

                        <div className="agency-metrics">
                            <div className="metric">
                                <span className="metric-label">Fit Score</span>
                                <span className="metric-value fit-score">
                                    {((agency.fit_score || 0) * 100).toFixed(0)}%
                                </span>
                            </div>
                            <div className="metric">
                                <span className="metric-label">Recovery Rate</span>
                                <span className="metric-value">{(agency.recovery_rate || 0).toFixed(1)}%</span>
                            </div>
                            <div className="metric">
                                <span className="metric-label">Cases</span>
                                <span className="metric-value">{agency.total_cases || 0}</span>
                            </div>
                        </div>

                        <div className="agency-footer">
                            <button className="btn btn-secondary">View Cases</button>
                            <button className="btn btn-secondary">Details</button>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}
