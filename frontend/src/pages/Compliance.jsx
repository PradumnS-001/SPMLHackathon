/**
 * Compliance Page
 * Violation tracking and transcript checking
 */
import { useState, useEffect } from 'react';
import {
    Shield,
    AlertTriangle,
    CheckCircle,
    XCircle,
    FileText,
    Send
} from 'lucide-react';
import { getViolations, checkTranscript, getComplianceStats } from '../services/api';
import './Compliance.css';

export default function Compliance() {
    const [violations, setViolations] = useState([]);
    const [stats, setStats] = useState(null);
    const [loading, setLoading] = useState(true);
    const [transcript, setTranscript] = useState('');
    const [checkResult, setCheckResult] = useState(null);
    const [checking, setChecking] = useState(false);

    useEffect(() => {
        loadData();
    }, []);

    const loadData = async () => {
        try {
            const [violationsRes, statsRes] = await Promise.all([
                getViolations({ limit: 10 }),
                getComplianceStats()
            ]);
            setViolations(violationsRes.data);
            setStats(statsRes.data);
        } catch (error) {
            console.error('Failed to load compliance data:', error);
        } finally {
            setLoading(false);
        }
    };

    const handleCheckTranscript = async () => {
        if (!transcript.trim()) return;

        setChecking(true);
        try {
            const response = await checkTranscript(transcript);
            setCheckResult(response.data);
        } catch (error) {
            console.error('Transcript check failed:', error);
        } finally {
            setChecking(false);
        }
    };

    const getSeverityIcon = (severity) => {
        switch (severity) {
            case 'critical': return <XCircle className="severity-icon critical" />;
            case 'high': return <AlertTriangle className="severity-icon high" />;
            case 'medium': return <AlertTriangle className="severity-icon medium" />;
            default: return <AlertTriangle className="severity-icon low" />;
        }
    };

    if (loading) {
        return (
            <div className="loading-container">
                <div className="spinner"></div>
                <span>Loading compliance data...</span>
            </div>
        );
    }

    return (
        <div className="compliance-page">
            <div className="page-header">
                <div>
                    <h1 className="page-title">Compliance Monitor</h1>
                    <p className="page-subtitle">Track violations and check transcripts</p>
                </div>
            </div>

            {/* Stats Overview */}
            <div className="compliance-stats">
                <div className="stat-card">
                    <Shield size={24} />
                    <div>
                        <span className="stat-value">{stats?.total_violations || 0}</span>
                        <span className="stat-label">Total Violations</span>
                    </div>
                </div>
                <div className="stat-card warning">
                    <AlertTriangle size={24} />
                    <div>
                        <span className="stat-value">{stats?.unresolved_violations || 0}</span>
                        <span className="stat-label">Unresolved</span>
                    </div>
                </div>
                {stats?.by_severity?.map((s) => (
                    <div key={s.severity} className={`stat-card ${s.severity}`}>
                        <div>
                            <span className="stat-value">{s.count}</span>
                            <span className="stat-label">{s.severity}</span>
                        </div>
                    </div>
                ))}
            </div>

            <div className="compliance-grid">
                {/* Transcript Checker */}
                <div className="card transcript-checker">
                    <div className="card-header">
                        <h3 className="card-title">
                            <FileText size={18} />
                            Transcript Checker
                        </h3>
                    </div>

                    <textarea
                        className="transcript-input"
                        placeholder="Paste call transcript or email content here to check for compliance violations..."
                        value={transcript}
                        onChange={(e) => setTranscript(e.target.value)}
                        rows={6}
                    ></textarea>

                    <button
                        className="btn btn-primary check-btn"
                        onClick={handleCheckTranscript}
                        disabled={checking || !transcript.trim()}
                    >
                        <Send size={16} />
                        {checking ? 'Checking...' : 'Check Compliance'}
                    </button>

                    {checkResult && (
                        <div className={`check-result ${checkResult.compliant ? 'compliant' : 'non-compliant'}`}>
                            <div className="result-header">
                                {checkResult.compliant ? (
                                    <>
                                        <CheckCircle size={20} />
                                        <span>Compliant</span>
                                    </>
                                ) : (
                                    <>
                                        <XCircle size={20} />
                                        <span>Violations Found ({checkResult.violation_count})</span>
                                    </>
                                )}
                            </div>

                            {checkResult.violations.length > 0 && (
                                <ul className="violation-list">
                                    {checkResult.violations.map((v, i) => (
                                        <li key={i}>
                                            <strong>{v.type}:</strong> {v.keyword || v.disclosure || v.contact_time}
                                        </li>
                                    ))}
                                </ul>
                            )}

                            {checkResult.recommendations.length > 0 && (
                                <div className="recommendations">
                                    <strong>Recommendations:</strong>
                                    <ul>
                                        {checkResult.recommendations.map((r, i) => (
                                            <li key={i}>{r}</li>
                                        ))}
                                    </ul>
                                </div>
                            )}
                        </div>
                    )}
                </div>

                {/* Recent Violations */}
                <div className="card">
                    <div className="card-header">
                        <h3 className="card-title">Recent Violations</h3>
                    </div>

                    <div className="violations-list">
                        {violations.length === 0 ? (
                            <div className="empty-state">
                                <CheckCircle size={48} />
                                <p>No violations recorded</p>
                            </div>
                        ) : (
                            violations.map((violation) => (
                                <div key={violation.id} className="violation-item">
                                    <div className="violation-icon">
                                        {getSeverityIcon(violation.severity)}
                                    </div>
                                    <div className="violation-content">
                                        <div className="violation-header">
                                            <span className="violation-type">{violation.violation_type.replace('_', ' ')}</span>
                                            <span className={`severity-badge ${violation.severity}`}>
                                                {violation.severity}
                                            </span>
                                        </div>
                                        <p className="violation-desc">{violation.description}</p>
                                        <div className="violation-meta">
                                            <span>Case #{violation.case_id}</span>
                                            <span>Agency #{violation.agency_id}</span>
                                            <span>{new Date(violation.detected_at).toLocaleDateString()}</span>
                                        </div>
                                    </div>
                                    <div className="violation-status">
                                        {violation.is_resolved ? (
                                            <span className="resolved">
                                                <CheckCircle size={16} /> Resolved
                                            </span>
                                        ) : (
                                            <button className="btn btn-secondary btn-sm">Resolve</button>
                                        )}
                                    </div>
                                </div>
                            ))
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
