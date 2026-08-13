/**
 * Compliance Page
 * Violation tracking, FDCPA transcript checking, and audit-logged resolution
 */
import { useState, useEffect } from 'react';
import {
    Shield,
    AlertTriangle,
    CheckCircle,
    XCircle,
    FileText,
    Send,
    Filter,
    X,
    Check,
    Info
} from 'lucide-react';
import { getViolations, checkTranscript, getComplianceStats, resolveViolation } from '../services/api';
import './Compliance.css';

export default function Compliance() {
    const [violations, setViolations] = useState([]);
    const [stats, setStats] = useState(null);
    const [loading, setLoading] = useState(true);
    const [transcript, setTranscript] = useState('');
    const [checkResult, setCheckResult] = useState(null);
    const [checking, setChecking] = useState(false);
    const [filterSeverity, setFilterSeverity] = useState('all');
    const [filterStatus, setFilterStatus] = useState('all');
    const [selectedViolation, setSelectedViolation] = useState(null);
    const [resolutionNotes, setResolutionNotes] = useState('');
    const [resolving, setResolving] = useState(false);

    useEffect(() => {
        loadData();
    }, []);

    const loadData = async () => {
        try {
            const [violationsRes, statsRes] = await Promise.all([
                getViolations({ limit: 30 }),
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

    const handleCheckTranscript = async (sampleText) => {
        const textToCheck = sampleText || transcript;
        if (!textToCheck.trim()) return;

        if (sampleText) {
            setTranscript(sampleText);
        }

        setChecking(true);
        try {
            const response = await checkTranscript(textToCheck);
            setCheckResult(response.data);
            // Refresh stats & list as new violation was logged to DB
            loadData();
        } catch (error) {
            console.error('Transcript check failed:', error);
        } finally {
            setChecking(false);
        }
    };

    const handleOpenResolveModal = (violation) => {
        setSelectedViolation(violation);
        setResolutionNotes('Agent completed mandatory FDCPA retraining and call script review.');
    };

    const handleConfirmResolve = async () => {
        if (!selectedViolation || !resolutionNotes.trim()) return;

        setResolving(true);
        try {
            await resolveViolation(selectedViolation.id, resolutionNotes.trim());
            setSelectedViolation(null);
            setResolutionNotes('');
            await loadData();
        } catch (error) {
            console.error('Failed to resolve violation:', error);
        } finally {
            setResolving(false);
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

    const filteredViolations = violations.filter(v => {
        if (filterSeverity !== 'all' && v.severity !== filterSeverity) return false;
        if (filterStatus === 'unresolved' && v.is_resolved) return false;
        if (filterStatus === 'resolved' && !v.is_resolved) return false;
        return true;
    });

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
                    <p className="page-subtitle">FDCPA compliance inspection, violation logging & audit-trail resolution</p>
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
                        <span className="stat-label">Unresolved Queue</span>
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
                {/* Transcript Inspector */}
                <div className="card transcript-checker">
                    <div className="card-header">
                        <h3 className="card-title">
                            <FileText size={18} />
                            FDCPA Transcript Inspector
                        </h3>
                    </div>

                    <div className="sample-prompts">
                        <span className="sample-label">Test Sample:</span>
                        <button
                            className="btn btn-secondary btn-xs"
                            onClick={() => handleCheckTranscript("Hello, this is RecoverMax. This is an attempt to collect a debt. Please call us back regarding invoice INV-2024001.")}
                        >
                            <CheckCircle size={12} style={{ color: 'var(--success)', marginRight: 4 }} /> Compliant Call
                        </button>
                        <button
                            className="btn btn-secondary btn-xs"
                            onClick={() => handleCheckTranscript("Pay us immediately or we will sue you, garnish your salary, and send you to prison!")}
                        >
                            <AlertTriangle size={12} style={{ color: 'var(--danger)', marginRight: 4 }} /> Violation Threat
                        </button>
                    </div>

                    <textarea
                        className="transcript-input"
                        placeholder="Paste call transcript or agent email text here..."
                        value={transcript}
                        onChange={(e) => setTranscript(e.target.value)}
                        rows={5}
                    ></textarea>

                    <button
                        className="btn btn-primary check-btn"
                        onClick={() => handleCheckTranscript()}
                        disabled={checking || !transcript.trim()}
                    >
                        <Send size={16} />
                        {checking ? 'Analyzing Compliance...' : 'Run Compliance Inspection'}
                    </button>

                    {checkResult && (
                        <div className={`check-result ${checkResult.compliant ? 'compliant' : 'non-compliant'}`}>
                            <div className="result-header">
                                {checkResult.compliant ? (
                                    <>
                                        <CheckCircle size={20} />
                                        <span>Compliant (No FDCPA Violations Detected)</span>
                                    </>
                                ) : (
                                    <>
                                        <XCircle size={20} />
                                        <span>FDCPA Violation Detected ({checkResult.severity?.toUpperCase()} SEVERITY)</span>
                                    </>
                                )}
                            </div>

                            {checkResult.violations && checkResult.violations.length > 0 && (
                                <ul className="violation-list">
                                    {checkResult.violations.map((v, i) => (
                                        <li key={i}>
                                            <strong>{v.type}:</strong> {v.keyword ? `Forbidden keyword "${v.keyword}" used.` : v.disclosure ? `Missing required disclosure: ${v.disclosure}` : v.excerpt}
                                        </li>
                                    ))}
                                </ul>
                            )}

                            {checkResult.recommendations && checkResult.recommendations.length > 0 && (
                                <div className="recommendations">
                                    <strong>Coaching Recommendations:</strong>
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

                {/* Recorded Violations */}
                <div className="card">
                    <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <h3 className="card-title">Recorded Violations</h3>
                        <div className="filter-group" style={{ display: 'flex', gap: '8px' }}>
                            <select
                                className="form-select-sm"
                                value={filterStatus}
                                onChange={(e) => setFilterStatus(e.target.value)}
                            >
                                <option value="all">All Status</option>
                                <option value="unresolved">Unresolved Only</option>
                                <option value="resolved">Resolved Only</option>
                            </select>
                            <select
                                className="form-select-sm"
                                value={filterSeverity}
                                onChange={(e) => setFilterSeverity(e.target.value)}
                            >
                                <option value="all">All Severities</option>
                                <option value="critical">Critical</option>
                                <option value="high">High</option>
                                <option value="medium">Medium</option>
                            </select>
                        </div>
                    </div>

                    <div className="violations-list">
                        {filteredViolations.length === 0 ? (
                            <div className="empty-state">
                                <CheckCircle size={48} />
                                <p>No violations matching filter</p>
                            </div>
                        ) : (
                            filteredViolations.map((violation) => (
                                <div key={violation.id} className={`violation-item ${violation.is_resolved ? 'item-resolved' : ''}`}>
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

                                        {violation.is_resolved && violation.resolution_notes && (
                                            <div className="resolution-audit-box">
                                                <Info size={12} />
                                                <span><strong>Resolution Action:</strong> {violation.resolution_notes}</span>
                                            </div>
                                        )}

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
                                            <button
                                                className="btn btn-primary btn-sm resolve-btn"
                                                onClick={() => handleOpenResolveModal(violation)}
                                            >
                                                Resolve Violation
                                            </button>
                                        )}
                                    </div>
                                </div>
                            ))
                        )}
                    </div>
                </div>
            </div>

            {/* Resolution Modal */}
            {selectedViolation && (
                <div className="modal-overlay">
                    <div className="modal-content resolution-modal">
                        <div className="modal-header">
                            <h3 className="modal-title">
                                <Shield size={20} /> Resolve Violation #{selectedViolation.id}
                            </h3>
                            <button className="close-btn" onClick={() => setSelectedViolation(null)}>
                                <X size={18} />
                            </button>
                        </div>
                        
                        <div className="modal-body">
                            <div className="violation-summary">
                                <div className="summary-row">
                                    <span className="summary-label">Violation Type:</span>
                                    <strong className="summary-val">{selectedViolation.violation_type.replace('_', ' ')}</strong>
                                </div>
                                <div className="summary-row">
                                    <span className="summary-label">Severity:</span>
                                    <span className={`severity-badge ${selectedViolation.severity}`}>{selectedViolation.severity}</span>
                                </div>
                                <div className="summary-row">
                                    <span className="summary-label">Target Case / Agency:</span>
                                    <span className="summary-val">Case #{selectedViolation.case_id} (Agency #{selectedViolation.agency_id})</span>
                                </div>
                                <div className="summary-desc">
                                    <strong>Description:</strong> {selectedViolation.description}
                                </div>
                            </div>

                            <div className="form-group">
                                <label className="form-label">Quick Action Presets:</label>
                                <div className="preset-buttons">
                                    <button
                                        type="button"
                                        className="preset-pill"
                                        onClick={() => setResolutionNotes('Agent completed mandatory FDCPA retraining and call script review.')}
                                    >
                                        🎓 Agent FDCPA Retraining Completed
                                    </button>
                                    <button
                                        type="button"
                                        className="preset-pill"
                                        onClick={() => setResolutionNotes('Call scripts updated with mandatory Mini-Miranda disclosures.')}
                                    >
                                        📜 Call Script Updated
                                    </button>
                                    <button
                                        type="button"
                                        className="preset-pill"
                                        onClick={() => setResolutionNotes('Reviewed call audio recording with compliance team — verified false positive flag.')}
                                    >
                                        🔍 Audio Verified - False Positive
                                    </button>
                                </div>
                            </div>

                            <div className="form-group">
                                <label className="form-label">Resolution Justification & Audit Notes *</label>
                                <textarea
                                    className="form-textarea"
                                    rows={3}
                                    value={resolutionNotes}
                                    onChange={(e) => setResolutionNotes(e.target.value)}
                                    placeholder="Enter details of corrective action taken..."
                                ></textarea>
                            </div>
                        </div>

                        <div className="modal-footer">
                            <button
                                type="button"
                                className="btn btn-secondary"
                                onClick={() => setSelectedViolation(null)}
                                disabled={resolving}
                            >
                                Cancel
                            </button>
                            <button
                                type="button"
                                className="btn btn-primary"
                                onClick={handleConfirmResolve}
                                disabled={resolving || !resolutionNotes.trim()}
                            >
                                <Check size={16} />
                                {resolving ? 'Logging Resolution...' : 'Confirm & Log Resolution'}
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
