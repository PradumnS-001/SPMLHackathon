/**
 * API Service
 * Centralized API client with error handling
 */
import axios from 'axios';

const API_BASE_URL = '/api/v1';

// Create axios instance
const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

// Request interceptor for auth token
api.interceptors.request.use((config) => {
    const token = localStorage.getItem('token');
    if (token) {
        config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
});

// Response interceptor for error handling
api.interceptors.response.use(
    (response) => response,
    (error) => {
        if (error.response?.status === 401) {
            localStorage.removeItem('token');
            window.location.href = '/login';
        }
        return Promise.reject(error);
    }
);

// ============== Types ==============

export interface Case {
    id: number;
    invoice_id: string;
    customer_id: string;
    customer_name: string;
    customer_email: string;
    customer_phone: string;
    debt_amount: number;
    original_amount: number;
    days_overdue: number;
    segment: string;
    p2p_score: number;
    priority_score: number;
    status: string;
    agency_id: number | null;
    assigned_at: string | null;
    resolved_at: string | null;
    amount_recovered: number;
    has_dispute: boolean;
    is_escalated: boolean;
    created_at: string;
}

export interface Agency {
    id: number;
    name: string;
    category: string;
    performance_score: number;
    compliance_score: number;
    current_load: number;
    max_capacity: number;
    contact_email: string;
    contact_phone: string;
    created_at: string;
    fit_score?: number;
    total_cases?: number;
    resolved_cases?: number;
    recovery_rate?: number;
}

export interface DashboardStats {
    total_cases: number;
    unassigned_cases: number;
    assigned_cases: number;
    resolved_cases: number;
    total_debt: number;
    total_recovered: number;
    recovery_rate: number;
    avg_days_overdue: number;
    sla_compliance: number;
}

export interface AgencyPerformance {
    agency_id: number;
    agency_name: string;
    total_cases: number;
    resolved_cases: number;
    recovery_rate: number;
    avg_resolution_days: number;
    compliance_score: number;
    performance_score: number;
}

export interface Violation {
    id: number;
    case_id: number;
    agency_id: number;
    violation_type: string;
    severity: string;
    description: string;
    transcript_excerpt: string;
    detected_at: string;
    detection_method: string;
    is_resolved: boolean;
    resolved_at: string | null;
}

export interface AssignmentResult {
    case_id: number;
    invoice_id: string;
    agency_id: number;
    agency_name: string;
    fit_score: number;
    method: string;
}

// ============== API Functions ==============

// Cases
export const getCases = (params?: Record<string, any>) =>
    api.get<Case[]>('/cases', { params });

export const getCase = (id: number) =>
    api.get<Case>(`/cases/${id}`);

export const createCase = (data: Partial<Case>) =>
    api.post<Case>('/cases', data);

export const updateCase = (id: number, data: Partial<Case>) =>
    api.patch<Case>(`/cases/${id}`, data);

export const assignCases = (caseIds?: number[]) =>
    api.post<{ total_assigned: number; assignments: AssignmentResult[] }>('/cases/assign', { case_ids: caseIds });

export const resolveCase = (id: number, amount: number, notes?: string) =>
    api.post<Case>(`/cases/${id}/resolve`, null, { params: { amount_recovered: amount, notes } });

// Agencies
export const getAgencies = (params?: Record<string, any>) =>
    api.get<Agency[]>('/agencies', { params });

export const getAgency = (id: number) =>
    api.get<Agency>(`/agencies/${id}`);

export const getAgencyCases = (id: number, params?: Record<string, any>) =>
    api.get<Case[]>(`/agencies/${id}/cases`, { params });

export const createAgency = (data: Partial<Agency>) =>
    api.post<Agency>('/agencies', data);

// Analytics
export const getDashboardStats = () =>
    api.get<DashboardStats>('/analytics/dashboard');

export const getAgencyPerformance = (limit?: number) =>
    api.get<AgencyPerformance[]>('/analytics/agency-performance', { params: { limit } });

export const getCasesByStatus = () =>
    api.get<{ status: string; count: number; percentage: number }[]>('/analytics/cases-by-status');

export const getRecoveryTrend = (days?: number) =>
    api.get<{ date: string; recovered_amount: number; case_count: number }[]>('/analytics/recovery-trend', { params: { days } });

export const getSegmentBreakdown = () =>
    api.get<{ segment: string; case_count: number; total_debt: number; recovery_rate: number }[]>('/analytics/segment-breakdown');

// Compliance
export const getViolations = (params?: Record<string, any>) =>
    api.get<Violation[]>('/compliance/violations', { params });

export const checkTranscript = (transcript: string, caseId?: number, agencyId?: number) =>
    api.post('/compliance/check-transcript', {
        transcript,
        case_id: caseId,
        agency_id: agencyId,
    });

export const resolveViolation = (id: number, notes?: string) =>
    api.post(`/compliance/violations/${id}/resolve`, null, { params: { notes } });

export const getComplianceStats = () =>
    api.get('/compliance/stats');

// Auth
export const login = (email: string, password: string) =>
    api.post<{ access_token: string }>('/auth/login', { email, password });

export const register = (data: { email: string; password: string; full_name?: string }) =>
    api.post('/auth/register', data);

export const getCurrentUser = () =>
    api.get('/auth/me');

export default api;
