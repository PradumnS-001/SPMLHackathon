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

// ============== API Functions ==============

// Cases
export const getCases = (params) =>
    api.get('/cases', { params });

export const getCase = (id) =>
    api.get(`/cases/${id}`);

export const createCase = (data) =>
    api.post('/cases', data);

export const updateCase = (id, data) =>
    api.patch(`/cases/${id}`, data);

export const assignCases = (caseIds) =>
    api.post('/cases/assign', { case_ids: caseIds });

export const resolveCase = (id, amount, notes) =>
    api.post(`/cases/${id}/resolve`, null, { params: { amount_recovered: amount, notes } });

// Agencies
export const getAgencies = (params) =>
    api.get('/agencies', { params });

export const getAgency = (id) =>
    api.get(`/agencies/${id}`);

export const getAgencyCases = (id, params) =>
    api.get(`/agencies/${id}/cases`, { params });

export const createAgency = (data) =>
    api.post('/agencies', data);

// Analytics
export const getDashboardStats = () =>
    api.get('/analytics/dashboard');

export const getAgencyPerformance = (limit) =>
    api.get('/analytics/agency-performance', { params: { limit } });

export const getCasesByStatus = () =>
    api.get('/analytics/cases-by-status');

export const getRecoveryTrend = (days) =>
    api.get('/analytics/recovery-trend', { params: { days } });

export const getSegmentBreakdown = () =>
    api.get('/analytics/segment-breakdown');

// Compliance
export const getViolations = (params) =>
    api.get('/compliance/violations', { params });

export const checkTranscript = (transcript, caseId, agencyId) =>
    api.post('/compliance/check-transcript', {
        transcript,
        case_id: caseId,
        agency_id: agencyId,
    });

export const resolveViolation = (id, notes) =>
    api.post(`/compliance/violations/${id}/resolve`, null, { params: { notes } });

export const getComplianceStats = () =>
    api.get('/compliance/stats');

// Auth
export const login = (email, password) =>
    api.post('/auth/login', { email, password });

export const register = (data) =>
    api.post('/auth/register', data);

export const getCurrentUser = () =>
    api.get('/auth/me');

export default api;
