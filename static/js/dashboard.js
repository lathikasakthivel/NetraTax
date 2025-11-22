/**
 * Tax Fraud Detection Dashboard - JavaScript
 * Handles Plotly chart rendering and API interactions
 */

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Fetch JSON data from an API endpoint
 */
async function fetchAPI(endpoint) {
    try {
        const response = await fetch(endpoint);
        if (!response.ok) {
            throw new Error(`API Error: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error(`Error fetching ${endpoint}:`, error);
        return null;
    }
}

/**
 * Render a Plotly chart in a container
 */
function renderPlotlyChart(containerId, plotlyData) {
    if (!plotlyData || !plotlyData.data) {
        console.warn(`No data for chart: ${containerId}`);
        return;
    }
    
    const container = document.getElementById(containerId);
    if (!container) {
        console.warn(`Container not found: ${containerId}`);
        return;
    }
    
    try {
        Plotly.newPlot(containerId, plotlyData.data, plotlyData.layout, {
            responsive: true,
            displayModeBar: true
        });
    } catch (error) {
        console.error(`Error rendering chart ${containerId}:`, error);
    }
}

// ============================================================================
// Chart Loading Functions
// ============================================================================

function loadFraudDistributionChart() {
    console.log('Loading fraud distribution chart...');
    fetch('/api/chart/fraud_distribution')
        .then(response => {
            console.log('Fraud distribution response:', response.status);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return response.json();
        })
        .then(data => {
            console.log('Fraud distribution data:', data);
            const container = document.getElementById('fraudDistributionChart');
            if (container) {
                Plotly.newPlot('fraudDistributionChart', data.data, data.layout, {
                    responsive: true,
                    displayModeBar: true
                });
                console.log('Fraud distribution chart rendered');
            } else {
                console.error('Container not found: fraudDistributionChart');
            }
        })
        .catch(error => console.error('Error loading fraud distribution chart:', error));
}

function loadRiskDistributionChart() {
    console.log('Loading risk distribution chart...');
    fetch('/api/chart/risk_distribution')
        .then(response => {
            console.log('Risk distribution response:', response.status);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return response.json();
        })
        .then(data => {
            console.log('Risk distribution data:', data);
            const container = document.getElementById('riskDistributionChart');
            if (container) {
                Plotly.newPlot('riskDistributionChart', data.data, data.layout, {
                    responsive: true,
                    displayModeBar: true
                });
                console.log('Risk distribution chart rendered');
            } else {
                console.error('Container not found: riskDistributionChart');
            }
        })
        .catch(error => console.error('Error loading risk distribution chart:', error));
}

function loadRiskByLocationChart() {
    console.log('Loading risk by location chart...');
    fetch('/api/chart/risk_by_location')
        .then(response => {
            console.log('Risk by location response:', response.status);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return response.json();
        })
        .then(data => {
            console.log('Risk by location data:', data);
            const container = document.getElementById('riskByLocationChart');
            if (container) {
                Plotly.newPlot('riskByLocationChart', data.data, data.layout, {
                    responsive: true,
                    displayModeBar: true
                });
                console.log('Risk by location chart rendered');
            } else {
                console.error('Container not found: riskByLocationChart');
            }
        })
        .catch(error => console.error('Error loading risk by location chart:', error));
}

function loadTurnoverVsRiskChart() {
    console.log('Loading turnover vs risk chart...');
    fetch('/api/chart/turnover_vs_risk')
        .then(response => {
            console.log('Turnover vs risk response:', response.status);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return response.json();
        })
        .then(data => {
            console.log('Turnover vs risk data:', data);
            const container = document.getElementById('turnoverVsRiskChart');
            if (container) {
                Plotly.newPlot('turnoverVsRiskChart', data.data, data.layout, {
                    responsive: true,
                    displayModeBar: true
                });
                console.log('Turnover vs risk chart rendered');
            } else {
                console.error('Container not found: turnoverVsRiskChart');
            }
        })
        .catch(error => console.error('Error loading turnover vs risk chart:', error));
}
