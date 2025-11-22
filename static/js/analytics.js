// ============================================================================
// Analytics Page - JavaScript
// ============================================================================

document.addEventListener('DOMContentLoaded', function() {
    console.log('Analytics page loaded');
    loadStatistics();
    loadTopSendersChart();
    loadTopReceiversChart();
});

function loadStatistics() {
    console.log('Loading statistics...');
    fetch('/api/statistics')
        .then(response => {
            console.log('Statistics response:', response.status);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('Statistics data:', data);
            
            // Check for errors
            if (data.error) {
                console.error('API returned error:', data.error);
                showError('Error loading statistics: ' + data.error);
                return;
            }
            
            // Update stat cards
            if (document.getElementById('totalNodes')) {
                document.getElementById('totalNodes').textContent = data.total_companies || '-';
            }
            if (document.getElementById('totalEdges')) {
                document.getElementById('totalEdges').textContent = data.total_edges || '-';
            }
            if (document.getElementById('networkDensity')) {
                // Calculate network density (simplified)
                const density = data.total_edges && data.total_companies > 1 
                    ? (data.total_edges / (data.total_companies * (data.total_companies - 1) / 2)).toFixed(4)
                    : '0';
                document.getElementById('networkDensity').textContent = density;
            }
            if (document.getElementById('avgFraudProb')) {
                const avgProb = data.average_fraud_probability 
                    ? (data.average_fraud_probability * 100).toFixed(2) + '%'
                    : '-';
                document.getElementById('avgFraudProb').textContent = avgProb;
            }
            
            // Update risk distribution
            if (document.getElementById('highRiskValue')) {
                document.getElementById('highRiskValue').textContent = data.high_risk_count || 0;
            }
            if (document.getElementById('mediumRiskValue')) {
                document.getElementById('mediumRiskValue').textContent = data.medium_risk_count || 0;
            }
            if (document.getElementById('lowRiskValue')) {
                document.getElementById('lowRiskValue').textContent = data.low_risk_count || 0;
            }
        })
        .catch(error => {
            console.error('Error loading statistics:', error);
            showError('Failed to load statistics. Check console for details.');
        });
}

function loadTopSendersChart() {
    console.log('Loading top senders chart...');
    const container = document.getElementById('topSendersChart');
    if (!container) {
        console.error('Container topSendersChart not found');
        return;
    }
    
    fetch('/api/top_senders')
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => {
                    throw new Error(err.error || `HTTP ${response.status}`);
                });
            }
            return response.json();
        })
        .then(data => {
            console.log('Top senders chart data:', data);
            
            // Check for error in response
            if (data.error) {
                throw new Error(data.error);
            }
            
            if (data.data && data.layout) {
                // Add margins to prevent toolbar overlay and remove text below
                const layout = {
                    ...data.layout,
                    margin: {
                        l: 70,
                        r: 50,
                        t: 80,
                        b: 40
                    },
                    xaxis: {
                        ...data.layout.xaxis,
                        title: {
                            text: ''
                        },
                        showticklabels: true
                    },
                    yaxis: {
                        ...data.layout.yaxis,
                        title: {
                            ...data.layout.yaxis?.title,
                            font: {
                                size: 14
                            }
                        }
                    }
                };
                Plotly.newPlot('topSendersChart', data.data, layout, {
                    responsive: true,
                    displayModeBar: true,
                    modeBarButtonsToRemove: ['pan2d', 'lasso2d'],
                    displaylogo: false
                });
                console.log('Top senders chart rendered successfully');
            } else {
                throw new Error('Invalid chart data format - missing data or layout');
            }
        })
        .catch(error => {
            console.error('Error loading top senders chart:', error);
            container.innerHTML = `
                <div style="text-align: center; padding: 40px; color: #dc3545;">
                    <p style="font-size: 16px; font-weight: bold;">⚠️ Error loading chart</p>
                    <p style="font-size: 14px; margin-top: 10px;">${error.message}</p>
                    <p style="font-size: 12px; margin-top: 10px; color: #6c757d;">Check console for details</p>
                </div>
            `;
        });
}

function loadTopReceiversChart() {
    console.log('Loading top receivers chart...');
    const container = document.getElementById('topReceiversChart');
    if (!container) {
        console.error('Container topReceiversChart not found');
        return;
    }
    
    fetch('/api/top_receivers')
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => {
                    throw new Error(err.error || `HTTP ${response.status}`);
                });
            }
            return response.json();
        })
        .then(data => {
            console.log('Top receivers chart data:', data);
            
            // Check for error in response
            if (data.error) {
                throw new Error(data.error);
            }
            
            if (data.data && data.layout) {
                // Add margins to prevent toolbar overlay and remove text below
                const layout = {
                    ...data.layout,
                    margin: {
                        l: 70,
                        r: 50,
                        t: 80,
                        b: 40
                    },
                    xaxis: {
                        ...data.layout.xaxis,
                        title: {
                            text: ''
                        },
                        showticklabels: true
                    },
                    yaxis: {
                        ...data.layout.yaxis,
                        title: {
                            ...data.layout.yaxis?.title,
                            font: {
                                size: 14
                            }
                        }
                    }
                };
                Plotly.newPlot('topReceiversChart', data.data, layout, {
                    responsive: true,
                    displayModeBar: true,
                    modeBarButtonsToRemove: ['pan2d', 'lasso2d'],
                    displaylogo: false
                });
                console.log('Top receivers chart rendered successfully');
            } else {
                throw new Error('Invalid chart data format - missing data or layout');
            }
        })
        .catch(error => {
            console.error('Error loading top receivers chart:', error);
            container.innerHTML = `
                <div style="text-align: center; padding: 40px; color: #dc3545;">
                    <p style="font-size: 16px; font-weight: bold;">⚠️ Error loading chart</p>
                    <p style="font-size: 14px; margin-top: 10px;">${error.message}</p>
                    <p style="font-size: 12px; margin-top: 10px; color: #6c757d;">Check console for details</p>
                </div>
            `;
        });
}

function showError(message) {
    // Show error in UI elements
    ['totalNodes', 'totalEdges', 'networkDensity', 'avgFraudProb', 'highRiskValue', 'mediumRiskValue', 'lowRiskValue'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.textContent = 'Error';
    });
    console.error(message);
}
