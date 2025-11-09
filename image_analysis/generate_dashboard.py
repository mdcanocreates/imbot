"""
Generate HTML dashboard for visualizing analysis results.
"""

import json
from pathlib import Path
from typing import Optional
import pandas as pd


def generate_html_dashboard(
    metrics_csv_path: Path,
    output_dir: Path,
    qc_images_dir: Path,
    gemini_qc_path: Optional[Path] = None
) -> None:
    """
    Generate an HTML dashboard showing analysis results.
    
    Parameters
    ----------
    metrics_csv_path : Path
        Path to metrics.csv file
    output_dir : Path
        Directory where dashboard.html will be saved
    qc_images_dir : Path
        Directory containing QC images
    gemini_qc_path : Path, optional
        Path to gemini_qc_results.json file
    """
    # Load metrics
    metrics_df = pd.read_csv(metrics_csv_path, index_col='cell_id')
    
    # Convert to JSON for JavaScript
    metrics_json = metrics_df.to_dict(orient='index')
    
    # Get list of QC images
    qc_images = {}
    for cell_id in metrics_df.index:
        combo_path = qc_images_dir / f"{cell_id}_combo_with_masks.png"
        actin_path = qc_images_dir / f"{cell_id}_actin_with_cell_mask.png"
        nuclei_path = qc_images_dir / f"{cell_id}_nuclei_with_nuclear_mask.png"
        
        qc_images[cell_id] = {
            'combo': str(combo_path.relative_to(output_dir)) if combo_path.exists() else None,
            'actin': str(actin_path.relative_to(output_dir)) if actin_path.exists() else None,
            'nuclei': str(nuclei_path.relative_to(output_dir)) if nuclei_path.exists() else None
        }
    
    # Load Gemini QC results if available
    gemini_qc_data = {}
    if gemini_qc_path and gemini_qc_path.exists():
        try:
            with open(gemini_qc_path, 'r') as f:
                gemini_qc_list = json.load(f)
                # Convert list to dict keyed by cell_id
                for qc in gemini_qc_list:
                    cell_id = qc.get('cell_id')
                    if cell_id:
                        gemini_qc_data[cell_id] = qc
        except Exception as e:
            print(f"Warning: Could not load Gemini QC results: {e}")
    
    # HTML template
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cell Image Analysis Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            padding: 30px;
        }}
        
        h1 {{
            color: #333;
            margin-bottom: 10px;
            font-size: 2.5em;
        }}
        
        .subtitle {{
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }}
        
        .section {{
            margin-bottom: 40px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
        }}
        
        .section h2 {{
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.8em;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            background: white;
            border-radius: 8px;
            overflow: hidden;
        }}
        
        .metrics-table th {{
            background: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        
        .metrics-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #e0e0e0;
        }}
        
        .metrics-table tr:hover {{
            background: #f0f0f0;
        }}
        
        .cell-row {{
            font-weight: 600;
            color: #667eea;
        }}
        
        .images-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        
        .image-card {{
            background: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .image-card h3 {{
            color: #333;
            margin-bottom: 10px;
            font-size: 1.2em;
        }}
        
        .image-card img {{
            width: 100%;
            height: auto;
            border-radius: 8px;
            border: 2px solid #e0e0e0;
        }}
        
        .chart-container {{
            position: relative;
            height: 400px;
            margin-top: 20px;
            background: white;
            padding: 20px;
            border-radius: 10px;
        }}
        
        .similarity-results {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }}
        
        .similarity-results h3 {{
            color: #667eea;
            margin-bottom: 15px;
        }}
        
        .distance-item {{
            padding: 10px;
            margin: 5px 0;
            background: #f8f9fa;
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }}
        
        .conclusion {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            margin-top: 20px;
            font-size: 1.1em;
        }}
        
        .conclusion h3 {{
            margin-bottom: 15px;
            font-size: 1.5em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 Cell Image Analysis Dashboard</h1>
        <p class="subtitle">Quantitative comparison of endothelial cells</p>
        
        <!-- Metrics Table -->
        <div class="section">
            <h2>📊 Cell Metrics</h2>
            <table class="metrics-table" id="metricsTable">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th id="cellHeaders"></th>
                    </tr>
                </thead>
                <tbody id="metricsBody"></tbody>
            </table>
        </div>
        
        <!-- QC Images -->
        <div class="section">
            <h2>🖼️ Quality Control Images</h2>
            <div class="images-grid" id="imagesGrid"></div>
        </div>
        
        <!-- Charts -->
        <div class="section">
            <h2>📈 Metrics Visualization</h2>
            <div class="chart-container">
                <canvas id="metricsChart"></canvas>
            </div>
        </div>
        
        <!-- Similarity Analysis -->
        <div class="section">
            <h2>🔍 Similarity Analysis</h2>
            <div class="similarity-results" id="similarityResults"></div>
        </div>
        
        <!-- Gemini QC Results -->
        <div class="section">
            <h2>🤖 Gemini QC Evaluation</h2>
            <div class="gemini-qc-results" id="geminiQCResults"></div>
        </div>
    </div>
    
    <script>
        // Metrics data
        const metricsData = {json.dumps(metrics_json, indent=8)};
        
        // QC Images
        const qcImages = {json.dumps(qc_images, indent=8)};
        
        // Gemini QC Data
        const geminiQCData = {json.dumps(gemini_qc_data, indent=8)};
        
        // Populate metrics table
        function populateMetricsTable() {{
            const cellIds = Object.keys(metricsData);
            const metrics = Object.keys(metricsData[cellIds[0]]);
            
            // Header
            const headerRow = document.getElementById('cellHeaders');
            cellIds.forEach(cellId => {{
                const th = document.createElement('th');
                th.textContent = cellId;
                headerRow.appendChild(th);
            }});
            
            // Body
            const tbody = document.getElementById('metricsBody');
            metrics.forEach(metric => {{
                const tr = document.createElement('tr');
                const metricName = metric.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase());
                
                const tdMetric = document.createElement('td');
                tdMetric.className = 'cell-row';
                tdMetric.textContent = metricName;
                tr.appendChild(tdMetric);
                
                cellIds.forEach(cellId => {{
                    const td = document.createElement('td');
                    const value = metricsData[cellId][metric];
                    td.textContent = typeof value === 'number' ? value.toFixed(4) : value;
                    tr.appendChild(td);
                }});
                
                tbody.appendChild(tr);
            }});
        }}
        
        // Populate QC images
        function populateQCImages() {{
            const grid = document.getElementById('imagesGrid');
            Object.keys(qcImages).forEach(cellId => {{
                const images = qcImages[cellId];
                
                if (images.combo) {{
                    const card = document.createElement('div');
                    card.className = 'image-card';
                    card.innerHTML = `
                        <h3>${{cellId}} - Combo with Masks</h3>
                        <img src="${{images.combo}}" alt="${{cellId}} combo">
                    `;
                    grid.appendChild(card);
                }}
                
                if (images.actin) {{
                    const card = document.createElement('div');
                    card.className = 'image-card';
                    card.innerHTML = `
                        <h3>${{cellId}} - Actin with Cell Mask</h3>
                        <img src="${{images.actin}}" alt="${{cellId}} actin">
                    `;
                    grid.appendChild(card);
                }}
            }});
        }}
        
        // Create metrics chart
        function createMetricsChart() {{
            const cellIds = Object.keys(metricsData);
            const metrics = Object.keys(metricsData[cellIds[0]]);
            
            // Select key metrics for visualization
            const keyMetrics = [
                'cell_area', 'circularity', 'aspect_ratio', 
                'nuclear_count', 'nc_ratio', 
                'actin_mean_intensity', 'actin_anisotropy', 'mtub_mean_intensity'
            ].filter(m => metrics.includes(m));
            
            const datasets = cellIds.map((cellId, idx) => {{
                const colors = ['#667eea', '#f093fb', '#4facfe'];
                return {{
                    label: cellId,
                    data: keyMetrics.map(metric => metricsData[cellId][metric]),
                    backgroundColor: colors[idx % colors.length] + '40',
                    borderColor: colors[idx % colors.length],
                    borderWidth: 2,
                    tension: 0.4
                }};
            }});
            
            const ctx = document.getElementById('metricsChart').getContext('2d');
            new Chart(ctx, {{
                type: 'radar',
                data: {{
                    labels: keyMetrics.map(m => m.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase())),
                    datasets: datasets
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        r: {{
                            beginAtZero: true
                        }}
                    }},
                    plugins: {{
                        legend: {{
                            position: 'top',
                        }},
                        title: {{
                            display: true,
                            text: 'Cell Metrics Comparison'
                        }}
                    }}
                }}
            }});
        }}
        
        // Calculate similarity
        function calculateSimilarity() {{
            const cellIds = Object.keys(metricsData);
            const metrics = Object.keys(metricsData[cellIds[0]]);
            
            // Select metrics for similarity (exclude cell_id if present)
            const similarityMetrics = metrics.filter(m => m !== 'cell_id');
            
            // Normalize metrics (z-score)
            const normalized = {{}};
            cellIds.forEach(cellId => {{
                normalized[cellId] = {{}};
            }});
            
            similarityMetrics.forEach(metric => {{
                const values = cellIds.map(cid => metricsData[cid][metric]);
                const mean = values.reduce((a, b) => a + b, 0) / values.length;
                const std = Math.sqrt(values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length);
                
                cellIds.forEach(cellId => {{
                    normalized[cellId][metric] = std > 0 ? (metricsData[cellId][metric] - mean) / std : 0;
                }});
            }});
            
            // Calculate pairwise distances
            const distances = {{}};
            for (let i = 0; i < cellIds.length; i++) {{
                for (let j = i + 1; j < cellIds.length; j++) {{
                    const cell1 = cellIds[i];
                    const cell2 = cellIds[j];
                    
                    let dist = 0;
                    similarityMetrics.forEach(metric => {{
                        const diff = normalized[cell1][metric] - normalized[cell2][metric];
                        dist += diff * diff;
                    }});
                    dist = Math.sqrt(dist);
                    
                    distances[`${{cell1}}-${{cell2}}`] = dist;
                }}
            }}
            
            // Find most similar pair
            let minDist = Infinity;
            let mostSimilar = null;
            Object.entries(distances).forEach(([pair, dist]) => {{
                if (dist < minDist) {{
                    minDist = dist;
                    mostSimilar = pair;
                }}
            }});
            
            // Display results
            const resultsDiv = document.getElementById('similarityResults');
            let html = '<h3>Pairwise Distances</h3>';
            
            Object.entries(distances).forEach(([pair, dist]) => {{
                const [c1, c2] = pair.split('-');
                html += `
                    <div class="distance-item">
                        <strong>d(${{c1}}, ${{c2}})</strong> = ${{dist.toFixed(4)}}
                    </div>
                `;
            }});
            
            const [c1, c2] = mostSimilar.split('-');
            const outlier = cellIds.find(cid => cid !== c1 && cid !== c2);
            
            html += `
                <div class="conclusion">
                    <h3>Conclusion</h3>
                    <p>Cells <strong>${{c1}}</strong> and <strong>${{c2}}</strong> are most similar (distance = ${{minDist.toFixed(4)}}).</p>
                    <p>Cell <strong>${{outlier}}</strong> is the outlier based on the chosen metrics.</p>
                </div>
            `;
            
            resultsDiv.innerHTML = html;
        }}
        
        // Populate Gemini QC results
        function populateGeminiQC() {{
            const resultsDiv = document.getElementById('geminiQCResults');
            
            if (!geminiQCData || Object.keys(geminiQCData).length === 0) {{
                resultsDiv.innerHTML = '<p style="color: #666; padding: 20px;">No Gemini QC results available. Make sure GEMINI_API_KEY is set and the API call succeeded.</p>';
                return;
            }}
            
            let html = '';
            const cellIds = Object.keys(metricsData);
            
            cellIds.forEach(cellId => {{
                const qc = geminiQCData[cellId];
                if (!qc) {{
                    html += `
                        <div class="gemini-qc-card" style="margin-bottom: 20px; padding: 15px; background: #f8f9fa; border-radius: 8px;">
                            <h3 style="color: #667eea; margin-bottom: 10px;">${{cellId}}</h3>
                            <p style="color: #666;">No QC evaluation available</p>
                        </div>
                    `;
                    return;
                }}
                
                const cellScore = qc.cell_mask_score;
                const nucleusScore = qc.nucleus_mask_score;
                const issues = qc.issues || [];
                const suggestions = qc.suggested_ops || [];
                
                // Determine score color
                const getScoreColor = (score) => {{
                    if (score === null || score === undefined) return '#999';
                    if (score >= 0.8) return '#4caf50';
                    if (score >= 0.6) return '#ff9800';
                    return '#f44336';
                }};
                
                const getScoreLabel = (score) => {{
                    if (score === null || score === undefined) return 'N/A';
                    if (score >= 0.8) return 'Excellent';
                    if (score >= 0.6) return 'Good';
                    return 'Needs Review';
                }};
                
                html += `
                    <div class="gemini-qc-card" style="margin-bottom: 20px; padding: 20px; background: white; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        <h3 style="color: #667eea; margin-bottom: 15px; font-size: 1.3em;">${{cellId}} - ${{qc.channel || 'nuclei'}} Channel</h3>
                        
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px;">
                            <div style="padding: 15px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid ${{getScoreColor(cellScore)}};">
                                <div style="font-size: 0.9em; color: #666; margin-bottom: 5px;">Cell Mask Score</div>
                                <div style="font-size: 1.8em; font-weight: bold; color: ${{getScoreColor(cellScore)}};">
                                    ${{cellScore !== null && cellScore !== undefined ? cellScore.toFixed(2) : 'N/A'}}
                                </div>
                                <div style="font-size: 0.8em; color: #999; margin-top: 5px;">${{getScoreLabel(cellScore)}}</div>
                            </div>
                            
                            <div style="padding: 15px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid ${{getScoreColor(nucleusScore)}};">
                                <div style="font-size: 0.9em; color: #666; margin-bottom: 5px;">Nuclear Mask Score</div>
                                <div style="font-size: 1.8em; font-weight: bold; color: ${{getScoreColor(nucleusScore)}};">
                                    ${{nucleusScore !== null && nucleusScore !== undefined ? nucleusScore.toFixed(2) : 'N/A'}}
                                </div>
                                <div style="font-size: 0.8em; color: #999; margin-top: 5px;">${{getScoreLabel(nucleusScore)}}</div>
                            </div>
                        </div>
                        
                        ${{issues.length > 0 ? `
                            <div style="margin-bottom: 15px;">
                                <h4 style="color: #f44336; margin-bottom: 10px; font-size: 1em;">⚠️ Issues Identified:</h4>
                                <ul style="margin-left: 20px; color: #555;">
                                    ${{issues.map(issue => `<li style="margin-bottom: 5px;">${{issue}}</li>`).join('')}}
                                </ul>
                            </div>
                        ` : ''}}
                        
                        ${{suggestions.length > 0 ? `
                            <div style="margin-bottom: 15px;">
                                <h4 style="color: #2196f3; margin-bottom: 10px; font-size: 1em;">💡 Suggested Improvements:</h4>
                                <ul style="margin-left: 20px; color: #555;">
                                    ${{suggestions.map(suggestion => `<li style="margin-bottom: 5px;">${{suggestion}}</li>`).join('')}}
                                </ul>
                            </div>
                        ` : ''}}
                    </div>
                `;
            }});
            
            resultsDiv.innerHTML = html;
        }}
        
        // Initialize dashboard
        window.onload = function() {{
            populateMetricsTable();
            populateQCImages();
            createMetricsChart();
            calculateSimilarity();
            populateGeminiQC();
        }};
    </script>
</body>
</html>
"""
    
    # Write HTML file
    dashboard_path = output_dir / "dashboard.html"
    with open(dashboard_path, 'w') as f:
        f.write(html_content)
    
    print(f"Dashboard generated: {dashboard_path}")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    if len(sys.argv) < 2:
        print("Usage: python generate_dashboard.py <output_dir>")
        sys.exit(1)
    
    output_dir = Path(sys.argv[1])
    metrics_csv = output_dir / "metrics.csv"
    qc_images_dir = output_dir
    
    if not metrics_csv.exists():
        print(f"Error: {metrics_csv} not found. Run the analysis first.")
        sys.exit(1)
    
    generate_html_dashboard(metrics_csv, output_dir, qc_images_dir)

