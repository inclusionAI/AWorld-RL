#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HTML visualization report generator
Generate interactive HTML report from JSON evaluation results
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import re


class HTMLReportGenerator:
    """HTML Report Generator"""

    def __init__(self, json_report_path: str):
        self.json_path = Path(json_report_path)
        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def generate_html(self) -> str:
        """Generate complete HTML report"""
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SimWebBench Evaluation Report</title>
    <style>
        {self.get_css()}
    </style>
</head>
<body>
    <div class="container">
        {self.generate_header()}
        {self.generate_overview()}
        {self.generate_model_comparison()}
        {self.generate_mode_comparison()}
        {self.generate_project_comparison()}
        {self.generate_detailed_table()}
        {self.generate_model_mode_heatmap()}
    </div>
    <script>
        {self.get_javascript()}
    </script>
</body>
</html>"""
        return html

    def get_css(self) -> str:
        """Generate CSS styles"""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        }

        h1 {
            color: #2d3748;
            font-size: 2.5em;
            margin-bottom: 10px;
            text-align: center;
        }

        h2 {
            color: #4a5568;
            font-size: 1.8em;
            margin-top: 40px;
            margin-bottom: 20px;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }

        .header {
            text-align: center;
            margin-bottom: 40px;
        }

        .timestamp {
            color: #718096;
            font-size: 1em;
        }

        .overview {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            transition: transform 0.3s, box-shadow 0.3s;
        }

        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
        }

        .stat-card h3 {
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 10px;
        }

        .stat-card .value {
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 5px;
        }

        .stat-card .detail {
            font-size: 0.85em;
            opacity: 0.8;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }

        thead {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        th {
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }

        td {
            padding: 12px 15px;
            border-bottom: 1px solid #e2e8f0;
        }

        tbody tr:hover {
            background-color: #f7fafc;
        }

        .progress-bar {
            width: 100%;
            height: 25px;
            background-color: #e2e8f0;
            border-radius: 12px;
            overflow: hidden;
            position: relative;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #48bb78 0%, #38a169 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 0.75em;
            font-weight: bold;
            transition: width 0.5s ease;
        }

        .progress-fill.medium {
            background: linear-gradient(90deg, #ed8936 0%, #dd6b20 100%);
        }

        .progress-fill.low {
            background: linear-gradient(90deg, #f56565 0%, #e53e3e 100%);
        }

        .heatmap {
            display: grid;
            gap: 2px;
            margin: 20px 0;
        }

        .heatmap-row {
            display: grid;
            gap: 2px;
        }

        .heatmap-cell {
            padding: 15px;
            text-align: center;
            border-radius: 5px;
            color: white;
            font-weight: 600;
            transition: transform 0.2s;
        }

        .heatmap-cell:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }

        .heatmap-header {
            background: #4a5568;
            font-weight: bold;
        }

        .comparison-section {
            margin: 30px 0;
        }

        .chart-container {
            margin: 20px 0;
        }

        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
        }

        .badge-success {
            background: #c6f6d5;
            color: #22543d;
        }

        .badge-warning {
            background: #feebc8;
            color: #744210;
        }

        .badge-danger {
            background: #fed7d7;
            color: #742a2a;
        }

        .filter-buttons {
            margin: 20px 0;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }

        .filter-btn {
            padding: 8px 16px;
            border: 2px solid #667eea;
            background: white;
            color: #667eea;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s;
        }

        .filter-btn:hover, .filter-btn.active {
            background: #667eea;
            color: white;
        }

        .detailed-section {
            margin-top: 40px;
        }

        .batch-details {
            display: none;
            margin-top: 10px;
            padding: 15px;
            background: #f7fafc;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }

        .expandable {
            cursor: pointer;
            user-select: none;
        }

        .expandable:hover {
            background-color: #edf2f7;
        }

        .expand-icon {
            display: inline-block;
            margin-right: 8px;
            transition: transform 0.3s;
        }

        .expand-icon.expanded {
            transform: rotate(90deg);
        }

        @media (max-width: 768px) {
            .container {
                padding: 20px;
            }

            h1 {
                font-size: 1.8em;
            }

            .overview {
                grid-template-columns: 1fr;
            }

            table {
                font-size: 0.85em;
            }
        }
        """

    def generate_header(self) -> str:
        """Generate report header"""
        timestamp = self.data.get('evaluation_timestamp', 'N/A')
        total_batches = self.data.get('total_batches', 0)

        return f"""
        <div class="header">
            <h1>🎯 SimWebBench Evaluation Report</h1>
            <p class="timestamp">Generated: {timestamp}</p>
            <p class="timestamp">Total Batches: {total_batches}</p>
        </div>
        """

    def generate_overview(self) -> str:
        """Generate overview statistics cards"""
        summary = self.data.get('summary', {})

        # Calculate overall statistics
        total_queries = 0
        total_success = 0
        total_checkpoints = 0
        passed_checkpoints = 0
        total_steps = 0
        count = 0

        for batch in self.data.get('batches', []):
            if 'statistics' in batch:
                stats = batch['statistics']
                total_queries += stats.get('valid_queries', 0)
                total_success += stats.get('task_success_count', 0)
                total_checkpoints += stats.get('total_checkpoints', 0)
                passed_checkpoints += stats.get('passed_checkpoints', 0)
                total_steps += stats.get('avg_steps', 0) * stats.get('valid_queries', 0)
                count += 1

        success_rate = (total_success / total_queries * 100) if total_queries > 0 else 0
        checkpoint_rate = (passed_checkpoints / total_checkpoints * 100) if total_checkpoints > 0 else 0
        avg_steps = (total_steps / total_queries) if total_queries > 0 else 0

        return f"""
        <div class="overview">
            <div class="stat-card">
                <h3>Total Queries</h3>
                <div class="value">{total_queries}</div>
                <div class="detail">{len(self.data.get('batches', []))} batches</div>
            </div>
            <div class="stat-card">
                <h3>Task Success Rate</h3>
                <div class="value">{success_rate:.1f}%</div>
                <div class="detail">{total_success}/{total_queries} succeeded</div>
            </div>
            <div class="stat-card">
                <h3>Checkpoint Pass Rate</h3>
                <div class="value">{checkpoint_rate:.1f}%</div>
                <div class="detail">{passed_checkpoints}/{total_checkpoints} passed</div>
            </div>
            <div class="stat-card">
                <h3>Average Steps</h3>
                <div class="value">{avg_steps:.1f}</div>
                <div class="detail">per task</div>
            </div>
        </div>
        """

    def generate_model_comparison(self) -> str:
        """Generate model comparison table"""
        by_model = self.data.get('summary', {}).get('by_model', {})

        if not by_model:
            return ""

        rows = []
        for model in sorted(by_model.keys()):
            stats = by_model[model]
            task_rate = stats['task_success_rate'] * 100
            checkpoint_rate = stats['checkpoint_pass_rate'] * 100
            agent_rate = stats['agent_success_rate'] * 100

            task_class = self.get_rate_class(task_rate)
            checkpoint_class = self.get_rate_class(checkpoint_rate)

            rows.append(f"""
            <tr>
                <td><strong>{model}</strong></td>
                <td>{stats['batches_count']}</td>
                <td>{stats['total_queries']}</td>
                <td>
                    <div class="progress-bar">
                        <div class="progress-fill {task_class}" style="width: {task_rate}%">
                            {task_rate:.1f}%
                        </div>
                    </div>
                </td>
                <td>
                    <div class="progress-bar">
                        <div class="progress-fill {checkpoint_class}" style="width: {checkpoint_rate}%">
                            {checkpoint_rate:.1f}%
                        </div>
                    </div>
                </td>
                <td>{stats['avg_steps']:.1f}</td>
                <td>{agent_rate:.1f}%</td>
            </tr>
            """)

        return f"""
        <div class="comparison-section">
            <h2>📊 Comparison by Model</h2>
            <table>
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Batches</th>
                        <th>Queries</th>
                        <th>Task Success Rate</th>
                        <th>Checkpoint Pass Rate</th>
                        <th>Avg Steps</th>
                        <th>Agent Self-Rate</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
        """

    def generate_mode_comparison(self) -> str:
        """Generate mode comparison table"""
        by_mode = self.data.get('summary', {}).get('by_mode', {})

        if not by_mode:
            return ""

        # Define mode sorting
        mode_order = ['clean', 'chaos', 'failed', 'perturbed', 'semanticE', 'semanticS', 'dom']
        sorted_modes = sorted(by_mode.keys(), key=lambda x: mode_order.index(x) if x in mode_order else 99)

        rows = []
        for mode in sorted_modes:
            stats = by_mode[mode]
            task_rate = stats['task_success_rate'] * 100
            checkpoint_rate = stats['checkpoint_pass_rate'] * 100

            task_class = self.get_rate_class(task_rate)
            checkpoint_class = self.get_rate_class(checkpoint_rate)

            mode_badge = self.get_mode_badge(mode)

            rows.append(f"""
            <tr>
                <td>{mode_badge}</td>
                <td>{stats['batches_count']}</td>
                <td>{stats['total_queries']}</td>
                <td>
                    <div class="progress-bar">
                        <div class="progress-fill {task_class}" style="width: {task_rate}%">
                            {task_rate:.1f}%
                        </div>
                    </div>
                </td>
                <td>
                    <div class="progress-bar">
                        <div class="progress-fill {checkpoint_class}" style="width: {checkpoint_rate}%">
                            {checkpoint_rate:.1f}%
                        </div>
                    </div>
                </td>
                <td>{stats['avg_steps']:.1f}</td>
            </tr>
            """)

        return f"""
        <div class="comparison-section">
            <h2>🎭 Comparison by Mode</h2>
            <table>
                <thead>
                    <tr>
                        <th>Mode</th>
                        <th>Batches</th>
                        <th>Queries</th>
                        <th>Task Success Rate</th>
                        <th>Checkpoint Pass Rate</th>
                        <th>Avg Steps</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
        """

    def generate_project_comparison(self) -> str:
        """Generate project comparison table"""
        by_project = self.data.get('summary', {}).get('by_project', {})

        if not by_project:
            return ""

        rows = []
        for project in sorted(by_project.keys()):
            stats = by_project[project]
            task_rate = stats['task_success_rate'] * 100
            checkpoint_rate = stats['checkpoint_pass_rate'] * 100

            task_class = self.get_rate_class(task_rate)
            checkpoint_class = self.get_rate_class(checkpoint_rate)

            rows.append(f"""
            <tr>
                <td><strong>{project}</strong></td>
                <td>{stats['batches_count']}</td>
                <td>{stats['total_queries']}</td>
                <td>
                    <div class="progress-bar">
                        <div class="progress-fill {task_class}" style="width: {task_rate}%">
                            {task_rate:.1f}%
                        </div>
                    </div>
                </td>
                <td>
                    <div class="progress-bar">
                        <div class="progress-fill {checkpoint_class}" style="width: {checkpoint_rate}%">
                            {checkpoint_rate:.1f}%
                        </div>
                    </div>
                </td>
                <td>{stats['avg_steps']:.1f}</td>
            </tr>
            """)

        return f"""
        <div class="comparison-section">
            <h2>📁 Comparison by Project</h2>
            <table>
                <thead>
                    <tr>
                        <th>Project</th>
                        <th>Batches</th>
                        <th>Queries</th>
                        <th>Task Success Rate</th>
                        <th>Checkpoint Pass Rate</th>
                        <th>Avg Steps</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
        """

    def generate_model_mode_heatmap(self) -> str:
        """Generate model-mode heatmap"""
        by_model_mode = self.data.get('summary', {}).get('by_model_mode', {})

        if not by_model_mode:
            return ""

        # Extract all models and modes
        models = set()
        modes = set()
        for key in by_model_mode.keys():
            if '_' in key:
                model, mode = key.rsplit('_', 1)
                models.add(model)
                modes.add(mode)

        models = sorted(models)
        mode_order = ['clean', 'chaos', 'failed', 'perturbed', 'semanticE', 'semanticS', 'dom']
        modes = sorted(modes, key=lambda x: mode_order.index(x) if x in mode_order else 99)

        if not models or not modes:
            return ""

        # Generate table header
        header_cells = '<div class="heatmap-cell heatmap-header">Model/Mode</div>'
        for mode in modes:
            header_cells += f'<div class="heatmap-cell heatmap-header">{mode}</div>'

        header = f'<div class="heatmap-row" style="grid-template-columns: 150px repeat({len(modes)}, 1fr);">{header_cells}</div>'

        # Generate data rows
        rows = []
        for model in models:
            cells = f'<div class="heatmap-cell heatmap-header">{model}</div>'
            for mode in modes:
                key = f"{model}_{mode}"
                if key in by_model_mode:
                    stats = by_model_mode[key]
                    rate = stats['task_success_rate'] * 100
                    color = self.get_heatmap_color(rate)
                    cells += f'<div class="heatmap-cell" style="background-color: {color};">{rate:.1f}%</div>'
                else:
                    cells += '<div class="heatmap-cell" style="background-color: #e2e8f0; color: #a0aec0;">-</div>'

            rows.append(f'<div class="heatmap-row" style="grid-template-columns: 150px repeat({len(modes)}, 1fr);">{cells}</div>')

        return f"""
        <div class="comparison-section">
            <h2>🔥 Model-Mode Success Rate Heatmap</h2>
            <div class="heatmap">
                {header}
                {''.join(rows)}
            </div>
        </div>
        """

    def generate_detailed_table(self) -> str:
        """Generate detailed batch table"""
        batches = self.data.get('batches', [])

        if not batches:
            return ""

        rows = []
        for i, batch in enumerate(batches):
            if 'error' in batch and 'statistics' not in batch:
                continue

            batch_name = batch.get('batch_name', 'Unknown')
            project = batch.get('project', 'N/A')
            model = batch.get('model', 'N/A')
            mode = batch.get('mode', 'N/A')
            stats = batch.get('statistics', {})

            if not stats:
                continue

            task_rate = stats.get('task_success_rate', 0) * 100
            checkpoint_rate = stats.get('checkpoint_pass_rate', 0) * 100

            task_class = self.get_rate_class(task_rate)
            checkpoint_class = self.get_rate_class(checkpoint_rate)
            mode_badge = self.get_mode_badge(mode)

            # Generate detailed information
            queries_details = []
            for q in batch.get('queries', []):
                q_num = q.get('query_num', '?')
                q_success = '✓' if q.get('overall_success', False) else '✗'
                q_steps = q.get('total_steps', 0)
                queries_details.append(f"Query {q_num}: {q_success} ({q_steps} steps)")

            details_html = '<br>'.join(queries_details)

            rows.append(f"""
            <tr class="expandable" onclick="toggleDetails({i})">
                <td>
                    <span class="expand-icon" id="icon-{i}">▶</span>
                    {batch_name[:50]}...
                </td>
                <td>{project}</td>
                <td>{model}</td>
                <td>{mode_badge}</td>
                <td>{stats.get('valid_queries', 0)}</td>
                <td>{task_rate:.1f}%</td>
                <td>{checkpoint_rate:.1f}%</td>
                <td>{stats.get('avg_steps', 0):.1f}</td>
            </tr>
            <tr>
                <td colspan="8">
                    <div class="batch-details" id="details-{i}">
                        {details_html}
                    </div>
                </td>
            </tr>
            """)

        return f"""
        <div class="detailed-section">
            <h2>📋 Detailed Batch List</h2>
            <table id="detailedTable">
                <thead>
                    <tr>
                        <th>Batch Name</th>
                        <th>Project</th>
                        <th>Model</th>
                        <th>Mode</th>
                        <th>Queries</th>
                        <th>Task Success Rate</th>
                        <th>Checkpoint Pass Rate</th>
                        <th>Avg Steps</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
        """

    def get_rate_class(self, rate: float) -> str:
        """Return CSS class based on success rate"""
        if rate >= 70:
            return ""
        elif rate >= 40:
            return "medium"
        else:
            return "low"

    def get_heatmap_color(self, rate: float) -> str:
        """Return heatmap color based on success rate"""
        if rate >= 80:
            return "#38a169"  # Green
        elif rate >= 60:
            return "#48bb78"  # Light green
        elif rate >= 40:
            return "#ed8936"  # Orange
        elif rate >= 20:
            return "#f56565"  # Red
        else:
            return "#e53e3e"  # Dark red

    def get_mode_badge(self, mode: str) -> str:
        """Return mode badge HTML"""
        badge_map = {
            'clean': 'badge-success',
            'chaos': 'badge-warning',
            'failed': 'badge-danger',
            'perturbed': 'badge-warning',
            'semanticE': 'badge-warning',
            'semanticS': 'badge-warning',
            'dom': 'badge-warning'
        }
        badge_class = badge_map.get(mode, 'badge-success')
        return f'<span class="badge {badge_class}">{mode}</span>'

    def get_javascript(self) -> str:
        """Generate JavaScript code"""
        return """
        function toggleDetails(index) {
            const details = document.getElementById('details-' + index);
            const icon = document.getElementById('icon-' + index);

            if (details.style.display === 'block') {
                details.style.display = 'none';
                icon.classList.remove('expanded');
            } else {
                details.style.display = 'block';
                icon.classList.add('expanded');
            }
        }

        // Page load animation
        window.addEventListener('load', function() {
            const statCards = document.querySelectorAll('.stat-card');
            statCards.forEach((card, index) => {
                setTimeout(() => {
                    card.style.opacity = '0';
                    card.style.transform = 'translateY(20px)';
                    setTimeout(() => {
                        card.style.transition = 'all 0.5s ease';
                        card.style.opacity = '1';
                        card.style.transform = 'translateY(0)';
                    }, 50);
                }, index * 100);
            });
        });
        """

    def save_html(self, output_path: str = None) -> Path:
        """Save HTML report"""
        if output_path is None:
            output_path = self.json_path.parent / self.json_path.name.replace('.json', '.html')
        else:
            output_path = Path(output_path)

        html_content = self.generate_html()

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return output_path


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python generate_html_report.py <json_report_path> [output_html_path]")
        print("\nExamples:")
        print("  python generate_html_report.py comprehensive_evaluation_20260306_120000.json")
        print("  python generate_html_report.py report.json output.html")
        sys.exit(1)

    json_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    if not Path(json_path).exists():
        print(f"Error: JSON file does not exist: {json_path}")
        sys.exit(1)

    print(f"📊 Generating HTML report...")
    print(f"Input: {json_path}")

    generator = HTMLReportGenerator(json_path)
    output_file = generator.save_html(output_path)

    print(f"✅ HTML report generated: {output_file}")


if __name__ == "__main__":
    main()
