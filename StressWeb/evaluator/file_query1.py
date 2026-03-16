#!/usr/bin/env python3
# Evaluator for file Query 1

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class AnalyticsPageParser(HTMLParser):
    """Parse HTML to detect Analytics page elements."""

    def __init__(self):
        super().__init__()
        self.in_analytics_main = False
        self.in_analytics_title = False
        self.analytics_title_text = ""
        self.analytics_stat_labels = []
        self.in_stat_label = False
        self.current_stat_label = ""
        self.has_analytics_section = False
        self.has_storage_analytics_title = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        class_attr = attrs_dict.get('class', '')
        
        if 'analytics-main' in class_attr:
            self.in_analytics_main = True
            self.has_analytics_section = True
        
        if 'analytics-title' in class_attr:
            self.in_analytics_title = True
        
        if 'analytics-stat-label' in class_attr:
            self.in_stat_label = True
            self.current_stat_label = ""

    def handle_data(self, data):
        text = data.strip()
        if self.in_analytics_title and text:
            self.analytics_title_text = text
            if 'Storage Analytics' in text:
                self.has_storage_analytics_title = True
        
        if self.in_stat_label and text:
            self.current_stat_label += text

    def handle_endtag(self, tag):
        if tag in ['div', 'main']:
            if self.in_analytics_main:
                # Check if we're closing the analytics-main div
                self.in_analytics_main = False
            if self.in_analytics_title:
                self.in_analytics_title = False
        
        if self.in_stat_label:
            if self.current_stat_label:
                self.analytics_stat_labels.append(self.current_stat_label.strip())
            self.in_stat_label = False
            self.current_stat_label = ""


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task
        self.checkpoints = {
            "cp1_navigated_to_myfiles": False,
            "cp2_clicked_view_analytics": False,
            "cp3_analytics_page_displayed": False,
        }

        self.issues = []
        self.details = {}

    def find_final_html(self) -> Optional[Path]:
        """Find final HTML state file."""
        final_files = list(self.result_dir.glob("final_*_raw.html"))
        if final_files:
            return final_files[0]
        step_files = sorted(self.result_dir.glob("step_*_raw.html"))
        if step_files:
            return step_files[-1]
        return None

    def find_step_html(self, step_pattern: str) -> List[Path]:
        """Find intermediate step HTML files.

        Args:
            step_pattern: Pattern like "step_*", "step_10_*", "step_[12]*"
        Returns:
            List of matching HTML files sorted by step number
        """
        return sorted(self.result_dir.glob(f"{step_pattern}_raw.html"))

    def get_trajectory(self) -> List[Dict]:
        """Load trajectory data from traj.jsonl."""
        if not self.traj_file.exists():
            return []

        trajectory = []
        with open(self.traj_file, 'r', encoding='utf-8') as f:
            for line in f:
                trajectory.append(json.loads(line))
        return trajectory

    def load_result_json(self) -> Dict:
        """Load result.json file."""
        if not self.result_file.exists():
            raise FileNotFoundError(f"result.json not found")

        with open(self.result_file, 'r') as f:
            return json.load(f)

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Navigated to My Files section"""
        try:
            trajectory = self.get_trajectory()
            
            # Check if agent clicked on My Files link
            for step in trajectory:
                action = step.get('action', {})
                action_type = action.get('action_type', '')
                
                if action_type == 'CLICK':
                    selector = action.get('parameters', {}).get('selector', '')
                    
                    # Check for My Files navigation
                    if 'My Files' in selector or 'my-files' in selector.lower() or 'myfiles' in selector.lower():
                        self.details['myfiles_navigation'] = 'Agent clicked on My Files'
                        return True
            
            # Also check if URL contains my-files or myfiles
            for step in trajectory:
                url = step.get('url', '')
                if 'my-files' in url.lower() or 'myfiles' in url.lower():
                    self.details['myfiles_navigation'] = 'Agent navigated to My Files URL'
                    return True
            
            self.issues.append("Agent did not navigate to My Files section")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking My Files navigation: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Clicked View Analytics button"""
        try:
            trajectory = self.get_trajectory()
            
            # Check if agent clicked on View Analytics
            for step in trajectory:
                action = step.get('action', {})
                action_type = action.get('action_type', '')
                
                if action_type == 'CLICK':
                    selector = action.get('parameters', {}).get('selector', '')
                    
                    # Check for View Analytics click
                    if 'View Analytics' in selector or 'analytics' in selector.lower():
                        self.details['analytics_click'] = 'Agent clicked on View Analytics'
                        return True
            
            self.issues.append("Agent did not click on View Analytics")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking View Analytics click: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Analytics page is displayed in final state"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML file found")
                return False
            
            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Parse HTML to check for Analytics page
            parser = AnalyticsPageParser()
            parser.feed(html_content)
            
            # Check if Analytics page elements are present
            if parser.has_analytics_section and parser.has_storage_analytics_title:
                self.details['analytics_page'] = 'Analytics page displayed with Storage Analytics title'
                self.details['analytics_stats'] = parser.analytics_stat_labels
                return True
            
            # Alternative check: Look for analytics-related classes or text
            if 'analytics-main' in html_content and 'Storage Analytics' in html_content:
                self.details['analytics_page'] = 'Analytics page elements found in HTML'
                return True
            
            # Check URL in trajectory
            trajectory = self.get_trajectory()
            if trajectory:
                last_url = trajectory[-1].get('url', '')
                if 'analytics' in last_url.lower():
                    self.details['analytics_url'] = f'Final URL contains analytics: {last_url}'
                    # Even if URL is correct, HTML should show analytics page
                    if 'analytics-main' not in html_content:
                        self.issues.append("URL is /analytics but page content doesn't show Analytics page")
                        return False
                    return True
            
            self.issues.append("Analytics page is not displayed in final state")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking Analytics page display: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_navigated_to_myfiles'] = self.check_checkpoint_1()
            self.checkpoints['cp2_clicked_view_analytics'] = self.check_checkpoint_2()
            self.checkpoints['cp3_analytics_page_displayed'] = self.check_checkpoint_3()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 1,
                'query': result_data.get('query', ''),
                'overall_success': overall_success,
                'success_rate': success_rate,
                'checkpoints': self.checkpoints,
                'checkpoints_passed': f"{passed_count}/{total_count}",
                'issues': self.issues,
                'details': self.details,
                'agent_claimed_success': result_data.get('success', False),
                'execution_time': result_data.get('execution_time', 0),
                'total_steps': result_data.get('steps', 0)
            }

        except Exception as e:
            return {
                'query_id': 1,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python file_query1.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
