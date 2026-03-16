#!/usr/bin/env python3
# Evaluator for email Query 14

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class TemplateHTMLParser(HTMLParser):
    """Parse HTML to extract template information."""

    def __init__(self):
        super().__init__()
        self.templates = []
        self.current_template = {}
        self.in_template_card = False
        self.in_template_name = False
        self.in_detail_label = False
        self.in_detail_value = False
        self.in_template_preview = False
        self.current_label = ""
        self.current_data = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        if tag == 'div' and attrs_dict.get('class') == 'template-card':
            self.in_template_card = True
            self.current_template = {}
        elif self.in_template_card:
            if tag == 'h3' and 'template-name' in attrs_dict.get('class', ''):
                self.in_template_name = True
            elif tag == 'span' and 'detail-label' in attrs_dict.get('class', ''):
                self.in_detail_label = True
            elif tag == 'span' and 'detail-value' in attrs_dict.get('class', ''):
                self.in_detail_value = True
            elif tag == 'div' and 'template-preview' in attrs_dict.get('class', ''):
                self.in_template_preview = True

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
            
        if self.in_template_name:
            self.current_template['name'] = data
        elif self.in_detail_label:
            self.current_label = data.replace(':', '').strip()
        elif self.in_detail_value:
            if self.current_label:
                self.current_template[self.current_label.lower()] = data
                self.current_label = ""
        elif self.in_template_preview:
            self.current_template['body'] = data

    def handle_endtag(self, tag):
        if tag == 'div':
            if self.in_template_preview:
                self.in_template_preview = False
            elif self.in_template_card:
                if self.current_template:
                    self.templates.append(self.current_template)
                    self.current_template = {}
                self.in_template_card = False
        elif tag == 'h3':
            self.in_template_name = False
        elif tag == 'span':
            if self.in_detail_label:
                self.in_detail_label = False
            elif self.in_detail_value:
                self.in_detail_value = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints for template creation task
        self.checkpoints = {
            "cp1_navigated_to_templates": False,
            "cp2_template_name_correct": False,
            "cp3_subject_correct": False,
            "cp4_body_correct": False,
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
        """Find intermediate step HTML files."""
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

    def parse_templates_from_html(self, html_file: Path) -> List[Dict]:
        """Parse templates from HTML file."""
        parser = TemplateHTMLParser()
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
            parser.feed(html_content)
        return parser.templates

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Agent navigated to Templates page"""
        try:
            trajectory = self.get_trajectory()
            
            # Check if agent clicked on Templates
            for entry in trajectory:
                action = entry.get('action', {})
                selector = action.get('parameters', {}).get('selector', '')
                
                if 'Templates' in selector or 'templates' in selector.lower():
                    self.details['templates_navigation'] = "Agent navigated to Templates page"
                    return True
            
            # Check final HTML for Templates page
            final_html = self.find_final_html()
            if final_html:
                with open(final_html, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'class="templates-main"' in content or 'Email Templates' in content:
                        self.details['templates_navigation'] = "Found Templates page in final state"
                        return True
            
            self.issues.append("Agent did not navigate to Templates page")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking templates navigation: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Template name is 'Meeting Reminder Template'"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML file found")
                return False

            templates = self.parse_templates_from_html(final_html)
            
            # Look for the new template
            for template in templates:
                template_name = template.get('name', '')
                if template_name == 'Meeting Reminder Template':
                    self.details['template_name'] = template_name
                    return True
            
            self.issues.append("Template 'Meeting Reminder Template' not found in templates list")
            self.details['found_templates'] = [t.get('name', '') for t in templates]
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking template name: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Subject is 'Meeting Reminder'"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML file found")
                return False

            templates = self.parse_templates_from_html(final_html)
            
            # Find the Meeting Reminder Template
            for template in templates:
                if template.get('name') == 'Meeting Reminder Template':
                    subject = template.get('subject', '')
                    self.details['template_subject'] = subject
                    
                    if subject == 'Meeting Reminder':
                        return True
                    else:
                        self.issues.append(f"Template subject is '{subject}', expected 'Meeting Reminder'")
                        return False
            
            self.issues.append("Template 'Meeting Reminder Template' not found to check subject")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking template subject: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Body is 'This is a reminder for our upcoming meeting.'"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML file found")
                return False

            templates = self.parse_templates_from_html(final_html)
            
            # Find the Meeting Reminder Template
            for template in templates:
                if template.get('name') == 'Meeting Reminder Template':
                    body = template.get('body', '')
                    self.details['template_body'] = body
                    
                    expected_body = 'This is a reminder for our upcoming meeting.'
                    if body == expected_body:
                        return True
                    else:
                        self.issues.append(f"Template body is '{body}', expected '{expected_body}'")
                        return False
            
            self.issues.append("Template 'Meeting Reminder Template' not found to check body")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking template body: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_navigated_to_templates'] = self.check_checkpoint_1()
            self.checkpoints['cp2_template_name_correct'] = self.check_checkpoint_2()
            self.checkpoints['cp3_subject_correct'] = self.check_checkpoint_3()
            self.checkpoints['cp4_body_correct'] = self.check_checkpoint_4()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 14,
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
                'query_id': 14,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python email_query14.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
