#!/usr/bin/env python3
# Evaluator for transportation Query 2

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class EmergencyContactParser(HTMLParser):
    """Parse HTML to extract emergency contact information."""

    def __init__(self):
        super().__init__()
        self.contact_name = None
        self.contact_phone = None
        self.contact_relationship = None
        self.ride_sharing_enabled = False
        self.in_contact_name = False
        self.in_contact_phone = False
        self.in_contact_relationship = False
        self.in_status_indicator = False
        self.current_data = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check for contact name
        if tag in ['h2', 'h1'] and 'class' in attrs_dict:
            if 'contact-name-large' in attrs_dict['class']:
                self.in_contact_name = True
                self.current_data = []
        
        # Check for relationship
        if tag == 'p' and 'class' in attrs_dict:
            if 'contact-relationship-large' in attrs_dict['class']:
                self.in_contact_relationship = True
                self.current_data = []
        
        # Check for phone value
        if tag == 'p' and 'class' in attrs_dict:
            if 'info-value' in attrs_dict['class']:
                self.current_data = []
        
        # Check for ride sharing status
        if tag == 'div' and 'class' in attrs_dict:
            if 'status-indicator' in attrs_dict['class'] and 'enabled' in attrs_dict['class']:
                self.ride_sharing_enabled = True

    def handle_data(self, data):
        data = data.strip()
        if data:
            if self.in_contact_name:
                self.contact_name = data
                self.in_contact_name = False
            elif self.in_contact_relationship:
                self.contact_relationship = data
                self.in_contact_relationship = False
            else:
                # Check if this is a phone number pattern
                if re.match(r'^\d{3}-\d{3}-\d{4}$', data):
                    self.contact_phone = data


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_navigation_to_account": False,
            "cp2_contact_created_with_name": False,
            "cp3_phone_number_correct": False,
            "cp4_relationship_set_to_sibling": False,
            "cp5_ride_sharing_enabled": False,
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

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Navigate to Account section"""
        try:
            trajectory = self.get_trajectory()
            
            # Check if the agent clicked on Account link in the first few steps
            for action in trajectory[:5]:
                if action.get('action', {}).get('action_type') == 'CLICK':
                    selector = action.get('action', {}).get('parameters', {}).get('selector', '')
                    if 'Account' in selector or 'account' in selector.lower():
                        self.details['account_navigation'] = 'Successfully navigated to Account section'
                        return True
            
            self.issues.append("Did not navigate to Account section")
            return False
        except Exception as e:
            self.issues.append(f"Error checking account navigation: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Contact created with name 'Friend'"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML file found")
                return False
            
            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            parser = EmergencyContactParser()
            parser.feed(html_content)
            
            if parser.contact_name and parser.contact_name == "Friend":
                self.details['contact_name'] = parser.contact_name
                return True
            else:
                self.issues.append(f"Contact name is '{parser.contact_name}', expected 'Friend'")
                return False
        except Exception as e:
            self.issues.append(f"Error checking contact name: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Phone number set to '123-123-1234'"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML file found")
                return False
            
            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            parser = EmergencyContactParser()
            parser.feed(html_content)
            
            if parser.contact_phone and parser.contact_phone == "123-123-1234":
                self.details['contact_phone'] = parser.contact_phone
                return True
            else:
                self.issues.append(f"Phone number is '{parser.contact_phone}', expected '123-123-1234'")
                return False
        except Exception as e:
            self.issues.append(f"Error checking phone number: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Relationship set to 'Sibling'"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML file found")
                return False
            
            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            parser = EmergencyContactParser()
            parser.feed(html_content)
            
            if parser.contact_relationship and parser.contact_relationship == "Sibling":
                self.details['contact_relationship'] = parser.contact_relationship
                return True
            else:
                self.issues.append(f"Relationship is '{parser.contact_relationship}', expected 'Sibling'")
                return False
        except Exception as e:
            self.issues.append(f"Error checking relationship: {str(e)}")
            return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Ride Sharing enabled"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML file found")
                return False
            
            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            parser = EmergencyContactParser()
            parser.feed(html_content)
            
            if parser.ride_sharing_enabled:
                self.details['ride_sharing'] = 'Enabled'
                return True
            else:
                self.issues.append("Ride Sharing is not enabled")
                return False
        except Exception as e:
            self.issues.append(f"Error checking ride sharing status: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_navigation_to_account'] = self.check_checkpoint_1()
            self.checkpoints['cp2_contact_created_with_name'] = self.check_checkpoint_2()
            self.checkpoints['cp3_phone_number_correct'] = self.check_checkpoint_3()
            self.checkpoints['cp4_relationship_set_to_sibling'] = self.check_checkpoint_4()
            self.checkpoints['cp5_ride_sharing_enabled'] = self.check_checkpoint_5()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 2,
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
                'query_id': 2,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python transportation_query2.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
