#!/usr/bin/env python3
# Evaluator for food Query 3

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class AddressHTMLParser(HTMLParser):
    """Parse HTML to extract address information."""

    def __init__(self):
        super().__init__()
        self.in_address_card = False
        self.in_address_content = False
        self.in_address_label = False
        self.in_address_text = False
        self.in_default_badge = False
        self.current_label = None
        self.current_address = None
        self.addresses = []
        self.default_address = None

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        if tag == 'div' and attrs_dict.get('class') == 'address-card':
            self.in_address_card = True
            self.current_label = None
            self.current_address = None
            
        if self.in_address_card:
            if tag == 'div' and 'default-badge-top' in attrs_dict.get('class', ''):
                self.in_default_badge = True
            elif tag == 'div' and attrs_dict.get('class') == 'address-content':
                self.in_address_content = True
            elif tag == 'h3' and 'address-label' in attrs_dict.get('class', ''):
                self.in_address_label = True
            elif tag == 'p' and 'address-text' in attrs_dict.get('class', ''):
                self.in_address_text = True

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
            
        if self.in_address_label:
            self.current_label = data
        elif self.in_address_text:
            self.current_address = data

    def handle_endtag(self, tag):
        if tag == 'div' and self.in_address_card and not self.in_address_content:
            # End of address card
            if self.current_label and self.current_address:
                address_info = {
                    'label': self.current_label,
                    'address': self.current_address,
                    'is_default': self.in_default_badge
                }
                self.addresses.append(address_info)
                if self.in_default_badge:
                    self.default_address = address_info
            self.in_address_card = False
            self.in_default_badge = False
            
        if tag == 'div' and self.in_address_content:
            self.in_address_content = False
        if tag == 'h3':
            self.in_address_label = False
        if tag == 'p':
            self.in_address_text = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_navigated_to_addresses": False,
            "cp2_address_updated": False,
            "cp3_correct_address": False,
            "cp4_is_default": False,
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

    def parse_addresses(self, html_content: str) -> Tuple[List[Dict], Optional[Dict]]:
        """Parse HTML to extract address information."""
        parser = AddressHTMLParser()
        parser.feed(html_content)
        return parser.addresses, parser.default_address

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Successfully navigated to addresses page"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML file found")
                return False

            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()

            # Check if on the addresses page
            if 'address-management-page' in html_content or 'My Saved Addresses' in html_content:
                self.details['navigated_to_addresses'] = True
                return True
            else:
                self.issues.append("Did not navigate to addresses page")
                self.details['navigated_to_addresses'] = False
                return False
        except Exception as e:
            self.issues.append(f"Error checking navigation: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Address was modified/saved"""
        try:
            trajectory = self.get_trajectory()
            if not trajectory:
                self.issues.append("No trajectory data found")
                return False

            # Check if there were FILL actions and a Save Changes action
            fill_actions = [step for step in trajectory if step.get('action', {}).get('action_type') == 'FILL']
            save_actions = [step for step in trajectory if 
                          step.get('action', {}).get('action_type') == 'CLICK' and
                          'Save Changes' in str(step.get('action', {}).get('parameters', {}))]

            if fill_actions and save_actions:
                self.details['form_filled'] = len(fill_actions)
                self.details['save_clicked'] = True
                return True
            else:
                if not fill_actions:
                    self.issues.append("No address fields were filled")
                if not save_actions:
                    self.issues.append("Save Changes button was not clicked")
                return False
        except Exception as e:
            self.issues.append(f"Error checking address modification: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Address contains correct values"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML file found")
                return False

            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()

            addresses, default_address = self.parse_addresses(html_content)
            
            if not addresses:
                # Try simpler regex search
                match = re.search(r'123 Main Street[^<]*San Francisco[^<]*CA[^<]*94102', html_content)
                if match:
                    self.details['address_found'] = match.group(0)
                    return True
                else:
                    self.issues.append("Target address '123 Main Street, San Francisco, CA 94102' not found in HTML")
                    return False

            # Check if any address matches the target
            target_address = "123 Main Street, San Francisco, CA 94102"
            for addr in addresses:
                if target_address in addr['address']:
                    self.details['address_match'] = addr['address']
                    self.details['address_label'] = addr['label']
                    return True

            self.issues.append(f"Address '123 Main Street, San Francisco, CA 94102' not found. Found addresses: {[a['address'] for a in addresses]}")
            return False
        except Exception as e:
            self.issues.append(f"Error checking address correctness: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Updated address is marked as default"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML file found")
                return False

            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()

            addresses, default_address = self.parse_addresses(html_content)
            
            if not default_address:
                # Try regex search for default badge near the target address
                default_pattern = r'default-badge-top[^>]*>.*?DEFAULT.*?123 Main Street[^<]*San Francisco[^<]*CA[^<]*94102'
                if re.search(default_pattern, html_content, re.IGNORECASE | re.DOTALL):
                    self.details['is_default'] = True
                    return True
                else:
                    self.issues.append("Updated address is not marked as default")
                    return False

            # Check if the default address is the updated one
            target_address = "123 Main Street, San Francisco, CA 94102"
            if target_address in default_address['address']:
                self.details['default_address'] = default_address['address']
                self.details['default_label'] = default_address['label']
                return True
            else:
                self.issues.append(f"Default address is '{default_address['address']}', not the updated address")
                return False
        except Exception as e:
            self.issues.append(f"Error checking default status: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_navigated_to_addresses'] = self.check_checkpoint_1()
            self.checkpoints['cp2_address_updated'] = self.check_checkpoint_2()
            self.checkpoints['cp3_correct_address'] = self.check_checkpoint_3()
            self.checkpoints['cp4_is_default'] = self.check_checkpoint_4()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 3,
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
                'query_id': 3,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python food_query3.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
