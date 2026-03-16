#!/usr/bin/env python3
# Evaluator for reservation Query 14

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class ReservationHTMLParser(HTMLParser):
    """Parse HTML to extract reservation information."""

    def __init__(self):
        super().__init__()
        self.in_party_size = False
        self.in_summary_value = False
        self.in_success_title = False
        self.in_reservation_meta = False
        self.party_size = None
        self.success_message = None
        self.reservation_confirmed = False
        self.current_label = None
        self.capture_next = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check for success page title
        if tag == 'h1' and attrs_dict.get('class') == 'success-title':
            self.in_success_title = True
        
        # Check for summary label
        if tag == 'span' and 'summary-label' in attrs_dict.get('class', ''):
            self.capture_next = True
        
        # Check for summary value
        if tag == 'span' and 'summary-value' in attrs_dict.get('class', ''):
            self.in_summary_value = True
        
        # Check for reservation meta
        if tag == 'p' and 'reservation-meta' in attrs_dict.get('class', ''):
            self.in_reservation_meta = True

    def handle_data(self, data):
        text = data.strip()
        
        if self.in_success_title and text:
            self.success_message = text
            self.in_success_title = False
        
        if self.capture_next and text:
            self.current_label = text
            self.capture_next = False
        
        if self.in_summary_value and text:
            if self.current_label and 'Party Size' in self.current_label:
                self.party_size = text
            self.in_summary_value = False
            self.current_label = None
        
        # Extract party size from reservation card (e.g., "4 guests")
        if self.in_reservation_meta and 'guest' in text.lower():
            match = re.search(r'(\d+)\s*guests?', text, re.IGNORECASE)
            if match:
                self.party_size = f"{match.group(1)} guests"

    def handle_endtag(self, tag):
        if tag == 'p' and self.in_reservation_meta:
            self.in_reservation_meta = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints
        self.checkpoints = {
            "cp1_navigated_to_reservations": False,
            "cp2_accessed_reservation_details": False,
            "cp3_opened_modification_form": False,
            "cp4_saved_changes": False,
            "cp5_confirmation_displayed": False,
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
        """Checkpoint 1: Navigated to reservations page"""
        try:
            trajectory = self.get_trajectory()
            
            # Check if agent navigated to /reservations
            for step in trajectory:
                page_info = step.get('page_info', {})
                url = page_info.get('url', '')
                if '/reservations' in url:
                    self.details['reservations_page_reached'] = True
                    return True
            
            self.issues.append("Agent did not navigate to the reservations page")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking navigation: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Accessed reservation details"""
        try:
            trajectory = self.get_trajectory()
            
            # Check if agent expanded/clicked on a reservation to see details
            for step in trajectory:
                action = step.get('action', {})
                if action:
                    selector = action.get('parameters', {}).get('selector', '')
                    # Check for clicking on restaurant name or reservation card
                    if 'Garden Bistro' in selector or 'Modify' in selector:
                        self.details['reservation_details_accessed'] = True
                        return True
            
            self.issues.append("Agent did not access reservation details")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking reservation access: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Opened modification form"""
        try:
            trajectory = self.get_trajectory()
            
            # Check if agent navigated to modification page
            for step in trajectory:
                page_info = step.get('page_info', {})
                url = page_info.get('url', '')
                if '/modify' in url:
                    self.details['modification_form_opened'] = True
                    
                    # Also check if we can find the modification form in intermediate steps
                    step_num = step.get('step_num', 0)
                    step_files = self.find_step_html(f"step_{step_num}_*")
                    if step_files:
                        with open(step_files[0], 'r', encoding='utf-8') as f:
                            html_content = f.read()
                            if 'party-size-btn' in html_content or 'Party Size' in html_content:
                                self.details['modification_form_contains_party_size'] = True
                    
                    return True
            
            self.issues.append("Agent did not open the modification form")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking modification form: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Saved changes (clicked Save Changes button)"""
        try:
            trajectory = self.get_trajectory()
            
            # Check if agent clicked the Save Changes button
            for step in trajectory:
                action = step.get('action', {})
                if action.get('action_type') == 'CLICK':
                    selector = action.get('parameters', {}).get('selector', '')
                    if 'Save Changes' in selector:
                        result = action.get('result', {}) if 'result' in action else step.get('result', {})
                        if result and result.get('success'):
                            self.details['save_changes_clicked'] = True
                            return True
            
            self.issues.append("Agent did not click the Save Changes button")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking save action: {str(e)}")
            return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Confirmation page displayed with correct party size"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML found")
                return False
            
            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            parser = ReservationHTMLParser()
            parser.feed(html_content)
            
            # Check for success message
            if parser.success_message and 'Modified Successfully' in parser.success_message:
                self.details['success_message_found'] = parser.success_message
            elif parser.success_message:
                self.details['unexpected_message'] = parser.success_message
                self.issues.append(f"Unexpected success message: {parser.success_message}")
                return False
            else:
                # Check if we're on a success/confirmation page
                if 'modification-success-page' not in html_content and 'success' not in html_content.lower():
                    self.issues.append("No confirmation page displayed")
                    return False
            
            # Check party size in final state
            if parser.party_size:
                self.details['final_party_size'] = parser.party_size
                
                # The task asks to change from 2 to 4 guests
                # Extract number from party size
                match = re.search(r'(\d+)', parser.party_size)
                if match:
                    party_num = int(match.group(1))
                    if party_num == 4:
                        self.details['party_size_correct'] = True
                        return True
                    else:
                        self.issues.append(f"Party size is {party_num}, expected 4")
                        return False
            
            # Check for confirmation text in HTML
            if '4 guests' in html_content or '4 Guests' in html_content:
                self.details['party_size_in_html'] = True
                return True
            
            self.issues.append("Party size of 4 guests not found in confirmation")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking confirmation: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_navigated_to_reservations'] = self.check_checkpoint_1()
            self.checkpoints['cp2_accessed_reservation_details'] = self.check_checkpoint_2()
            self.checkpoints['cp3_opened_modification_form'] = self.check_checkpoint_3()
            self.checkpoints['cp4_saved_changes'] = self.check_checkpoint_4()
            self.checkpoints['cp5_confirmation_displayed'] = self.check_checkpoint_5()

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
        print("Usage: python reservation_query14.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
