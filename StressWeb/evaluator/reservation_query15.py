#!/usr/bin/env python3
# Evaluator for reservation Query 15

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional
from html.parser import HTMLParser


class ReservationHTMLParser(HTMLParser):
    """Parse HTML to extract reservation-related information."""

    def __init__(self):
        super().__init__()
        self.in_reservation_form = False
        self.in_confirmation = False
        self.in_success_message = False
        self.current_tag = None
        self.current_attrs = {}
        self.reservation_data = {}
        self.restaurant_name = None
        self.party_size = None
        self.date = None
        self.time = None
        self.occasion = None
        self.special_request = None
        self.success_message = None
        self.text_buffer = []
        
    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        self.current_attrs = dict(attrs)
        
        # Check for form inputs with reservation data
        if tag == 'input' or tag == 'select':
            name = self.current_attrs.get('name', '')
            value = self.current_attrs.get('value', '')
            
            if 'party' in name.lower() or 'guests' in name.lower():
                self.party_size = value
            elif 'date' in name.lower():
                self.date = value
            elif 'time' in name.lower():
                self.time = value
            elif 'occasion' in name.lower():
                self.occasion = value
                
    def handle_data(self, data):
        text = data.strip()
        if text:
            self.text_buffer.append(text)
            
            # Look for success/confirmation messages
            if any(keyword in text.lower() for keyword in ['confirmed', 'success', 'reservation complete', 'booking confirmed']):
                self.success_message = text
                self.in_confirmation = True
                
            # Look for restaurant name
            if 'lotus' in text.lower() and 'dragon' in text.lower():
                self.restaurant_name = text
                
    def get_results(self):
        return {
            'restaurant_name': self.restaurant_name,
            'party_size': self.party_size,
            'date': self.date,
            'time': self.time,
            'occasion': self.occasion,
            'success_message': self.success_message,
            'in_confirmation': self.in_confirmation,
        }


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints for the reservation task
        self.checkpoints = {
            "cp1_found_restaurant": False,
            "cp2_accessed_booking_form": False,
            "cp3_set_party_size_6": False,
            "cp4_set_date_feb25_2026": False,
            "cp5_set_time_8pm": False,
            "cp6_selected_birthday_occasion": False,
            "cp7_added_special_request": False,
            "cp8_submitted_reservation": False,
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

    def check_html_content(self, html_path: Path, patterns: List[str]) -> bool:
        """Check if HTML file contains any of the given patterns."""
        if not html_path.exists():
            return False
        
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                content = f.read()
                return any(pattern.lower() in content.lower() for pattern in patterns)
        except Exception:
            return False

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Found Lotus & Dragon restaurant"""
        # Check trajectory for search actions
        trajectory = self.get_trajectory()
        
        # Look for search actions with Lotus & Dragon
        for entry in trajectory:
            action = entry.get('action', {})
            params = action.get('parameters', {})
            
            # Check if filled search with Lotus & Dragon
            if action.get('action_type') == 'FILL':
                text = params.get('text', '')
                if 'lotus' in text.lower() and 'dragon' in text.lower():
                    self.details['searched_for'] = text
                    
                    # Check if search was executed (next step should be clicking search/Let's go)
                    step_num = entry.get('step', 0)
                    if step_num < len(trajectory) - 1:
                        next_action = trajectory[step_num + 1].get('action', {})
                        if next_action.get('action_type') == 'CLICK':
                            # Check intermediate steps for Lotus & Dragon in results
                            step_files = self.find_step_html(f"step_{step_num + 1}_*")
                            if step_files:
                                if self.check_html_content(step_files[0], ['Lotus &amp; Dragon', 'Lotus & Dragon']):
                                    return True
        
        return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Accessed booking/reservation form"""
        trajectory = self.get_trajectory()
        
        # Look for clicks on "Book Now" button
        book_now_clicked = False
        for entry in trajectory:
            action = entry.get('action', {})
            if action.get('action_type') == 'CLICK':
                selector = action.get('parameters', {}).get('selector', '')
                if 'book now' in selector.lower():
                    book_now_clicked = True
                    self.details['book_now_clicked'] = True
                    break
        
        if not book_now_clicked:
            self.issues.append("Never clicked 'Book Now' button to access reservation form")
            return False
            
        # The agent clicked Book Now many times but got stuck in a loop
        # This suggests the button click doesn't lead to the booking form
        # Check if agent got stuck in loop
        if book_now_clicked:
            book_clicks = sum(1 for entry in trajectory if entry.get('action', {}).get('action_type') == 'CLICK' 
                            and 'book now' in entry.get('action', {}).get('parameters', {}).get('selector', '').lower())
            if book_clicks > 10:
                self.issues.append(f"Agent clicked 'Book Now' {book_clicks} times, indicating a loop/non-functional button")
                return False
        
        return book_now_clicked

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Set party size to 6"""
        # Check all intermediate steps for party size input
        all_steps = self.find_step_html("step_*")
        
        for step_file in all_steps:
            try:
                with open(step_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Look for party size input with value 6
                    if re.search(r'(party|guest|people).*value="6"', content, re.IGNORECASE):
                        return True
                    if re.search(r'value="6".*?(party|guest|people)', content, re.IGNORECASE):
                        return True
            except Exception:
                pass
        
        self.issues.append("Party size was never set to 6")
        return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Set date to February 25, 2026"""
        all_steps = self.find_step_html("step_*")
        
        for step_file in all_steps:
            try:
                with open(step_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Look for date input with Feb 25, 2026 (various formats)
                    patterns = [
                        r'value="2026-02-25"',
                        r'value="02/25/2026"',
                        r'value="25/02/2026"',
                        r'February 25,? 2026',
                        r'Feb 25,? 2026'
                    ]
                    if any(re.search(pattern, content, re.IGNORECASE) for pattern in patterns):
                        return True
            except Exception:
                pass
        
        self.issues.append("Date was never set to February 25, 2026")
        return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Set time to 8:00 PM"""
        all_steps = self.find_step_html("step_*")
        
        for step_file in all_steps:
            try:
                with open(step_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Look for time input with 8:00 PM
                    patterns = [
                        r'value="20:00"',
                        r'value="8:00 PM"',
                        r'value="08:00 PM"',
                        r'8:00.*?PM',
                        r'20:00'
                    ]
                    if any(re.search(pattern, content, re.IGNORECASE) for pattern in patterns):
                        return True
            except Exception:
                pass
        
        self.issues.append("Time was never set to 8:00 PM")
        return False

    def check_checkpoint_6(self) -> bool:
        """Checkpoint 6: Selected 'Birthday' as occasion type"""
        all_steps = self.find_step_html("step_*")
        
        for step_file in all_steps:
            try:
                with open(step_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Look for occasion selection with Birthday
                    patterns = [
                        r'occasion.*?value="[Bb]irthday"',
                        r'value="[Bb]irthday".*?occasion',
                        r'selected.*?[Bb]irthday',
                        r'[Bb]irthday.*?selected'
                    ]
                    if any(re.search(pattern, content, re.IGNORECASE) for pattern in patterns):
                        return True
            except Exception:
                pass
        
        self.issues.append("Occasion type 'Birthday' was never selected")
        return False

    def check_checkpoint_7(self) -> bool:
        """Checkpoint 7: Added special request 'Birthday dessert needed'"""
        trajectory = self.get_trajectory()
        
        # Check trajectory for fill action with special request
        for entry in trajectory:
            action = entry.get('action', {})
            if action.get('action_type') == 'FILL':
                text = action.get('parameters', {}).get('text', '')
                selector = action.get('parameters', {}).get('selector', '')
                
                # Check if this is a special request field with the right text
                if 'birthday dessert needed' in text.lower():
                    if any(keyword in selector.lower() for keyword in ['request', 'note', 'comment', 'special']):
                        return True
        
        # Also check HTML for textarea/input with the special request
        all_steps = self.find_step_html("step_*")
        for step_file in all_steps:
            try:
                with open(step_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'birthday dessert needed' in content.lower():
                        return True
            except Exception:
                pass
        
        self.issues.append("Special request 'Birthday dessert needed' was never added")
        return False

    def check_checkpoint_8(self) -> bool:
        """Checkpoint 8: Submitted the reservation successfully"""
        trajectory = self.get_trajectory()
        
        # Check if there was a submit/confirm action
        submit_clicked = False
        for entry in trajectory:
            action = entry.get('action', {})
            if action.get('action_type') == 'CLICK':
                selector = action.get('parameters', {}).get('selector', '')
                if any(keyword in selector.lower() for keyword in ['submit', 'confirm', 'complete', 'reserve']):
                    submit_clicked = True
                    break
        
        if not submit_clicked:
            self.issues.append("Never submitted/confirmed the reservation")
            return False
        
        # Check for confirmation message in final or late steps
        final_html = self.find_final_html()
        if final_html:
            try:
                with open(final_html, 'r', encoding='utf-8') as f:
                    content = f.read()
                    confirmation_patterns = [
                        r'reservation.*?confirmed',
                        r'booking.*?confirmed',
                        r'successfully.*?reserved',
                        r'confirmation',
                        r'reservation.*?complete'
                    ]
                    if any(re.search(pattern, content, re.IGNORECASE) for pattern in confirmation_patterns):
                        return True
            except Exception:
                pass
        
        self.issues.append("No reservation confirmation found in final state")
        return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Check if agent failed due to loop
            if not result_data.get('success', False):
                reason = result_data.get('message', '')
                if 'loop' in reason.lower() or 'stuck' in reason.lower():
                    self.issues.append("Agent got stuck in an endless loop searching for the restaurant")
                    self.details['failure_reason'] = reason

            # Run checkpoint validations
            self.checkpoints['cp1_found_restaurant'] = self.check_checkpoint_1()
            self.checkpoints['cp2_accessed_booking_form'] = self.check_checkpoint_2()
            self.checkpoints['cp3_set_party_size_6'] = self.check_checkpoint_3()
            self.checkpoints['cp4_set_date_feb25_2026'] = self.check_checkpoint_4()
            self.checkpoints['cp5_set_time_8pm'] = self.check_checkpoint_5()
            self.checkpoints['cp6_selected_birthday_occasion'] = self.check_checkpoint_6()
            self.checkpoints['cp7_added_special_request'] = self.check_checkpoint_7()
            self.checkpoints['cp8_submitted_reservation'] = self.check_checkpoint_8()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 15,
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
                'query_id': 15,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python reservation_query15.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
