#!/usr/bin/env python3
# Evaluator for reservation Query 12

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class ReservationSuccessParser(HTMLParser):
    """Parse HTML to extract reservation confirmation details."""

    def __init__(self):
        super().__init__()
        self.in_booking_number = False
        self.in_restaurant_name = False
        self.in_detail_value = False
        self.current_detail_label = None
        self.booking_number = None
        self.restaurant_name = None
        self.reservation_date = None
        self.reservation_time = None
        self.party_size = None
        self.is_confirmed = False
        self.in_booking_status = False
        self.current_section_title = None
        self.in_section_title = False
        self.in_detail_label = False
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        class_name = attrs_dict.get('class', '')
        
        if 'booking-number' in class_name:
            self.in_booking_number = True
        elif 'booking-status' in class_name:
            self.in_booking_status = True
        elif 'section-title' in class_name:
            self.in_section_title = True
        elif 'detail-label' in class_name:
            self.in_detail_label = True
        elif 'detail-value' in class_name:
            self.in_detail_value = True
        elif 'success-title' in class_name:
            self.in_booking_number = True  # To catch "Reservation Confirmed"

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
            
        if self.in_booking_number and 'BK' in data:
            self.booking_number = data.replace('Booking #', '').strip()
            self.in_booking_number = False
        elif self.in_booking_status and 'Confirmed' in data:
            self.is_confirmed = True
            self.in_booking_status = False
        elif self.in_section_title:
            self.current_section_title = data
            self.in_section_title = False
        elif self.in_detail_label:
            self.current_detail_label = data
            self.in_detail_label = False
        elif self.in_detail_value and self.current_detail_label:
            if self.current_section_title == 'Restaurant':
                if self.current_detail_label == 'Name':
                    self.restaurant_name = data
            elif self.current_section_title == 'Reservation Details':
                if '📅' in self.current_detail_label or 'Date' in self.current_detail_label:
                    self.reservation_date = data
                elif '🕐' in self.current_detail_label or 'Time' in self.current_detail_label:
                    self.reservation_time = data
                elif '👥' in self.current_detail_label or 'Party Size' in self.current_detail_label:
                    self.party_size = data
            self.current_detail_label = None
            self.in_detail_value = False
        elif 'Reservation Confirmed' in data:
            self.is_confirmed = True


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_found_restaurant": False,
            "cp2_selected_correct_date": False,
            "cp3_selected_correct_time": False,
            "cp4_selected_correct_party_size": False,
            "cp5_reservation_confirmed": False,
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
        """Checkpoint 1: Found Bamboo Modern Kitchen restaurant"""
        try:
            trajectory = self.get_trajectory()
            
            # Check if user navigated to or clicked on Bamboo Modern Kitchen
            for entry in trajectory:
                action = entry.get('action', {})
                reasoning = action.get('reasoning', '')
                params = action.get('parameters', {})
                selector = params.get('selector', '')
                
                if 'Bamboo Modern Kitchen' in reasoning or 'Bamboo Modern Kitchen' in selector:
                    self.details['found_restaurant'] = 'Bamboo Modern Kitchen'
                    return True
                    
            # Also check final HTML for restaurant name
            final_html = self.find_final_html()
            if final_html:
                with open(final_html, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'Bamboo Modern Kitchen' in content:
                        self.details['found_restaurant'] = 'Bamboo Modern Kitchen'
                        return True
            
            self.issues.append("Did not find or navigate to Bamboo Modern Kitchen restaurant")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking restaurant: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Selected correct date (February 15, 2026)"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML found")
                return False
            
            with open(final_html, 'r', encoding='utf-8') as f:
                content = f.read()
            
            parser = ReservationSuccessParser()
            parser.feed(content)
            
            if parser.reservation_date:
                self.details['reservation_date'] = parser.reservation_date
                # Check for February 15, 2026 or 2026/02/15
                if '2026/02/15' in parser.reservation_date or \
                   ('February 15' in parser.reservation_date and '2026' in parser.reservation_date) or \
                   ('15' in parser.reservation_date and 'February' in parser.reservation_date and '2026' in parser.reservation_date):
                    return True
                else:
                    self.issues.append(f"Incorrect date: {parser.reservation_date} (expected February 15, 2026)")
                    return False
            else:
                # Check in intermediate steps for the date selection
                step_files = self.find_step_html("step_*")
                for step_file in step_files:
                    with open(step_file, 'r', encoding='utf-8') as f:
                        step_content = f.read()
                        if '2026/02/15' in step_content:
                            self.details['reservation_date'] = '2026/02/15'
                            return True
                
                self.issues.append("Reservation date not found in confirmation")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking date: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Selected correct time (7:00 PM)"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                return False
            
            with open(final_html, 'r', encoding='utf-8') as f:
                content = f.read()
            
            parser = ReservationSuccessParser()
            parser.feed(content)
            
            if parser.reservation_time:
                self.details['reservation_time'] = parser.reservation_time
                # Check for 7:00 PM or 7:00 pm
                if '7:00' in parser.reservation_time and 'pm' in parser.reservation_time.lower():
                    return True
                else:
                    self.issues.append(f"Incorrect time: {parser.reservation_time} (expected 7:00 PM)")
                    return False
            else:
                self.issues.append("Reservation time not found in confirmation")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking time: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Selected correct party size (2 people)"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                return False
            
            with open(final_html, 'r', encoding='utf-8') as f:
                content = f.read()
            
            parser = ReservationSuccessParser()
            parser.feed(content)
            
            if parser.party_size:
                self.details['party_size'] = parser.party_size
                # Check for 2 guests or 2 people
                if '2' in parser.party_size and ('guest' in parser.party_size.lower() or 'people' in parser.party_size.lower()):
                    return True
                else:
                    self.issues.append(f"Incorrect party size: {parser.party_size} (expected 2 people)")
                    return False
            else:
                self.issues.append("Party size not found in confirmation")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking party size: {str(e)}")
            return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Reservation successfully confirmed"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML found")
                return False
            
            with open(final_html, 'r', encoding='utf-8') as f:
                content = f.read()
            
            parser = ReservationSuccessParser()
            parser.feed(content)
            
            if parser.is_confirmed and parser.booking_number:
                self.details['booking_number'] = parser.booking_number
                self.details['confirmed'] = True
                return True
            elif 'Reservation Confirmed' in content and 'BK' in content:
                # Fallback: check for confirmation text and booking number pattern
                booking_match = re.search(r'BK\d+', content)
                if booking_match:
                    self.details['booking_number'] = booking_match.group(0)
                    self.details['confirmed'] = True
                    return True
            
            self.issues.append("Reservation not confirmed or booking number not found")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking confirmation: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_found_restaurant'] = self.check_checkpoint_1()
            self.checkpoints['cp2_selected_correct_date'] = self.check_checkpoint_2()
            self.checkpoints['cp3_selected_correct_time'] = self.check_checkpoint_3()
            self.checkpoints['cp4_selected_correct_party_size'] = self.check_checkpoint_4()
            self.checkpoints['cp5_reservation_confirmed'] = self.check_checkpoint_5()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 12,
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
                'query_id': 12,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python reservation_query12.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
