#!/usr/bin/env python3
# Evaluator for transportation Query 12

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class SavedPlaceParser(HTMLParser):
    """Parse HTML to extract saved places information."""

    def __init__(self):
        super().__init__()
        self.in_saved_place = False
        self.in_label = False
        self.in_address = False
        self.current_label = ""
        self.current_address = ""
        self.saved_places = []
        self.recording_data = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check for saved place card
        if 'class' in attrs_dict:
            class_value = attrs_dict['class']
            if 'saved-place-card' in class_value or 'place-card' in class_value:
                self.in_saved_place = True
            if 'place-label' in class_value or 'saved-label' in class_value:
                self.in_label = True
            if 'place-address' in class_value or 'saved-address' in class_value:
                self.in_address = True

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
            
        if self.in_label:
            self.current_label = data
        elif self.in_address:
            self.current_address = data

    def handle_endtag(self, tag):
        if self.in_label:
            self.in_label = False
        if self.in_address:
            self.in_address = False
        if self.in_saved_place and self.current_label and self.current_address:
            self.saved_places.append({
                'label': self.current_label,
                'address': self.current_address
            })
            self.current_label = ""
            self.current_address = ""
            self.in_saved_place = False


class TripConfirmationParser(HTMLParser):
    """Parse HTML to check for trip confirmation page."""

    def __init__(self):
        super().__init__()
        self.on_confirmation_page = False
        self.trip_confirmed = False
        self.in_title = False
        self.title_text = ""

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        if 'class' in attrs_dict:
            class_value = attrs_dict['class']
            if 'confirmation' in class_value.lower() or 'success' in class_value.lower():
                if 'title' in class_value or 'heading' in class_value:
                    self.in_title = True

    def handle_data(self, data):
        data = data.strip()
        if self.in_title:
            self.title_text += data
            if 'confirmed' in data.lower() or 'booked' in data.lower():
                self.trip_confirmed = True

    def handle_endtag(self, tag):
        if self.in_title:
            self.in_title = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_saved_place_added": False,
            "cp2_correct_address": False,
            "cp3_correct_label": False,
            "cp4_ride_requested_to_shopping": False,
            "cp5_trip_confirmed": False,
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

    def check_saved_place_in_html(self, html_path: Path) -> Tuple[bool, Optional[str], Optional[str]]:
        """Check if saved place exists in HTML and return label and address."""
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Look for Shopping label and address in saved places
            parser = SavedPlaceParser()
            parser.feed(content)
            
            for place in parser.saved_places:
                if 'shopping' in place['label'].lower():
                    return True, place['label'], place['address']
            
            # Fallback: search for text patterns
            if re.search(r'Shopping', content, re.IGNORECASE):
                address_match = re.search(r'123\s*Market\s*Street[^<]*San\s*Francisco[^<]*94102', content, re.IGNORECASE)
                if address_match:
                    return True, "Shopping", address_match.group(0)
                    
        except Exception as e:
            self.issues.append(f"Error parsing saved place HTML: {e}")
            
        return False, None, None

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Saved place added"""
        # Check across all steps to see if saved place was added
        all_steps = sorted(self.result_dir.glob("step_*_raw.html"))
        
        for step_file in all_steps:
            found, label, address = self.check_saved_place_in_html(step_file)
            if found:
                self.details['saved_place_found_at'] = step_file.name
                self.details['saved_place_label'] = label
                self.details['saved_place_address'] = address
                return True
        
        self.issues.append("Saved place with label 'Shopping' not found in any step")
        return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Correct address (123 Market Street, San Francisco, CA 94102)"""
        if 'saved_place_address' not in self.details:
            self.issues.append("Cannot verify address - saved place not found")
            return False
        
        address = self.details['saved_place_address']
        # Check if address contains key components
        has_street = '123' in address and 'market' in address.lower() and 'street' in address.lower()
        has_city = 'san francisco' in address.lower()
        has_zip = '94102' in address
        
        if has_street and has_city and has_zip:
            return True
        else:
            missing = []
            if not has_street:
                missing.append("street number/name")
            if not has_city:
                missing.append("city")
            if not has_zip:
                missing.append("zip code")
            self.issues.append(f"Address incomplete or incorrect. Missing: {', '.join(missing)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Correct label (Shopping)"""
        if 'saved_place_label' not in self.details:
            self.issues.append("Cannot verify label - saved place not found")
            return False
        
        label = self.details['saved_place_label']
        if 'shopping' in label.lower():
            return True
        else:
            self.issues.append(f"Label incorrect. Expected 'Shopping', found '{label}'")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Ride requested to Shopping destination"""
        trajectory = self.get_trajectory()
        
        # Look for actions related to ride request with Shopping destination
        for entry in trajectory:
            action = entry.get('action', {})
            reasoning = action.get('reasoning', '')
            
            # Check if reasoning mentions requesting ride to Shopping
            if 'shopping' in reasoning.lower() and ('request' in reasoning.lower() or 'ride' in reasoning.lower()):
                self.details['ride_request_found'] = True
                
                # Check HTML for ride request form or confirmation
                step = entry.get('step', 0)
                step_files = self.find_step_html(f"step_{step}_*")
                
                for step_file in step_files:
                    try:
                        with open(step_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                        # Look for destination set to Shopping
                        if re.search(r'Shopping.*destination', content, re.IGNORECASE | re.DOTALL) or \
                           re.search(r'destination.*Shopping', content, re.IGNORECASE | re.DOTALL) or \
                           re.search(r'to:\s*Shopping', content, re.IGNORECASE):
                            return True
                    except Exception as e:
                        pass
        
        self.issues.append("No ride request to Shopping destination found")
        return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Trip confirmed"""
        # Check final HTML and later steps for confirmation
        all_steps = sorted(self.result_dir.glob("step_*_raw.html"))
        
        # Check last 10 steps for confirmation
        for step_file in all_steps[-10:]:
            try:
                with open(step_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                parser = TripConfirmationParser()
                parser.feed(content)
                
                if parser.trip_confirmed:
                    self.details['trip_confirmed_at'] = step_file.name
                    return True
                
                # Fallback: search for confirmation keywords
                if re.search(r'(trip|ride)\s+(confirmed|booked|scheduled)', content, re.IGNORECASE) or \
                   re.search(r'confirmation', content, re.IGNORECASE):
                    # Additional verification - check for success indicators
                    if re.search(r'success|confirmed|complete', content, re.IGNORECASE):
                        self.details['trip_confirmed_at'] = step_file.name
                        return True
                        
            except Exception as e:
                pass
        
        self.issues.append("Trip confirmation not found")
        return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_saved_place_added'] = self.check_checkpoint_1()
            self.checkpoints['cp2_correct_address'] = self.check_checkpoint_2()
            self.checkpoints['cp3_correct_label'] = self.check_checkpoint_3()
            self.checkpoints['cp4_ride_requested_to_shopping'] = self.check_checkpoint_4()
            self.checkpoints['cp5_trip_confirmed'] = self.check_checkpoint_5()

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
        print("Usage: python transportation_query12.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
