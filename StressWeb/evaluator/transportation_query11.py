#!/usr/bin/env python3
# Evaluator for transportation Query 11

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class SavedPlaceHTMLParser(HTMLParser):
    """Parse HTML to extract saved places information."""

    def __init__(self):
        super().__init__()
        self.in_saved_addresses = False
        self.in_saved_place_item = False
        self.saved_places = []
        self.current_place = {}
        self.current_tag = None
        self.current_attrs = []
        self.capture_text = False
        self.text_buffer = []
        
    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        self.current_attrs = dict(attrs)
        
        # Look for saved addresses section
        if tag == 'div':
            class_attr = self.current_attrs.get('class', '')
            # Look for saved place items or address cards
            if 'saved-place' in class_attr.lower() or 'address-card' in class_attr.lower() or 'place-item' in class_attr.lower():
                self.in_saved_place_item = True
                self.current_place = {}
            
        # Capture icon classes
        if tag == 'i' and self.in_saved_place_item:
            class_attr = self.current_attrs.get('class', '')
            if 'fa-shopping-cart' in class_attr or 'shopping' in class_attr.lower():
                self.current_place['icon'] = 'shopping-cart'
            elif 'fa-' in class_attr:
                # Extract icon name
                for cls in class_attr.split():
                    if cls.startswith('fa-') and cls != 'fa-solid' and cls != 'fa-regular':
                        self.current_place['icon'] = cls.replace('fa-', '')
        
        # Capture text content for labels and addresses
        if self.in_saved_place_item and tag in ['span', 'div', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'label']:
            self.capture_text = True

    def handle_data(self, data):
        text = data.strip()
        if text and self.in_saved_place_item:
            if self.capture_text:
                self.text_buffer.append(text)
            
            # Look for the label "Shopping"
            if text == 'Shopping' or text.lower() == 'shopping':
                self.current_place['label'] = 'Shopping'
            
            # Look for the address
            if '123 market street' in text.lower() or '123 market st' in text.lower():
                self.current_place['address'] = text
            elif 'san francisco' in text.lower() and '94102' in text:
                self.current_place['address'] = text

    def handle_endtag(self, tag):
        if tag == 'div' and self.in_saved_place_item:
            # Check if we have collected enough info about a saved place
            if self.current_place:
                # Also check text buffer for address and label
                full_text = ' '.join(self.text_buffer)
                if not self.current_place.get('address'):
                    if '123 market' in full_text.lower() and 'san francisco' in full_text.lower():
                        self.current_place['address'] = full_text
                if not self.current_place.get('label') and 'shopping' in full_text.lower():
                    self.current_place['label'] = 'Shopping'
                
                self.saved_places.append(self.current_place)
            self.in_saved_place_item = False
            self.current_place = {}
            self.text_buffer = []
        
        if tag in ['span', 'div', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'label']:
            self.capture_text = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_navigated_to_saved_places": False,
            "cp2_saved_place_created": False,
            "cp3_correct_address": False,
            "cp4_correct_label": False,
            "cp5_correct_icon": False,
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
        """Checkpoint 1: Navigated to Account/Saved Places section"""
        try:
            trajectory = self.get_trajectory()
            
            # Check if agent clicked on Account or navigated to saved places
            account_clicked = False
            for entry in trajectory:
                if 'action' in entry:
                    action = entry['action']
                    if action.get('action_type') == 'CLICK':
                        selector = action.get('parameters', {}).get('selector', '').lower()
                        if 'account' in selector or 'saved' in selector or 'preferences' in selector:
                            account_clicked = True
                            break
            
            if account_clicked:
                self.details['navigation'] = 'Agent attempted to navigate to Account/Saved Places'
                return True
            else:
                self.issues.append('Agent did not navigate to Account section')
                return False
                
        except Exception as e:
            self.issues.append(f'Error checking navigation: {str(e)}')
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: A saved place was created"""
        try:
            # Check all step HTML files for saved places
            all_html_files = sorted(self.result_dir.glob("step_*_raw.html"))
            
            for html_file in all_html_files:
                with open(html_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    
                # Look for indicators of saved places in HTML
                if 'shopping' in html_content.lower() and ('123 market' in html_content.lower() or '123 market street' in html_content.lower()):
                    parser = SavedPlaceHTMLParser()
                    parser.feed(html_content)
                    
                    if len(parser.saved_places) > 0:
                        self.details['saved_places_found'] = len(parser.saved_places)
                        return True
            
            # Also check final HTML
            final_html = self.find_final_html()
            if final_html:
                with open(final_html, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    
                # Look for saved places count changing from 0
                if re.search(r'saved\s+addresses[:\s]*[1-9]\d*\s+address', html_content, re.IGNORECASE):
                    self.details['saved_address_count'] = 'Changed from 0 to 1+'
                    return True
                
                # Look for presence of shopping label
                if 'shopping' in html_content.lower() and '123 market' in html_content.lower():
                    return True
                    
            self.issues.append('No saved place was created')
            return False
            
        except Exception as e:
            self.issues.append(f'Error checking saved place creation: {str(e)}')
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Address is correct (123 Market Street, San Francisco, CA 94102)"""
        try:
            # Check all HTML files for the address
            all_html_files = sorted(self.result_dir.glob("step_*_raw.html"))
            all_html_files.extend([self.find_final_html()] if self.find_final_html() else [])
            
            for html_file in all_html_files:
                if html_file is None:
                    continue
                with open(html_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    
                # Look for the address in various formats
                address_patterns = [
                    r'123\s+market\s+street.*?san\s+francisco.*?94102',
                    r'123\s+market\s+st.*?san\s+francisco.*?94102',
                ]
                
                for pattern in address_patterns:
                    if re.search(pattern, html_content, re.IGNORECASE | re.DOTALL):
                        self.details['address_found'] = 'Correct address found in HTML'
                        return True
            
            # Check trajectory for form fills
            trajectory = self.get_trajectory()
            for entry in trajectory:
                if 'action' in entry:
                    action = entry['action']
                    if action.get('action_type') == 'TYPE':
                        text = action.get('parameters', {}).get('text', '').lower()
                        if '123 market' in text and ('san francisco' in text or '94102' in text):
                            self.details['address_typed'] = 'Address typed in form'
                            return True
            
            self.issues.append('Correct address not found (should be 123 Market Street, San Francisco, CA 94102)')
            return False
            
        except Exception as e:
            self.issues.append(f'Error checking address: {str(e)}')
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Label is set to "Shopping" """
        try:
            # Check all HTML files for the label
            all_html_files = sorted(self.result_dir.glob("step_*_raw.html"))
            all_html_files.extend([self.find_final_html()] if self.find_final_html() else [])
            
            for html_file in all_html_files:
                if html_file is None:
                    continue
                with open(html_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    
                # Look for "Shopping" label near the address or in saved places
                if 'shopping' in html_content.lower():
                    # Check if it's actually a saved place label, not just text
                    parser = SavedPlaceHTMLParser()
                    parser.feed(html_content)
                    
                    for place in parser.saved_places:
                        if place.get('label', '').lower() == 'shopping':
                            self.details['label_found'] = 'Shopping label found'
                            return True
                    
                    # Also do regex check for common patterns
                    if re.search(r'label["\s:]*shopping', html_content, re.IGNORECASE):
                        self.details['label_found'] = 'Shopping label found in HTML'
                        return True
            
            # Check trajectory for typing "Shopping"
            trajectory = self.get_trajectory()
            for entry in trajectory:
                if 'action' in entry:
                    action = entry['action']
                    if action.get('action_type') == 'TYPE':
                        text = action.get('parameters', {}).get('text', '')
                        if text.strip().lower() == 'shopping':
                            self.details['label_typed'] = 'Shopping typed as label'
                            return True
            
            self.issues.append('Label "Shopping" not found')
            return False
            
        except Exception as e:
            self.issues.append(f'Error checking label: {str(e)}')
            return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Icon is set to shopping cart icon"""
        try:
            # Check all HTML files for shopping cart icon
            all_html_files = sorted(self.result_dir.glob("step_*_raw.html"))
            all_html_files.extend([self.find_final_html()] if self.find_final_html() else [])
            
            for html_file in all_html_files:
                if html_file is None:
                    continue
                with open(html_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    
                # Look for shopping cart icon classes
                if re.search(r'fa-shopping-cart|shopping-cart|cart.*icon', html_content, re.IGNORECASE):
                    # Parse to verify it's associated with the Shopping saved place
                    parser = SavedPlaceHTMLParser()
                    parser.feed(html_content)
                    
                    for place in parser.saved_places:
                        if place.get('icon') == 'shopping-cart' or 'cart' in place.get('icon', ''):
                            self.details['icon_found'] = 'Shopping cart icon found'
                            return True
                    
                    # Also accept if icon is present near "Shopping" text
                    if 'shopping' in html_content.lower():
                        # Check if shopping cart icon is nearby
                        shopping_pos = html_content.lower().find('shopping')
                        nearby_text = html_content[max(0, shopping_pos-500):min(len(html_content), shopping_pos+500)]
                        if 'fa-shopping-cart' in nearby_text or 'shopping-cart' in nearby_text:
                            self.details['icon_found'] = 'Shopping cart icon near Shopping label'
                            return True
            
            # Check trajectory for icon selection
            trajectory = self.get_trajectory()
            for entry in trajectory:
                if 'action' in entry:
                    action = entry['action']
                    if action.get('action_type') == 'CLICK':
                        selector = action.get('parameters', {}).get('selector', '').lower()
                        reasoning = action.get('reasoning', '').lower()
                        if 'shopping-cart' in selector or 'cart' in selector or 'shopping' in selector and 'icon' in selector:
                            self.details['icon_clicked'] = 'Shopping cart icon clicked'
                            return True
                        if 'shopping' in reasoning and 'cart' in reasoning and 'icon' in reasoning:
                            self.details['icon_selected'] = 'Shopping cart icon selected based on reasoning'
                            return True
            
            self.issues.append('Shopping cart icon not found')
            return False
            
        except Exception as e:
            self.issues.append(f'Error checking icon: {str(e)}')
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_navigated_to_saved_places'] = self.check_checkpoint_1()
            self.checkpoints['cp2_saved_place_created'] = self.check_checkpoint_2()
            self.checkpoints['cp3_correct_address'] = self.check_checkpoint_3()
            self.checkpoints['cp4_correct_label'] = self.check_checkpoint_4()
            self.checkpoints['cp5_correct_icon'] = self.check_checkpoint_5()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 11,
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
                'query_id': 11,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python transportation_query11.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
