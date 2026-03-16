#!/usr/bin/env python3
# Evaluator for reservation Query 9

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class TaskHTMLParser(HTMLParser):
    """Parse HTML to extract task-specific information."""

    def __init__(self):
        super().__init__()
        self.in_filter_section = False
        self.in_checkbox_label = False
        self.current_checkbox_checked = False
        self.checkbox_states = {}
        self.current_label_text = ""
        
        self.in_restaurant_detail = False
        self.in_detail_title = False
        self.restaurant_name = ""
        self.in_feature_tag = False
        self.restaurant_features = []
        
        self.in_restaurant_card = False
        self.in_card_name = False
        self.current_card_name = ""
        self.in_card_features = False
        self.current_card_features = []
        self.restaurant_cards = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check for filter section
        if tag == 'div' and attrs_dict.get('class') == 'filter-section':
            self.in_filter_section = True
        
        # Check for checkbox in filter section
        if self.in_filter_section and tag == 'input' and attrs_dict.get('type') == 'checkbox':
            self.current_checkbox_checked = 'checked' in attrs_dict or attrs_dict.get('checked') == ''
        
        # Check for checkbox label
        if self.in_filter_section and tag == 'label' and 'filter-checkbox-label' in attrs_dict.get('class', ''):
            self.in_checkbox_label = True
            self.current_label_text = ""
        
        # Check for restaurant detail page
        if tag == 'div' and 'restaurant-detail-page' in attrs_dict.get('class', ''):
            self.in_restaurant_detail = True
        
        # Check for detail title
        if self.in_restaurant_detail and tag == 'h1' and 'detail-title' in attrs_dict.get('class', ''):
            self.in_detail_title = True
        
        # Check for feature tags in detail page
        if self.in_restaurant_detail and tag == 'span' and 'feature-tag' in attrs_dict.get('class', ''):
            self.in_feature_tag = True
        
        # Check for restaurant cards in search results
        if tag == 'div' and 'restaurant-card' in attrs_dict.get('class', ''):
            self.in_restaurant_card = True
            self.current_card_name = ""
            self.current_card_features = []
        
        # Check for restaurant name in card
        if self.in_restaurant_card and tag == 'h3' and 'restaurant-name' in attrs_dict.get('class', ''):
            self.in_card_name = True
        
        # Check for feature tags in card
        if self.in_restaurant_card and tag == 'span' and 'feature-tag' in attrs_dict.get('class', ''):
            self.in_card_features = True

    def handle_endtag(self, tag):
        if tag == 'div':
            if self.in_filter_section:
                self.in_filter_section = False
            if self.in_restaurant_card:
                if self.current_card_name:
                    self.restaurant_cards.append({
                        'name': self.current_card_name,
                        'features': self.current_card_features.copy()
                    })
                self.in_restaurant_card = False
        
        if tag == 'label' and self.in_checkbox_label:
            self.in_checkbox_label = False
            if self.current_label_text:
                self.checkbox_states[self.current_label_text.strip()] = self.current_checkbox_checked
        
        if tag == 'h1' and self.in_detail_title:
            self.in_detail_title = False
        
        if tag == 'span' and self.in_feature_tag:
            self.in_feature_tag = False
        
        if tag == 'h3' and self.in_card_name:
            self.in_card_name = False
        
        if tag == 'span' and self.in_card_features:
            self.in_card_features = False

    def handle_data(self, data):
        if self.in_checkbox_label:
            self.current_label_text += data
        
        if self.in_detail_title:
            self.restaurant_name += data.strip()
        
        if self.in_feature_tag:
            self.restaurant_features.append(data.strip())
        
        if self.in_card_name:
            self.current_card_name += data.strip()
        
        if self.in_card_features:
            self.current_card_features.append(data.strip())


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task
        self.checkpoints = {
            "cp1_navigated_to_search": False,
            "cp2_applied_all_filters": False,
            "cp3_results_filtered_correctly": False,
            "cp4_viewed_detail_page": False,
            "cp5_restaurant_has_required_features": False,
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
        """Checkpoint 1: Navigated to search page with filters visible"""
        try:
            trajectory = self.get_trajectory()
            
            # Check if agent navigated to search page
            for entry in trajectory:
                if 'url' in entry and '/search' in entry.get('url', ''):
                    self.details['search_page_reached'] = True
                    
                    # Check if filters are visible in any intermediate step
                    step_files = self.find_step_html("step_*")
                    for step_file in step_files:
                        with open(step_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if 'filter-section' in content and 'Valet Parking' in content:
                                self.details['filters_visible'] = True
                                return True
            
            # Also check final HTML
            final_html = self.find_final_html()
            if final_html:
                with open(final_html, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'search-results-page' in content or 'filter-section' in content:
                        self.details['search_page_reached'] = True
                        return True
            
            self.issues.append("Did not navigate to search page")
            return False
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 1: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Applied all three required filters"""
        try:
            # Check intermediate steps around filter application (steps 54-58)
            step_files = self.find_step_html("step_5[4-8]*")
            
            filters_checked = {
                'valet_parking': False,
                'outdoor_seating': False,
                'wheelchair_accessible': False
            }
            
            for step_file in step_files:
                parser = TaskHTMLParser()
                with open(step_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    parser.feed(content)
                
                # Check checkbox states
                for label_text, is_checked in parser.checkbox_states.items():
                    if is_checked:
                        if 'Valet Parking' in label_text:
                            filters_checked['valet_parking'] = True
                        if 'Outdoor Seating' in label_text:
                            filters_checked['outdoor_seating'] = True
                        if 'Wheelchair Accessible' in label_text:
                            filters_checked['wheelchair_accessible'] = True
            
            self.details['filters_applied'] = filters_checked
            
            all_filters_applied = all(filters_checked.values())
            if not all_filters_applied:
                missing = [k for k, v in filters_checked.items() if not v]
                self.issues.append(f"Not all filters applied. Missing: {missing}")
            
            return all_filters_applied
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 2: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Results filtered correctly to show only matching restaurants"""
        try:
            # Check step after all filters applied (around step 58-59)
            step_files = self.find_step_html("step_5[8-9]*")
            
            for step_file in step_files:
                parser = TaskHTMLParser()
                with open(step_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    parser.feed(content)
                
                if parser.restaurant_cards:
                    # Check if displayed restaurants have all 3 required features
                    valid_restaurants = []
                    invalid_restaurants = []
                    
                    for card in parser.restaurant_cards:
                        features_text = ' '.join(card['features'])
                        has_valet = '��' in features_text or 'Valet Parking' in features_text
                        has_outdoor = '🌿' in features_text or 'Outdoor Seating' in features_text
                        has_accessible = '♿' in features_text or 'Accessible' in features_text
                        
                        if has_valet and has_outdoor and has_accessible:
                            valid_restaurants.append(card['name'])
                        else:
                            invalid_restaurants.append(card['name'])
                    
                    self.details['valid_restaurants'] = valid_restaurants
                    self.details['invalid_restaurants'] = invalid_restaurants
                    self.details['total_restaurants_shown'] = len(parser.restaurant_cards)
                    
                    # At least one valid restaurant should be shown
                    if valid_restaurants:
                        return True
                    else:
                        self.issues.append("No restaurants with all 3 features shown in results")
                        return False
            
            # If no restaurant cards found, check if search was performed
            self.issues.append("Could not verify filtered results")
            return False
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 3: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Viewed restaurant detail page"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML found")
                return False
            
            parser = TaskHTMLParser()
            with open(final_html, 'r', encoding='utf-8') as f:
                content = f.read()
                parser.feed(content)
            
            if parser.in_restaurant_detail or 'restaurant-detail-page' in content:
                if parser.restaurant_name:
                    self.details['viewed_restaurant'] = parser.restaurant_name
                    return True
                else:
                    # Try to extract restaurant name from HTML
                    match = re.search(r'<h1[^>]*class="[^"]*detail-title[^"]*"[^>]*>.*?([A-Z][^<]+)</h1>', content)
                    if match:
                        self.details['viewed_restaurant'] = match.group(1).strip()
                        return True
            
            self.issues.append("Did not reach restaurant detail page")
            return False
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 4: {str(e)}")
            return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Viewed restaurant has all required features"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                return False
            
            with open(final_html, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for the three required features in detail page
            has_valet = '🚗' in content or 'Valet Parking' in content
            has_outdoor = '🌿' in content or 'Outdoor Seating' in content
            has_accessible = '♿' in content or 'Accessible' in content or 'Wheelchair Accessible' in content
            
            features_found = {
                'valet_parking': has_valet,
                'outdoor_seating': has_outdoor,
                'wheelchair_accessible': has_accessible
            }
            
            self.details['restaurant_features'] = features_found
            
            # The viewed restaurant should have all 3 features
            if all(features_found.values()):
                return True
            else:
                missing = [k for k, v in features_found.items() if not v]
                self.issues.append(f"Viewed restaurant missing features: {missing}")
                return False
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 5: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_navigated_to_search'] = self.check_checkpoint_1()
            self.checkpoints['cp2_applied_all_filters'] = self.check_checkpoint_2()
            self.checkpoints['cp3_results_filtered_correctly'] = self.check_checkpoint_3()
            self.checkpoints['cp4_viewed_detail_page'] = self.check_checkpoint_4()
            self.checkpoints['cp5_restaurant_has_required_features'] = self.check_checkpoint_5()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 9,
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
                'query_id': 9,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python reservation_query9.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
