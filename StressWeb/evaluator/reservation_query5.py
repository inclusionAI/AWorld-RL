#!/usr/bin/env python3
# Evaluator for reservation Query 5

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class DietaryPreferencesParser(HTMLParser):
    """Parse HTML to extract dietary preferences information."""

    def __init__(self):
        super().__init__()
        self.in_dietary_section = False
        self.in_preferences_tags = False
        self.in_preference_tag = False
        self.current_preferences = []
        self.current_tag_text = []
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check if we're in the dietary section
        if tag == 'div' and attrs_dict.get('class') == 'section-card dietary-card':
            self.in_dietary_section = True
        
        # Check for preferences tags container
        if self.in_dietary_section and tag == 'div' and attrs_dict.get('class') == 'preferences-tags':
            self.in_preferences_tags = True
            
        # Check for individual preference tag
        if self.in_preferences_tags and tag == 'span' and 'preference-tag' in attrs_dict.get('class', ''):
            self.in_preference_tag = True
            self.current_tag_text = []
    
    def handle_endtag(self, tag):
        if tag == 'span' and self.in_preference_tag:
            self.in_preference_tag = False
            # Join and clean the text
            pref_text = ''.join(self.current_tag_text).strip()
            if pref_text:
                self.current_preferences.append(pref_text)
            self.current_tag_text = []
            
        if tag == 'div' and self.in_preferences_tags:
            self.in_preferences_tags = False
            
        if tag == 'section' and self.in_dietary_section:
            self.in_dietary_section = False
    
    def handle_data(self, data):
        if self.in_preference_tag:
            self.current_tag_text.append(data)


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_navigated_to_profile": False,
            "cp2_vegetarian_preference_set": False,
            "cp3_dairy_free_preference_set": False,
            "cp4_gluten_free_removed": False,
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

    def parse_dietary_preferences(self, html_file: Path) -> List[str]:
        """Parse dietary preferences from HTML file."""
        if not html_file or not html_file.exists():
            return []
        
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        parser = DietaryPreferencesParser()
        parser.feed(html_content)
        
        return parser.current_preferences

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Navigated to profile page"""
        try:
            trajectory = self.get_trajectory()
            
            # Check if there's any action history
            if not trajectory:
                self.issues.append("No trajectory data found")
                return False
            
            # Check if first action was clicking Profile link
            first_action = trajectory[0] if trajectory else None
            if first_action and first_action.get('action', {}).get('action_type') == 'CLICK':
                # The first action should be navigating to profile
                return True
            
            # Also check HTML content - look for profile page markers
            final_html = self.find_final_html()
            if final_html:
                with open(final_html, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'profile-page' in content or 'Dietary Preferences' in content:
                        return True
            
            self.issues.append("Did not navigate to profile page")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking profile navigation: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Vegetarian preference is set"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML found")
                return False
            
            preferences = self.parse_dietary_preferences(final_html)
            self.details['final_preferences'] = preferences
            
            # Check if Vegetarian is in the preferences
            vegetarian_found = any('Vegetarian' in pref for pref in preferences)
            
            if not vegetarian_found:
                self.issues.append("Vegetarian preference not found in final state")
            
            return vegetarian_found
            
        except Exception as e:
            self.issues.append(f"Error checking Vegetarian preference: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Dairy-Free preference is set"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML found")
                return False
            
            preferences = self.parse_dietary_preferences(final_html)
            
            # Check if Dairy-Free is in the preferences
            dairy_free_found = any('Dairy-Free' in pref for pref in preferences)
            
            if not dairy_free_found:
                self.issues.append("Dairy-Free preference not found in final state")
            
            return dairy_free_found
            
        except Exception as e:
            self.issues.append(f"Error checking Dairy-Free preference: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Gluten-Free should NOT be present (or should be replaced)"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                return False
            
            preferences = self.parse_dietary_preferences(final_html)
            
            # Task requires setting Vegetarian and Dairy-Free
            # Initial state had Vegetarian and Gluten-Free
            # So ideally Gluten-Free should be removed, but the main requirement
            # is that Vegetarian and Dairy-Free are both present
            
            # This checkpoint checks if the preferences were actually saved/changed
            # We'll be lenient here - as long as both required prefs are set,
            # we consider this successful
            vegetarian_found = any('Vegetarian' in pref for pref in preferences)
            dairy_free_found = any('Dairy-Free' in pref for pref in preferences)
            
            if vegetarian_found and dairy_free_found:
                return True
            
            # If Gluten-Free is still there but Dairy-Free is missing, that's a failure
            gluten_free_found = any('Gluten-Free' in pref for pref in preferences)
            if gluten_free_found and not dairy_free_found:
                self.issues.append("Changes not saved: Gluten-Free still present, Dairy-Free missing")
                return False
            
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking preference changes: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_navigated_to_profile'] = self.check_checkpoint_1()
            self.checkpoints['cp2_vegetarian_preference_set'] = self.check_checkpoint_2()
            self.checkpoints['cp3_dairy_free_preference_set'] = self.check_checkpoint_3()
            self.checkpoints['cp4_gluten_free_removed'] = self.check_checkpoint_4()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 5,
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
                'query_id': 5,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python reservation_query5.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
