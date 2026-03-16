#!/usr/bin/env python3
# Evaluator for reservation Query 10

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
        self.in_review_card = False
        self.in_review_actions = False
        self.in_helpful_btn = False
        self.current_review_author = None
        self.in_author_name = False
        self.helpful_buttons = []
        self.current_btn_classes = []
        self.current_btn_text = ""
        
        # For restaurant features
        self.in_feature_tag = False
        self.features = []
        self.current_feature = ""

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Detect review card
        if tag == 'div' and 'class' in attrs_dict:
            if 'review-card' in attrs_dict['class']:
                self.in_review_card = True
                self.current_review_author = None
            elif 'review-actions' in attrs_dict['class']:
                self.in_review_actions = True
            elif 'author-name' in attrs_dict['class']:
                self.in_author_name = True
        
        # Detect helpful button
        if tag == 'button' and 'class' in attrs_dict:
            if 'helpful-btn' in attrs_dict['class']:
                self.in_helpful_btn = True
                self.current_btn_classes = attrs_dict['class'].split()
                self.current_btn_text = ""
        
        # Detect feature tags
        if tag == 'span' and 'class' in attrs_dict:
            if 'feature-tag' in attrs_dict['class']:
                self.in_feature_tag = True
                self.current_feature = ""

    def handle_data(self, data):
        if self.in_author_name:
            self.current_review_author = data.strip()
        
        if self.in_helpful_btn:
            self.current_btn_text += data.strip()
        
        if self.in_feature_tag:
            self.current_feature += data.strip()

    def handle_endtag(self, tag):
        if tag == 'div':
            if self.in_review_actions:
                self.in_review_actions = False
            if self.in_review_card:
                self.in_review_card = False
            if self.in_author_name:
                self.in_author_name = False
        
        if tag == 'button' and self.in_helpful_btn:
            self.in_helpful_btn = False
            is_marked = 'marked' in self.current_btn_classes
            self.helpful_buttons.append({
                'author': self.current_review_author,
                'text': self.current_btn_text,
                'marked': is_marked,
                'classes': self.current_btn_classes
            })
        
        if tag == 'span' and self.in_feature_tag:
            self.in_feature_tag = False
            self.features.append(self.current_feature)


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task
        self.checkpoints = {
            "cp1_searched_with_valet_parking": False,
            "cp2_searched_with_outdoor_seating": False,
            "cp3_viewed_restaurant_details": False,
            "cp4_marked_first_review_helpful": False,
            "cp5_marked_second_review_helpful": False,
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
        """Checkpoint 1: Searched with 'Valet Parking' filter"""
        try:
            # Check step 21 or 22 where filters were applied
            step_files = self.find_step_html("step_2[12]*")
            
            for step_file in step_files:
                with open(step_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    
                # Check if Valet Parking checkbox is checked
                if 'Valet Parking' in html_content:
                    # Look for checked checkbox near Valet Parking
                    parser = TaskHTMLParser()
                    parser.feed(html_content)
                    
                    if any('Valet Parking' in feature for feature in parser.features):
                        self.details['valet_parking_filter'] = 'Found in features'
                        return True
            
            self.issues.append("Valet Parking filter not properly applied")
            return False
        except Exception as e:
            self.issues.append(f"Error checking Valet Parking filter: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Searched with 'Outdoor Seating' filter"""
        try:
            # Check step 22 where both filters should be applied
            step_files = self.find_step_html("step_22*")
            
            for step_file in step_files:
                with open(step_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    
                parser = TaskHTMLParser()
                parser.feed(html_content)
                
                if any('Outdoor Seating' in feature for feature in parser.features):
                    self.details['outdoor_seating_filter'] = 'Found in features'
                    return True
            
            self.issues.append("Outdoor Seating filter not properly applied")
            return False
        except Exception as e:
            self.issues.append(f"Error checking Outdoor Seating filter: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Viewed restaurant details with reviews"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML found")
                return False
            
            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Check if on restaurant detail page with reviews
            if 'Guest Reviews' in html_content or 'reviews-section' in html_content:
                # Verify it's Bamboo Modern Kitchen (the restaurant with both features)
                if 'Bamboo Modern Kitchen' in html_content:
                    self.details['restaurant'] = 'Bamboo Modern Kitchen'
                    return True
                else:
                    self.issues.append("Wrong restaurant viewed")
                    return False
            
            self.issues.append("Restaurant details page not reached or reviews section not found")
            return False
        except Exception as e:
            self.issues.append(f"Error checking restaurant details: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Marked first review as helpful"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML found")
                return False
            
            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            parser = TaskHTMLParser()
            parser.feed(html_content)
            
            if len(parser.helpful_buttons) >= 1:
                first_review = parser.helpful_buttons[0]
                if first_review['marked']:
                    self.details['first_review_author'] = first_review.get('author', 'Unknown')
                    self.details['first_review_marked'] = True
                    return True
                else:
                    self.issues.append("First review not marked as helpful")
                    return False
            
            self.issues.append("No helpful buttons found")
            return False
        except Exception as e:
            self.issues.append(f"Error checking first review: {str(e)}")
            return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Marked second review as helpful"""
        try:
            # Check intermediate step where second helpful was clicked (step 33)
            step_33_files = self.find_step_html("step_33*")
            
            # Also check final state
            final_html = self.find_final_html()
            
            html_files_to_check = []
            if step_33_files:
                html_files_to_check.extend(step_33_files)
            if final_html:
                html_files_to_check.append(final_html)
            
            for html_file in html_files_to_check:
                with open(html_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                parser = TaskHTMLParser()
                parser.feed(html_content)
                
                if len(parser.helpful_buttons) >= 2:
                    marked_count = sum(1 for btn in parser.helpful_buttons[:2] if btn['marked'])
                    
                    if marked_count >= 2:
                        self.details['second_review_author'] = parser.helpful_buttons[1].get('author', 'Unknown')
                        self.details['second_review_marked'] = True
                        self.details['marked_reviews_count'] = marked_count
                        return True
            
            # If not found in any file, check if at least one was marked
            self.issues.append("Second review not marked as helpful")
            self.details['marked_reviews_count'] = 1  # Only first was marked
            return False
        except Exception as e:
            self.issues.append(f"Error checking second review: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_searched_with_valet_parking'] = self.check_checkpoint_1()
            self.checkpoints['cp2_searched_with_outdoor_seating'] = self.check_checkpoint_2()
            self.checkpoints['cp3_viewed_restaurant_details'] = self.check_checkpoint_3()
            self.checkpoints['cp4_marked_first_review_helpful'] = self.check_checkpoint_4()
            self.checkpoints['cp5_marked_second_review_helpful'] = self.check_checkpoint_5()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 10,
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
                'query_id': 10,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python reservation_query10.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
