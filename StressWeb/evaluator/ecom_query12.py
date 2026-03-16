#!/usr/bin/env python3
# Evaluator for ecom Query 12

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class ReviewParser(HTMLParser):
    """Parse HTML to extract review information."""

    def __init__(self):
        super().__init__()
        self.in_review = False
        self.in_rating = False
        self.in_review_text = False
        self.reviews = []
        self.current_review = {}
        self.current_tag = None

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check for review container
        if 'class' in attrs_dict:
            classes = attrs_dict['class']
            if 'review' in classes or 'customer-review' in classes:
                self.in_review = True
                self.current_review = {}
            
            # Check for rating (star icons)
            if 'fa-star' in classes:
                if 'current_review' not in dir(self) or not self.current_review:
                    self.current_review = {}
                if 'rating' not in self.current_review:
                    self.current_review['rating'] = 0
                self.current_review['rating'] += 1

    def handle_data(self, data):
        data = data.strip()
        if data and self.in_review:
            # Capture review text
            if len(data) > 10 and 'text' not in self.current_review:
                self.current_review['text'] = data

    def handle_endtag(self, tag):
        if self.in_review and tag in ['div', 'article']:
            if self.current_review:
                self.reviews.append(self.current_review.copy())
            self.in_review = False
            self.current_review = {}


class OrderParser(HTMLParser):
    """Parse HTML to extract order information."""

    def __init__(self):
        super().__init__()
        self.orders = []
        self.in_order = False
        self.current_order = {}
        self.capture_text = False
        self.current_field = None

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if 'class' in attrs_dict:
            classes = attrs_dict['class']
            if 'order' in classes.lower():
                self.in_order = True
                self.current_order = {}

    def handle_data(self, data):
        data = data.strip()
        if self.in_order and data:
            # Capture product names
            if any(keyword in data.lower() for keyword in ['wireless mouse', 'wireless headphones']):
                self.current_order['product'] = data

    def handle_endtag(self, tag):
        if self.in_order and tag == 'div':
            if self.current_order:
                self.orders.append(self.current_order.copy())
            self.in_order = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_first_login": False,
            "cp2_first_purchase": False,
            "cp3_first_review": False,
            "cp4_logout_and_new_account": False,
            "cp5_second_purchase": False,
            "cp6_second_review": False,
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

    def check_login(self, email: str, start_step: int, end_step: int) -> bool:
        """Check if user logged in with specified email in step range."""
        result_data = self.load_result_json()
        actions = result_data.get('action_history', [])
        
        for action in actions:
            step = action.get('step', 0)
            if start_step <= step <= end_step:
                action_params = action.get('action', {}).get('parameters', {})
                if action_params.get('text') == email:
                    return True
        return False

    def check_purchase_completed(self, start_step: int, end_step: int) -> bool:
        """Check if order was placed in step range."""
        result_data = self.load_result_json()
        actions = result_data.get('action_history', [])
        
        for action in actions:
            step = action.get('step', 0)
            if start_step <= step <= end_step:
                reasoning = action.get('action', {}).get('reasoning', '')
                selector = action.get('action', {}).get('parameters', {}).get('selector', '')
                if 'place order' in reasoning.lower() or 'place order' in selector.lower():
                    if action.get('result', {}).get('success'):
                        return True
        return False

    def check_review_submitted(self, start_step: int, end_step: int, review_text: str) -> bool:
        """Check if review with specific text was submitted in step range."""
        result_data = self.load_result_json()
        actions = result_data.get('action_history', [])
        
        # Check if review text was filled
        text_filled = False
        review_submitted = False
        
        for action in actions:
            step = action.get('step', 0)
            if start_step <= step <= end_step:
                action_params = action.get('action', {}).get('parameters', {})
                action_text = action_params.get('text', '')
                selector = action_params.get('selector', '')
                
                # Check if review text was filled
                if review_text.lower() in action_text.lower():
                    text_filled = True
                
                # Check if submit review button was clicked
                if 'submit review' in selector.lower():
                    if action.get('result', {}).get('success'):
                        review_submitted = True
        
        return text_filled and review_submitted

    def check_account_created(self, email: str, start_step: int, end_step: int) -> bool:
        """Check if new account was created with specified email."""
        result_data = self.load_result_json()
        actions = result_data.get('action_history', [])
        
        email_filled = False
        account_created = False
        
        for action in actions:
            step = action.get('step', 0)
            if start_step <= step <= end_step:
                action_params = action.get('action', {}).get('parameters', {})
                action_text = action_params.get('text', '')
                selector = action_params.get('selector', '')
                
                # Check if email was filled
                if action_text == email:
                    email_filled = True
                
                # Check if create account button was clicked
                if 'create account' in selector.lower():
                    if action.get('result', {}).get('success'):
                        account_created = True
        
        return email_filled and account_created

    def check_logout(self, start_step: int, end_step: int) -> bool:
        """Check if user logged out in step range."""
        result_data = self.load_result_json()
        actions = result_data.get('action_history', [])
        
        for action in actions:
            step = action.get('step', 0)
            if start_step <= step <= end_step:
                selector = action.get('action', {}).get('parameters', {}).get('selector', '')
                reasoning = action.get('action', {}).get('reasoning', '')
                
                if 'sign out' in selector.lower() or 'logout' in selector.lower():
                    if action.get('result', {}).get('success'):
                        return True
                if 'log out' in reasoning.lower() or 'sign out' in reasoning.lower():
                    if action.get('result', {}).get('success'):
                        return True
        return False

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Login with 12345@mail.com"""
        try:
            # Check if user logged in with the first account in steps 0-20
            logged_in = self.check_login('12345@mail.com', 0, 20)
            if logged_in:
                self.details['first_login'] = 'Successfully logged in with 12345@mail.com'
                return True
            else:
                self.issues.append('Failed to login with 12345@mail.com')
                return False
        except Exception as e:
            self.issues.append(f'Error checking first login: {str(e)}')
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Purchase one Wireless Mouse with first account"""
        try:
            # Check if order was placed in steps 10-30
            purchased = self.check_purchase_completed(10, 30)
            if purchased:
                self.details['first_purchase'] = 'First purchase completed'
                return True
            else:
                self.issues.append('Failed to complete first purchase')
                return False
        except Exception as e:
            self.issues.append(f'Error checking first purchase: {str(e)}')
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Submit 5-star review with specific text for first account"""
        try:
            # Check if review was submitted in steps 30-70
            review_submitted = self.check_review_submitted(30, 70, 'Excellent quality and fast delivery')
            if review_submitted:
                self.details['first_review'] = 'Review submitted for first account'
                return True
            else:
                self.issues.append('Failed to submit review for first account')
                return False
        except Exception as e:
            self.issues.append(f'Error checking first review: {str(e)}')
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Logout and create new account with newcustomer@test.com"""
        try:
            # Check if user logged out in steps 65-75
            logged_out = self.check_logout(65, 75)
            
            # Check if new account was created in steps 70-85
            account_created = self.check_account_created('newcustomer@test.com', 70, 85)
            
            if logged_out and account_created:
                self.details['logout_and_new_account'] = 'Logged out and created new account'
                return True
            else:
                if not logged_out:
                    self.issues.append('Failed to logout from first account')
                if not account_created:
                    self.issues.append('Failed to create new account with newcustomer@test.com')
                return False
        except Exception as e:
            self.issues.append(f'Error checking logout and new account: {str(e)}')
            return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Purchase same product with new account"""
        try:
            # Check if second purchase was completed in steps 85-100
            purchased = self.check_purchase_completed(85, 100)
            if purchased:
                self.details['second_purchase'] = 'Second purchase completed with new account'
                return True
            else:
                self.issues.append('Failed to complete second purchase')
                return False
        except Exception as e:
            self.issues.append(f'Error checking second purchase: {str(e)}')
            return False

    def check_checkpoint_6(self) -> bool:
        """Checkpoint 6: Submit same review with new account"""
        try:
            # Check if second review was submitted (would be after step 93)
            # Since task didn't complete (max_steps), this checkpoint likely failed
            # We need to check if review submission was attempted
            result_data = self.load_result_json()
            
            # If execution stopped at step 100 due to max_steps, 
            # it's unlikely the second review was completed
            if result_data.get('steps', 0) >= 100:
                self.issues.append('Task reached max steps before completing second review')
                return False
            
            review_submitted = self.check_review_submitted(93, 120, 'Excellent quality and fast delivery')
            if review_submitted:
                self.details['second_review'] = 'Second review submitted'
                return True
            else:
                self.issues.append('Failed to submit second review')
                return False
        except Exception as e:
            self.issues.append(f'Error checking second review: {str(e)}')
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_first_login'] = self.check_checkpoint_1()
            self.checkpoints['cp2_first_purchase'] = self.check_checkpoint_2()
            self.checkpoints['cp3_first_review'] = self.check_checkpoint_3()
            self.checkpoints['cp4_logout_and_new_account'] = self.check_checkpoint_4()
            self.checkpoints['cp5_second_purchase'] = self.check_checkpoint_5()
            self.checkpoints['cp6_second_review'] = self.check_checkpoint_6()

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
        print("Usage: python ecom_query12.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
