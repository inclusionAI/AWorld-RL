#!/usr/bin/env python3
# Evaluator for network Query 14

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional
from html.parser import HTMLParser


class TaskHTMLParser(HTMLParser):
    """Parse HTML to extract task-specific information."""

    def __init__(self):
        super().__init__()
        self.in_notifications_main = False
        self.in_filter_button = False
        self.in_notification_card = False
        self.in_notification_content = False
        self.in_mark_button = False
        self.in_job_detail = False
        self.in_job_title = False
        self.in_network_page = False
        self.in_connection_requests = False
        
        self.current_filter_button_class = ""
        self.current_button_disabled = False
        self.current_notification_class = ""
        self.notification_text = ""
        self.filter_buttons = []
        self.notifications = []
        self.unread_count = 0
        self.mark_button_disabled = False
        self.job_title = ""
        self.page_class = ""
        self.connection_requests = []
        self.current_request_name = ""

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check for page type
        if tag == "div" and "class" in attrs_dict:
            if "notifications-page" in attrs_dict["class"]:
                self.in_notifications_main = True
            if "job-detail-page" in attrs_dict["class"]:
                self.in_job_detail = True
                self.page_class = attrs_dict["class"]
            if "network-page" in attrs_dict["class"]:
                self.in_network_page = True
        
        # Check for filter buttons (tabs)
        if tag == "button" and "class" in attrs_dict:
            if "filter-button" in attrs_dict["class"]:
                self.in_filter_button = True
                self.current_filter_button_class = attrs_dict["class"]
            if "mark-all-button" in attrs_dict["class"]:
                self.in_mark_button = True
                self.current_button_disabled = "disabled" in attrs_dict
        
        # Check for notification cards
        if tag == "div" and "class" in attrs_dict:
            if "notification-card" in attrs_dict["class"]:
                self.in_notification_card = True
                self.current_notification_class = attrs_dict["class"]
                self.notification_text = ""
            if "notification-content" in attrs_dict["class"]:
                self.in_notification_content = True
        
        # Check for job title
        if self.in_job_detail and tag == "h2":
            self.in_job_title = True
        
        # Check for connection requests
        if self.in_network_page and tag == "h3":
            self.current_request_name = ""

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
        
        # Capture filter button text
        if self.in_filter_button:
            self.filter_buttons.append({
                "text": data,
                "class": self.current_filter_button_class
            })
        
        # Capture notification content
        if self.in_notification_content:
            self.notification_text += data + " "
        
        # Capture job title
        if self.in_job_title:
            self.job_title = data
        
        # Count unread notifications
        if self.in_notification_card and "Unread" in data:
            match = re.search(r'Unread \((\d+)\)', data)
            if match:
                self.unread_count = int(match.group(1))

    def handle_endtag(self, tag):
        if tag == "button":
            if self.in_filter_button:
                self.in_filter_button = False
            if self.in_mark_button:
                self.mark_button_disabled = self.current_button_disabled
                self.in_mark_button = False
        
        if tag == "div":
            if self.in_notification_card:
                if self.notification_text:
                    self.notifications.append({
                        "text": self.notification_text.strip(),
                        "class": self.current_notification_class
                    })
                self.in_notification_card = False
                self.in_notification_content = False
        
        if tag == "h2" and self.in_job_title:
            self.in_job_title = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task
        self.checkpoints = {
            "cp1_notifications_marked_read": False,
            "cp2_posts_tab_opened": False,
            "cp3_post_notifications_viewed": False,
            "cp4_connection_requests_accepted": False,
            "cp5_john_accepted_others_rejected": False,
            "cp6_frontend_engineer_viewed": False,
            "cp7_product_search_executed": False,
            "cp8_product_job_viewed": False,
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

    def parse_html(self, html_path: Path) -> TaskHTMLParser:
        """Parse HTML file and return parser with extracted data."""
        parser = TaskHTMLParser()
        with open(html_path, 'r', encoding='utf-8') as f:
            parser.feed(f.read())
        return parser

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Notifications marked as read"""
        # Check step 2 (after marking as read) for disabled mark button and 0 unread
        step_files = self.find_step_html("step_2_*")
        if not step_files:
            self.issues.append("Step 2 HTML not found - cannot verify notifications marked")
            return False
        
        parser = self.parse_html(step_files[0])
        
        # Check if mark button is disabled
        if not parser.mark_button_disabled:
            self.issues.append("Mark all button not disabled after marking")
            return False
        
        # Check for Unread (0) in the HTML content
        with open(step_files[0], 'r', encoding='utf-8') as f:
            html_content = f.read()
            if 'Unread (0)' not in html_content:
                self.issues.append("Unread count not 0 after marking all as read")
                return False
        
        self.details['notifications_marked'] = True
        return True

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Posts tab opened"""
        # Check step 3 for active Posts tab
        step_files = self.find_step_html("step_3_*")
        if not step_files:
            self.issues.append("Step 3 HTML not found - cannot verify Posts tab opened")
            return False
        
        with open(step_files[0], 'r', encoding='utf-8') as f:
            html_content = f.read()
            # Look for active Posts tab
            if 'filter-button active">Posts' not in html_content and \
               '<button class="filter-button active">Posts</button>' not in html_content:
                self.issues.append("Posts tab not active in step 3")
                return False
        
        self.details['posts_tab_active'] = True
        return True

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Post notifications sequentially viewed"""
        trajectory = self.get_trajectory()
        
        # Look for clicks on post notifications
        post_notification_clicks = []
        for entry in trajectory:
            if 'action' in entry and entry['action'].get('action_type') == 'CLICK':
                selector = entry['action'].get('parameters', {}).get('selector', '')
                if 'commented on your post' in selector or 'liked your post' in selector:
                    post_notification_clicks.append(selector)
        
        # Should have viewed at least 3 post notifications based on task
        if len(post_notification_clicks) < 3:
            self.issues.append(f"Only {len(post_notification_clicks)} post notifications clicked, expected at least 3")
            return False
        
        self.details['post_notifications_viewed'] = len(post_notification_clicks)
        return True

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Connection requests accepted on Network page"""
        trajectory = self.get_trajectory()
        
        # Count Accept button clicks on Network page
        accept_clicks = 0
        for entry in trajectory:
            if 'action' in entry and entry['action'].get('action_type') == 'CLICK':
                selector = entry['action'].get('parameters', {}).get('selector', '')
                if 'Accept' in selector:
                    accept_clicks += 1
        
        # Should have at least 2 accept clicks based on the trajectory
        if accept_clicks < 2:
            self.issues.append(f"Only {accept_clicks} accept clicks found, expected at least 2")
            return False
        
        self.details['accept_clicks'] = accept_clicks
        return True

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: John Anderson accepted, others rejected"""
        trajectory = self.get_trajectory()
        
        # Look for decline/reject actions
        decline_clicks = 0
        for entry in trajectory:
            if 'action' in entry and entry['action'].get('action_type') == 'CLICK':
                selector = entry['action'].get('parameters', {}).get('selector', '')
                if 'Decline' in selector or 'Reject' in selector:
                    decline_clicks += 1
        
        # Should have at least 1 decline based on task (reject all others)
        if decline_clicks < 1:
            self.issues.append("No decline actions found for rejecting other requests")
            return False
        
        self.details['decline_clicks'] = decline_clicks
        return True

    def check_checkpoint_6(self) -> bool:
        """Checkpoint 6: Frontend Engineer job viewed"""
        trajectory = self.get_trajectory()
        
        # Look for click on Frontend Engineer
        frontend_clicked = False
        for entry in trajectory:
            if 'action' in entry and entry['action'].get('action_type') == 'CLICK':
                selector = entry['action'].get('parameters', {}).get('selector', '')
                if 'Frontend Engineer' in selector:
                    frontend_clicked = True
                    break
        
        if not frontend_clicked:
            self.issues.append("Frontend Engineer job not clicked")
            return False
        
        self.details['frontend_engineer_viewed'] = True
        return True

    def check_checkpoint_7(self) -> bool:
        """Checkpoint 7: Product search executed"""
        trajectory = self.get_trajectory()
        
        # Look for FILL action with "Product"
        product_search = False
        search_button_clicked = False
        
        for entry in trajectory:
            if 'action' in entry:
                action = entry['action']
                if action.get('action_type') == 'FILL':
                    text = action.get('parameters', {}).get('text', '')
                    if 'Product' in text:
                        product_search = True
                if action.get('action_type') == 'CLICK':
                    selector = action.get('parameters', {}).get('selector', '')
                    if 'Search' in selector:
                        search_button_clicked = True
        
        if not product_search:
            self.issues.append("Product search not filled")
            return False
        
        if not search_button_clicked:
            self.issues.append("Search button not clicked")
            return False
        
        self.details['product_search_executed'] = True
        return True

    def check_checkpoint_8(self) -> bool:
        """Checkpoint 8: Product job details viewed (final state)"""
        final_html = self.find_final_html()
        if not final_html:
            self.issues.append("Final HTML not found")
            return False
        
        parser = self.parse_html(final_html)
        
        # Check if on job detail page
        if not parser.in_job_detail:
            self.issues.append("Not on job detail page in final state")
            return False
        
        # Check if it's a Product-related job
        if not parser.job_title or 'Product' not in parser.job_title:
            self.issues.append(f"Job title '{parser.job_title}' does not contain 'Product'")
            return False
        
        self.details['final_job_title'] = parser.job_title
        return True

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_notifications_marked_read'] = self.check_checkpoint_1()
            self.checkpoints['cp2_posts_tab_opened'] = self.check_checkpoint_2()
            self.checkpoints['cp3_post_notifications_viewed'] = self.check_checkpoint_3()
            self.checkpoints['cp4_connection_requests_accepted'] = self.check_checkpoint_4()
            self.checkpoints['cp5_john_accepted_others_rejected'] = self.check_checkpoint_5()
            self.checkpoints['cp6_frontend_engineer_viewed'] = self.check_checkpoint_6()
            self.checkpoints['cp7_product_search_executed'] = self.check_checkpoint_7()
            self.checkpoints['cp8_product_job_viewed'] = self.check_checkpoint_8()

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
        print("Usage: python network_query14.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
