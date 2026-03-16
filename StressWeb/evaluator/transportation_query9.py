#!/usr/bin/env python3
# Evaluator for transportation Query 9

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class RideHistoryHTMLParser(HTMLParser):
    """Parse HTML to extract ride history information."""

    def __init__(self):
        super().__init__()
        self.in_filter_option = False
        self.in_ride_date = False
        self.in_date_div = False
        self.in_date_info = False
        self.in_date_large = False
        self.in_time_large = False
        self.current_date = None
        self.current_time = None
        self.filter_options = []
        self.dates_in_list = []
        self.ride_detail_date = None
        self.ride_detail_time = None
        self.active_filter = None
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check for active filter button
        if tag == 'button' and 'class' in attrs_dict:
            classes = attrs_dict['class']
            if 'filter-option' in classes:
                self.in_filter_option = True
                if 'active' in classes:
                    self.active_filter = 'active'
                else:
                    self.active_filter = None
        
        # Check for ride date in list
        if tag == 'div' and 'class' in attrs_dict:
            if attrs_dict['class'] == 'ride-date':
                self.in_ride_date = True
            elif attrs_dict['class'] == 'date' and self.in_ride_date:
                self.in_date_div = True
            elif attrs_dict['class'] == 'date-info':
                self.in_date_info = True
            elif attrs_dict['class'] == 'date-large' and self.in_date_info:
                self.in_date_large = True
            elif attrs_dict['class'] == 'time-large' and self.in_date_info:
                self.in_time_large = True

    def handle_data(self, data):
        text = data.strip()
        if not text:
            return
            
        # Capture filter options
        if self.in_filter_option:
            if self.active_filter == 'active':
                self.filter_options.append(('active', text))
            else:
                self.filter_options.append(('inactive', text))
        
        # Capture ride dates in list
        if self.in_date_div:
            self.current_date = text
        
        # Capture ride detail page date
        if self.in_date_large:
            self.ride_detail_date = text
        
        if self.in_time_large:
            self.ride_detail_time = text

    def handle_endtag(self, tag):
        if tag == 'button':
            self.in_filter_option = False
            self.active_filter = None
        if tag == 'div':
            if self.in_date_div:
                if self.current_date:
                    self.dates_in_list.append(self.current_date)
                    self.current_date = None
                self.in_date_div = False
            if self.in_ride_date:
                self.in_ride_date = False
            if self.in_date_info:
                self.in_date_info = False
            if self.in_date_large:
                self.in_date_large = False
            if self.in_time_large:
                self.in_time_large = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints
        self.checkpoints = {
            "cp1_navigated_to_ride_history": False,
            "cp2_filtered_last_7_days": False,
            "cp3_identified_earliest_trip": False,
            "cp4_viewed_trip_details": False,
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
        """Checkpoint 1: Navigated to Ride History page"""
        try:
            trajectory = self.get_trajectory()
            
            # Check if agent clicked on Ride History
            for entry in trajectory:
                if 'action' in entry:
                    action = entry['action']
                    if action.get('action_type') == 'CLICK':
                        selector = action.get('parameters', {}).get('selector', '')
                        if 'Ride History' in selector or 'ride-history' in selector.lower():
                            self.details['ride_history_navigation'] = 'Found Ride History click action'
                            return True
            
            # Also check if URL contains ride-history
            for entry in trajectory:
                if 'url' in entry:
                    if 'ride-history' in entry['url'].lower():
                        self.details['ride_history_navigation'] = 'Found ride-history in URL'
                        return True
            
            self.issues.append("Did not navigate to Ride History page")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking ride history navigation: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Filtered trips from last 7 days"""
        try:
            # Find step HTML files where the filter is shown
            step_files = self.find_step_html("step_*")
            
            for step_file in step_files:
                with open(step_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    
                parser = RideHistoryHTMLParser()
                parser.feed(html_content)
                
                # Check if "Last 7 Days" filter is active
                for status, filter_text in parser.filter_options:
                    if 'Last 7 Days' in filter_text or 'LAST 7 DAYS' in filter_text:
                        if status == 'active':
                            self.details['filter_applied'] = f'Last 7 Days filter active in {step_file.name}'
                            return True
            
            # Also check in trajectory for LAST 7 DAYS button click
            trajectory = self.get_trajectory()
            for entry in trajectory:
                if 'action' in entry:
                    action = entry['action']
                    if action.get('action_type') == 'CLICK':
                        selector = action.get('parameters', {}).get('selector', '')
                        if 'LAST 7 DAYS' in selector or 'Last 7 Days' in selector:
                            self.details['filter_clicked'] = 'Found LAST 7 DAYS button click'
                            # Need to verify it was actually applied
                            return True
            
            self.issues.append("Did not filter trips by Last 7 Days")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking filter: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Identified the earliest trip in the filtered list"""
        try:
            # Find step files that show the filtered list with Last 7 Days active
            step_files = self.find_step_html("step_*")
            
            earliest_date = None
            for step_file in reversed(step_files):  # Check from later steps
                with open(step_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    
                parser = RideHistoryHTMLParser()
                parser.feed(html_content)
                
                # Check if Last 7 Days is active and get dates
                filter_active = any(status == 'active' and ('Last 7 Days' in text or 'LAST 7 DAYS' in text)
                                   for status, text in parser.filter_options)
                
                if filter_active and parser.dates_in_list:
                    # The earliest trip should be the last date in the list (chronologically oldest)
                    earliest_date = parser.dates_in_list[-1]
                    self.details['earliest_trip_date'] = earliest_date
                    self.details['all_filtered_dates'] = parser.dates_in_list
                    break
            
            if earliest_date:
                # Expected earliest date based on result.json should be Jan 16, 2026
                if 'Jan 16' in earliest_date or 'January 16' in earliest_date:
                    return True
                else:
                    self.issues.append(f"Earliest trip identified as {earliest_date}, expected Jan 16, 2026")
                    return False
            
            self.issues.append("Could not identify earliest trip in filtered list")
            return False
            
        except Exception as e:
            self.issues.append(f"Error identifying earliest trip: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Viewed details of the earliest trip"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML found")
                return False
            
            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            parser = RideHistoryHTMLParser()
            parser.feed(html_content)
            
            # Check if we're on a ride detail page showing January 16, 2026
            if parser.ride_detail_date:
                self.details['detail_page_date'] = parser.ride_detail_date
                self.details['detail_page_time'] = parser.ride_detail_time
                
                # Verify it's January 16, 2026
                if 'January 16, 2026' in parser.ride_detail_date:
                    # Also verify trip details match
                    if '12:15 AM' in html_content and 'Office Tower' in html_content and 'Coffee Street' in html_content:
                        self.details['trip_details_verified'] = 'Trip details match expected earliest trip (Jan 16, 2026)'
                        return True
                    else:
                        self.issues.append("On detail page for Jan 16 but trip details don't match")
                        return False
                else:
                    self.issues.append(f"Viewed trip details for {parser.ride_detail_date}, not January 16, 2026")
                    return False
            
            # If not on detail page, check if it's a different trip
            if 'January 21' in html_content or 'Jan 21' in html_content:
                self.issues.append("Viewed Jan 21 trip instead of earliest trip (Jan 16)")
                return False
            
            self.issues.append("Did not view trip details page")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking trip details: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_navigated_to_ride_history'] = self.check_checkpoint_1()
            self.checkpoints['cp2_filtered_last_7_days'] = self.check_checkpoint_2()
            self.checkpoints['cp3_identified_earliest_trip'] = self.check_checkpoint_3()
            self.checkpoints['cp4_viewed_trip_details'] = self.check_checkpoint_4()

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
        print("Usage: python transportation_query9.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
