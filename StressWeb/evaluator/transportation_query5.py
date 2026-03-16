#!/usr/bin/env python3
# Evaluator for transportation Query 5

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class RideStatusParser(HTMLParser):
    """Parse HTML to extract ride status and driver information."""
    def __init__(self):
        super().__init__()
        self.in_status_title = False
        self.in_driver_name = False
        self.in_button = False
        self.in_location_address = False
        self.location_index = 0
        
        self.status_title = ""
        self.driver_name = ""
        self.buttons = []
        self.pickup_location = ""
        self.dropoff_location = ""
        self.current_button_text = ""
        self.current_location = ""
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check for status title
        if tag == "h1" and attrs_dict.get("class") == "status-title":
            self.in_status_title = True
            
        # Check for driver name
        if tag == "h2" and attrs_dict.get("class") == "driver-name":
            self.in_driver_name = True
            
        # Check for buttons
        if tag == "button":
            self.in_button = True
            self.current_button_text = ""
            
        # Check for location addresses
        if tag == "span" and attrs_dict.get("class") == "location-address":
            self.in_location_address = True
            self.current_location = ""
        
    def handle_endtag(self, tag):
        if tag == "h1" and self.in_status_title:
            self.in_status_title = False
            
        if tag == "h2" and self.in_driver_name:
            self.in_driver_name = False
            
        if tag == "button" and self.in_button:
            self.in_button = False
            if self.current_button_text:
                self.buttons.append(self.current_button_text.strip())
                
        if tag == "span" and self.in_location_address:
            self.in_location_address = False
            if self.current_location:
                if self.location_index == 0:
                    self.pickup_location = self.current_location.strip()
                    self.location_index += 1
                elif self.location_index == 1:
                    self.dropoff_location = self.current_location.strip()
                    self.location_index += 1
    
    def handle_data(self, data):
        if self.in_status_title:
            self.status_title += data
            
        if self.in_driver_name:
            self.driver_name += data
            
        if self.in_button:
            self.current_button_text += data
            
        if self.in_location_address:
            self.current_location += data


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"
        
        # Define checkpoints for Query 5
        # Task: Request a Standard ride using quick fill from current location to Home, 
        # after the driver is matched, call the driver and then cancel the ride.
        self.checkpoints = {
            "cp1_ride_requested_standard": False,
            "cp2_driver_matched": False,
            "cp3_driver_called": False,
            "cp4_ride_cancelled": False,
        }
        
        self.issues = []
        self.details = {}
    
    def find_final_html(self) -> Optional[Path]:
        """Find final HTML state file."""
        final_files = list(self.result_dir.glob("final_*_raw.html"))
        if final_files:
            return final_files[0]
        step_files = sorted(self.result_dir.glob("step_*_raw.html"))
        return step_files[-1] if step_files else None
    
    def find_step_html(self, step_pattern: str) -> List[Path]:
        """Find intermediate step HTML files."""
        return sorted(self.result_dir.glob(f"{step_pattern}_raw.html"))
    
    def get_trajectory(self) -> List[Dict]:
        """Load trajectory data."""
        if not self.traj_file.exists():
            return []
        trajectory = []
        with open(self.traj_file, 'r', encoding='utf-8') as f:
            for line in f:
                trajectory.append(json.loads(line))
        return trajectory
    
    def load_result_json(self) -> Dict:
        """Load result.json."""
        with open(self.result_file, 'r') as f:
            return json.load(f)
    
    def check_checkpoint_1(self) -> bool:
        """Check if Standard ride was requested from current location to Home."""
        try:
            trajectory = self.get_trajectory()
            
            # Look for REQUEST A RIDE or CONFIRM RIDE actions
            ride_requested = False
            standard_service = False
            correct_locations = False
            
            for step in trajectory:
                action = step.get('action', {})
                selector = action.get('parameters', {}).get('selector', '')
                
                # Check for ride request or confirmation
                if 'REQUEST A RIDE' in selector or 'CONFIRM' in selector.upper():
                    ride_requested = True
                    
                # Check page URL or content for Standard service
                page_url = step.get('page_info', {}).get('url', '')
                if 'booking' in page_url:
                    standard_service = True
                    
                # Check for Home quick fill button
                if 'Home' in selector and 'button' in selector.lower():
                    correct_locations = True
            
            # Also check if we reached driver-enroute page
            for step in trajectory:
                page_url = step.get('page_info', {}).get('url', '')
                if 'driver-enroute' in page_url:
                    ride_requested = True
                    standard_service = True
                    correct_locations = True
                    break
            
            self.details['ride_requested'] = ride_requested
            self.details['standard_service'] = standard_service
            self.details['correct_locations'] = correct_locations
            
            if not ride_requested:
                self.issues.append("Ride was not requested")
            if not correct_locations:
                self.issues.append("Did not use quick fill to set Home as destination")
                
            return ride_requested and standard_service and correct_locations
            
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 1: {str(e)}")
            return False
    
    def check_checkpoint_2(self) -> bool:
        """Check if driver was matched (reached driver-enroute page)."""
        try:
            trajectory = self.get_trajectory()
            
            driver_matched = False
            driver_name = None
            
            for step in trajectory:
                page_url = step.get('page_info', {}).get('url', '')
                if 'driver-enroute' in page_url:
                    driver_matched = True
                    break
            
            # Parse HTML to find driver information
            if driver_matched:
                html_files = self.find_step_html("step_*")
                for html_file in html_files:
                    with open(html_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if 'driver-enroute' in content:
                            parser = RideStatusParser()
                            parser.feed(content)
                            if parser.driver_name:
                                driver_name = parser.driver_name
                                self.details['driver_name'] = driver_name
                                break
            
            self.details['driver_matched'] = driver_matched
            
            if not driver_matched:
                self.issues.append("Driver was not matched (did not reach driver-enroute page)")
                
            return driver_matched
            
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 2: {str(e)}")
            return False
    
    def check_checkpoint_3(self) -> bool:
        """Check if driver was called."""
        try:
            trajectory = self.get_trajectory()
            
            call_clicked = False
            
            for step in trajectory:
                action = step.get('action', {})
                selector = action.get('parameters', {}).get('selector', '')
                
                # Check for CALL button click
                if 'CALL' in selector.upper() or 'Call' in selector:
                    call_clicked = True
                    self.details['call_step'] = step.get('step_num')
                    break
            
            self.details['driver_called'] = call_clicked
            
            if not call_clicked:
                self.issues.append("Driver was not called (no CALL button click found)")
                
            return call_clicked
            
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 3: {str(e)}")
            return False
    
    def check_checkpoint_4(self) -> bool:
        """Check if ride was cancelled."""
        try:
            trajectory = self.get_trajectory()
            
            cancel_clicked = False
            
            for step in trajectory:
                action = step.get('action', {})
                selector = action.get('parameters', {}).get('selector', '')
                
                # Check for CANCEL RIDE button click
                if 'CANCEL' in selector.upper() and 'RIDE' in selector.upper():
                    cancel_clicked = True
                    self.details['cancel_step'] = step.get('step_num')
                    break
            
            # Check final state - should still show driver-enroute or back to home
            final_html = self.find_final_html()
            if final_html:
                with open(final_html, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # If still on driver-enroute page with cancel button, cancellation initiated
                    if 'Cancel Ride' in content or 'CANCEL RIDE' in content:
                        self.details['cancel_button_present'] = True
            
            self.details['ride_cancelled'] = cancel_clicked
            
            if not cancel_clicked:
                self.issues.append("Ride was not cancelled (no CANCEL RIDE button click found)")
                
            return cancel_clicked
            
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 4: {str(e)}")
            return False
    
    def evaluate(self) -> Dict:
        """Execute evaluation."""
        try:
            result_data = self.load_result_json()
            
            # Run checkpoints
            self.checkpoints['cp1_ride_requested_standard'] = self.check_checkpoint_1()
            self.checkpoints['cp2_driver_matched'] = self.check_checkpoint_2()
            self.checkpoints['cp3_driver_called'] = self.check_checkpoint_3()
            self.checkpoints['cp4_ride_cancelled'] = self.check_checkpoint_4()
            
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
        print("Usage: python transportation_query5.py <result_directory>")
        sys.exit(1)
    
    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()
    
    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
