#!/usr/bin/env python3
# Evaluator for reservation Query 17

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class ReservationHTMLParser(HTMLParser):
    """Parse HTML to extract reservation and restaurant information."""

    def __init__(self):
        super().__init__()
        self.in_booking_number = False
        self.in_detail_value = False
        self.in_detail_label = False
        self.current_label = ""
        self.booking_number = None
        self.restaurant_name = None
        self.reservation_date = None
        self.reservation_time = None
        self.party_size = None
        self.occasion = None
        self.special_request = None
        self.seating_preferences = []
        self.capture_next_value = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        if tag == "h2" and attrs_dict.get("class") == "booking-number":
            self.in_booking_number = True
        elif tag == "span" and "detail-label" in attrs_dict.get("class", ""):
            self.in_detail_label = True
            self.current_label = ""
        elif tag == "span" and "detail-value" in attrs_dict.get("class", ""):
            self.in_detail_value = True

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
            
        if self.in_booking_number:
            # Extract booking number like "Booking #BK8202806286"
            match = re.search(r'#(BK\d+)', data)
            if match:
                self.booking_number = match.group(1)
            self.in_booking_number = False
            
        elif self.in_detail_label:
            self.current_label = data
            self.in_detail_label = False
            self.capture_next_value = True
            
        elif self.in_detail_value and self.capture_next_value:
            if self.current_label == "Name":
                self.restaurant_name = data
            elif "Date" in self.current_label or "📅" in self.current_label:
                self.reservation_date = data
            elif "Time" in self.current_label or "🕐" in self.current_label:
                self.reservation_time = data
            elif "Party Size" in self.current_label or "👥" in self.current_label:
                # Extract number from "4 guests"
                match = re.search(r'(\d+)', data)
                if match:
                    self.party_size = int(match.group(1))
            self.capture_next_value = False
            self.in_detail_value = False

    def handle_endtag(self, tag):
        if tag == "span":
            self.in_detail_value = False
            self.in_detail_label = False


class FavoritesHTMLParser(HTMLParser):
    """Parse HTML to check if restaurant is in favorites."""

    def __init__(self):
        super().__init__()
        self.favorited_restaurants = []
        self.in_restaurant_card = False
        self.in_restaurant_name = False
        self.capture_name = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check for favorited button or saved status
        if tag == "button":
            classes = attrs_dict.get("class", "")
            if "favorited" in classes.lower() or "saved" in classes.lower():
                self.in_restaurant_card = True

    def handle_data(self, data):
        data = data.strip()
        if data and self.in_restaurant_card:
            # Restaurant names are typically in title or heading elements
            if data and len(data) > 3:
                self.favorited_restaurants.append(data)
                self.in_restaurant_card = False


class SearchResultsParser(HTMLParser):
    """Parse search results page to find restaurant details."""

    def __init__(self):
        super().__init__()
        self.restaurants = []
        self.current_restaurant = {}
        self.in_restaurant_name = False
        self.in_rating = False
        self.in_cuisine = False
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        classes = attrs_dict.get("class", "")
        
        if "restaurant-name" in classes or "card-title" in classes:
            self.in_restaurant_name = True
        elif "rating" in classes:
            self.in_rating = True
        elif "cuisine" in classes:
            self.in_cuisine = True

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
            
        if self.in_restaurant_name:
            self.current_restaurant["name"] = data
            self.in_restaurant_name = False
        elif self.in_rating:
            # Extract rating like "4.6" or "4.6 ★"
            match = re.search(r'(\d+\.?\d*)', data)
            if match:
                self.current_restaurant["rating"] = float(match.group(1))
            self.in_rating = False
        elif self.in_cuisine:
            self.current_restaurant["cuisine"] = data
            self.in_cuisine = False
            
        # If we have name and rating, save this restaurant
        if "name" in self.current_restaurant and "rating" in self.current_restaurant:
            self.restaurants.append(self.current_restaurant.copy())
            self.current_restaurant = {}


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task
        self.checkpoints = {
            "cp1_search_italian_wheelchair": False,
            "cp2_sorted_by_rating": False,
            "cp3_viewed_top_rated_detail": False,
            "cp4_added_to_favorites": False,
            "cp5_reservation_confirmed": False,
            "cp6_correct_reservation_details": False,
            "cp7_special_request_included": False,
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
        """Checkpoint 1: Searched for Italian restaurants with wheelchair accessible filter"""
        try:
            trajectory = self.get_trajectory()
            
            # Check for Italian search in trajectory
            italian_searched = False
            wheelchair_checked = False
            
            for entry in trajectory:
                action = entry.get("action", {})
                params = action.get("parameters", {})
                
                # Check for Italian search
                if action.get("action_type") == "FILL":
                    text = params.get("text", "")
                    if "italian" in text.lower():
                        italian_searched = True
                        self.details["italian_search"] = text
                
                # Check for wheelchair accessible checkbox
                if action.get("action_type") in ["CHECK", "CLICK"]:
                    selector = params.get("selector", "")
                    if "wheelchair" in selector.lower():
                        wheelchair_checked = True
                        self.details["wheelchair_filter"] = True
            
            if not italian_searched:
                self.issues.append("Did not search for 'Italian' restaurants")
            if not wheelchair_checked:
                self.issues.append("Did not apply 'wheelchair accessible' filter")
                
            return italian_searched and wheelchair_checked
            
        except Exception as e:
            self.issues.append(f"Error checking search filters: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Results sorted by rating"""
        try:
            trajectory = self.get_trajectory()
            
            # Check for sorting action in trajectory
            for entry in trajectory:
                action = entry.get("action", {})
                reasoning = action.get("reasoning", "").lower()
                
                # Check if sorted by rating in reasoning or action
                if "sort" in reasoning and "rating" in reasoning:
                    self.details["sorted_by"] = "rating"
                    return True
                    
                # Check for search action that implies sorting
                if action.get("action_type") == "CLICK":
                    selector = action.get("parameters", {}).get("selector", "").lower()
                    if "sort" in selector or "rating" in selector:
                        self.details["sorted_by"] = "rating"
                        return True
            
            # Check in step HTML files for sort dropdown
            search_steps = self.find_step_html("step_2[0-9]*")
            for step_file in search_steps:
                with open(step_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if "highest rating" in content.lower() or "rating: high to low" in content.lower():
                        self.details["sorted_by"] = "rating"
                        return True
            
            self.issues.append("Results not sorted by rating")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking sorting: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Viewed top-rated Italian restaurant's detail"""
        try:
            trajectory = self.get_trajectory()
            
            # Look for "View Details" click after search
            viewed_details = False
            restaurant_name = None
            
            for i, entry in enumerate(trajectory):
                action = entry.get("action", {})
                params = action.get("parameters", {})
                reasoning = action.get("reasoning", "").lower()
                
                if action.get("action_type") == "CLICK":
                    selector = params.get("selector", "")
                    if "view details" in selector.lower():
                        viewed_details = True
                        # Check reasoning for restaurant name
                        if "bella" in reasoning:
                            restaurant_name = "Bella Italia"
                        self.details["viewed_restaurant"] = restaurant_name or "Top result"
                        break
            
            if not viewed_details:
                self.issues.append("Did not view restaurant details")
                return False
                
            return True
            
        except Exception as e:
            self.issues.append(f"Error checking restaurant detail view: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Added restaurant to favorites"""
        try:
            trajectory = self.get_trajectory()
            
            # Check for Save/Favorite action
            added_to_favorites = False
            
            for entry in trajectory:
                action = entry.get("action", {})
                params = action.get("parameters", {})
                
                if action.get("action_type") == "CLICK":
                    selector = params.get("selector", "")
                    if "save" in selector.lower() or "favorite" in selector.lower():
                        added_to_favorites = True
                        self.details["favorited"] = True
                        break
            
            if not added_to_favorites:
                self.issues.append("Did not add restaurant to favorites")
                return False
                
            return True
            
        except Exception as e:
            self.issues.append(f"Error checking favorites: {str(e)}")
            return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Reservation confirmed"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML file found")
                return False

            with open(final_html, 'r', encoding='utf-8') as f:
                content = f.read()
                
            parser = ReservationHTMLParser()
            parser.feed(content)
            
            if parser.booking_number:
                self.details["booking_number"] = parser.booking_number
                return True
            else:
                # Check for confirmation text
                if "reservation confirmed" in content.lower() or "booking confirmed" in content.lower():
                    self.details["confirmation_found"] = True
                    return True
                    
                self.issues.append("No booking confirmation found")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking reservation confirmation: {str(e)}")
            return False

    def check_checkpoint_6(self) -> bool:
        """Checkpoint 6: Correct reservation details (4 people, March 1 2026, 6:30 PM)"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML file found")
                return False

            with open(final_html, 'r', encoding='utf-8') as f:
                content = f.read()
                
            parser = ReservationHTMLParser()
            parser.feed(content)
            
            # Check party size
            correct_party_size = False
            if parser.party_size == 4:
                correct_party_size = True
                self.details["party_size"] = parser.party_size
            else:
                self.issues.append(f"Wrong party size: expected 4, got {parser.party_size}")
            
            # Check date (March 1, 2026)
            correct_date = False
            if parser.reservation_date:
                # Accept formats like "2026/03/01", "03/01/2026", "March 1, 2026"
                if "2026" in parser.reservation_date and ("03" in parser.reservation_date or "3" in parser.reservation_date or "march" in parser.reservation_date.lower()):
                    if "01" in parser.reservation_date or "/1" in parser.reservation_date or " 1" in parser.reservation_date:
                        correct_date = True
                        self.details["reservation_date"] = parser.reservation_date
                else:
                    self.issues.append(f"Wrong date: expected March 1, 2026, got {parser.reservation_date}")
            else:
                self.issues.append("Reservation date not found")
            
            # Check time (6:30 PM)
            correct_time = False
            if parser.reservation_time:
                # Accept formats like "6:30 pm", "6:30 PM", "18:30"
                if "6:30" in parser.reservation_time or "18:30" in parser.reservation_time:
                    correct_time = True
                    self.details["reservation_time"] = parser.reservation_time
                else:
                    self.issues.append(f"Wrong time: expected 6:30 PM, got {parser.reservation_time}")
            else:
                self.issues.append("Reservation time not found")
            
            # Check restaurant name
            if parser.restaurant_name:
                self.details["restaurant_name"] = parser.restaurant_name
            
            return correct_party_size and correct_date and correct_time
            
        except Exception as e:
            self.issues.append(f"Error checking reservation details: {str(e)}")
            return False

    def check_checkpoint_7(self) -> bool:
        """Checkpoint 7: Special request included ('Anniversary celebration, prefer quiet corner table')"""
        try:
            # Check in step HTML files before final confirmation for the special request
            reservation_steps = self.find_step_html("step_5[0-9]*")
            
            special_request_found = False
            
            for step_file in reservation_steps:
                with open(step_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Look for the special request text
                if "anniversary celebration" in content.lower() and "quiet corner" in content.lower():
                    special_request_found = True
                    self.details["special_request"] = "Anniversary celebration, prefer quiet corner table"
                    break
            
            # Also check trajectory for the special request input
            if not special_request_found:
                trajectory = self.get_trajectory()
                
                for entry in trajectory:
                    action = entry.get("action", {})
                    params = action.get("parameters", {})
                    
                    if action.get("action_type") in ["FILL", "TYPE"]:
                        text = params.get("text", "")
                        if "anniversary" in text.lower() and "quiet" in text.lower():
                            special_request_found = True
                            self.details["special_request"] = text
                            break
            
            if not special_request_found:
                self.issues.append("Special request 'Anniversary celebration, prefer quiet corner table' not included")
                return False
                
            return True
            
        except Exception as e:
            self.issues.append(f"Error checking special request: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_search_italian_wheelchair'] = self.check_checkpoint_1()
            self.checkpoints['cp2_sorted_by_rating'] = self.check_checkpoint_2()
            self.checkpoints['cp3_viewed_top_rated_detail'] = self.check_checkpoint_3()
            self.checkpoints['cp4_added_to_favorites'] = self.check_checkpoint_4()
            self.checkpoints['cp5_reservation_confirmed'] = self.check_checkpoint_5()
            self.checkpoints['cp6_correct_reservation_details'] = self.check_checkpoint_6()
            self.checkpoints['cp7_special_request_included'] = self.check_checkpoint_7()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 17,
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
                'query_id': 17,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python reservation_query17.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()
