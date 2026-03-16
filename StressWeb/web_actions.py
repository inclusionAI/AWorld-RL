


VIEWPORT_WIDTH = 1920
VIEWPORT_HEIGHT = 1080


KEYBOARD_KEYS = [
    '\t', '\n', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@',
    '[', '\\', ']', '^', '_', '`',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    '{', '|', '}', '~',
    'Alt', 'AltLeft', 'AltRight', 'Backspace', 'CapsLock', 'Control', 'ControlLeft', 'ControlRight',
    'Delete', 'ArrowDown', 'End', 'Enter', 'Escape', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8',
    'F9', 'F10', 'F11', 'F12', 'Home', 'Insert', 'ArrowLeft', 'Meta', 'MetaLeft', 'MetaRight',
    'PageDown', 'PageUp', 'ArrowRight', 'Shift', 'ShiftLeft', 'ShiftRight', 'Tab', 'ArrowUp'
]


ACTION_SPACE = [
    {
        "action_type": "NAVIGATE",
        "note": "Navigate to a URL",
        "parameters": {
            "url": {
                "type": str,
                "range": None,
                "optional": False,
                "description": "The URL to navigate to"
            }
        }
    },
    {
        "action_type": "CLICK",
        "note": "Click at the specified position or selector",
        "parameters": {
            "x": {
                "type": float,
                "range": [0, VIEWPORT_WIDTH],
                "optional": True,
                "description": "X coordinate (if selector not provided)"
            },
            "y": {
                "type": float,
                "range": [0, VIEWPORT_HEIGHT],
                "optional": True,
                "description": "Y coordinate (if selector not provided)"
            },
            "selector": {
                "type": str,
                "range": None,
                "optional": True,
                "description": "CSS selector or text to click"
            },
            "button": {
                "type": str,
                "range": ["left", "right", "middle"],
                "optional": True,
                "description": "Mouse button to click, default is 'left'"
            },
            "click_count": {
                "type": int,
                "range": [1, 2, 3],
                "optional": True,
                "description": "Number of clicks, default is 1"
            }
        }
    },
    {
        "action_type": "TYPE",
        "note": "Type text into an input field",
        "parameters": {
            "text": {
                "type": str,
                "range": None,
                "optional": False,
                "description": "Text to type"
            },
            "selector": {
                "type": str,
                "range": None,
                "optional": True,
                "description": "CSS selector of the input field (if not focused)"
            },
            "delay": {
                "type": int,
                "range": [0, 1000],
                "optional": True,
                "description": "Delay between key presses in milliseconds"
            }
        }
    },
    {
        "action_type": "PRESS",
        "note": "Press a keyboard key",
        "parameters": {
            "key": {
                "type": str,
                "range": KEYBOARD_KEYS,
                "optional": False,
                "description": "Key to press"
            }
        }
    },
    {
        "action_type": "HOTKEY",
        "note": "Press a key combination",
        "parameters": {
            "keys": {
                "type": str,
                "range": None,
                "optional": False,
                "description": "Key combination like 'Control+C' or 'Meta+V'"
            }
        }
    },
    {
        "action_type": "SCROLL",
        "note": "Scroll the page",
        "parameters": {
            "direction": {
                "type": str,
                "range": ["up", "down", "left", "right"],
                "optional": False,
                "description": "Scroll direction"
            },
            "amount": {
                "type": int,
                "range": None,
                "optional": True,
                "description": "Scroll amount in pixels, default is 100"
            },
            "selector": {
                "type": str,
                "range": None,
                "optional": True,
                "description": "CSS selector of element to scroll (if not page)"
            }
        }
    },
    {
        "action_type": "HOVER",
        "note": "Hover over an element",
        "parameters": {
            "x": {
                "type": float,
                "range": [0, VIEWPORT_WIDTH],
                "optional": True,
                "description": "X coordinate (if selector not provided)"
            },
            "y": {
                "type": float,
                "range": [0, VIEWPORT_HEIGHT],
                "optional": True,
                "description": "Y coordinate (if selector not provided)"
            },
            "selector": {
                "type": str,
                "range": None,
                "optional": True,
                "description": "CSS selector to hover over"
            }
        }
    },
    {
        "action_type": "SELECT",
        "note": "Select an option from a dropdown",
        "parameters": {
            "selector": {
                "type": str,
                "range": None,
                "optional": False,
                "description": "CSS selector of the select element"
            },
            "value": {
                "type": str,
                "range": None,
                "optional": True,
                "description": "Value to select"
            },
            "label": {
                "type": str,
                "range": None,
                "optional": True,
                "description": "Label text to select"
            },
            "index": {
                "type": int,
                "range": None,
                "optional": True,
                "description": "Index to select"
            }
        }
    },
    {
        "action_type": "FILL",
        "note": "Fill an input field (clears existing value first)",
        "parameters": {
            "selector": {
                "type": str,
                "range": None,
                "optional": False,
                "description": "CSS selector of the input field"
            },
            "text": {
                "type": str,
                "range": None,
                "optional": False,
                "description": "Text to fill"
            }
        }
    },
    {
        "action_type": "CHECK",
        "note": "Check a checkbox or radio button",
        "parameters": {
            "selector": {
                "type": str,
                "range": None,
                "optional": False,
                "description": "CSS selector of the checkbox/radio"
            }
        }
    },
    {
        "action_type": "UNCHECK",
        "note": "Uncheck a checkbox",
        "parameters": {
            "selector": {
                "type": str,
                "range": None,
                "optional": False,
                "description": "CSS selector of the checkbox"
            }
        }
    },
    {
        "action_type": "WAIT",
        "note": "Wait for a specified time or condition",
        "parameters": {
            "time": {
                "type": float,
                "range": [0, 60],
                "optional": True,
                "description": "Time to wait in seconds"
            },
            "selector": {
                "type": str,
                "range": None,
                "optional": True,
                "description": "CSS selector to wait for"
            },
            "state": {
                "type": str,
                "range": ["attached", "detached", "visible", "hidden"],
                "optional": True,
                "description": "State to wait for"
            }
        }
    },
    {
        "action_type": "GO_BACK",
        "note": "Navigate back in browser history",
        "parameters": {}
    },
    {
        "action_type": "GO_FORWARD",
        "note": "Navigate forward in browser history",
        "parameters": {}
    },
    {
        "action_type": "RELOAD",
        "note": "Reload the current page",
        "parameters": {}
    },
    {
        "action_type": "DONE",
        "note": "Mark the task as completed",
        "parameters": {}
    },
    {
        "action_type": "FAIL",
        "note": "Mark the task as failed",
        "parameters": {
            "reason": {
                "type": str,
                "range": None,
                "optional": True,
                "description": "Reason for failure"
            }
        }
    }
]


def get_action_space():
    
    return ACTION_SPACE


def get_action_description():
    
    descriptions = []
    for action in ACTION_SPACE:
        desc = f"{action['action_type']}: {action['note']}"
        if action.get('parameters'):
            params = []
            for param_name, param_info in action['parameters'].items():
                optional_str = " (optional)" if param_info.get('optional', False) else ""
                params.append(f"  - {param_name}{optional_str}: {param_info.get('description', '')}")
            if params:
                desc += "\n" + "\n".join(params)
        descriptions.append(desc)
    return "\n\n".join(descriptions)
