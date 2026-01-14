"""
Async Event publishing functionality.
"""
import asyncio
from aworld.logs.util import logger
from typing import Dict, List, Callable, Any, Union, Tuple, Set, Coroutine
from enum import Enum

from aworld.events.async_event import AsyncEvent

# Define subscription types (same as sync version)
class AsyncSubscriptionType(Enum):
    EVENT_CODE = "event_code"
    EVENT_GROUP = "event_group"
    EVENT_ID = "event_id"
    COMBINED = "combined"  # Type for combined conditions
    ALL = "all"  # Wildcard subscription, receives all events

# Subscriber registry for async handlers
_async_subscribers: Dict[str, Any] = {
    AsyncSubscriptionType.EVENT_CODE.value: {},
    AsyncSubscriptionType.EVENT_GROUP.value: {},
    AsyncSubscriptionType.EVENT_ID.value: {},
    AsyncSubscriptionType.COMBINED.value: {},
    AsyncSubscriptionType.ALL.value: []
}


def _generate_combined_hash(conditions: Dict[str, str]) -> str:
    """
    Generate a hash string for combined conditions.
    
    Args:
        conditions: Dictionary of condition name to value
        
    Returns:
        A string hash representing the combined conditions
    """
    # Sort by keys to ensure consistent hash
    sorted_items = sorted(conditions.items())
    return ":".join(f"{k}={v}" for k, v in sorted_items)


def register_async_subscriber(
    handler: Callable[[AsyncEvent], Coroutine[Any, Any, Any]],
    subscription_type: AsyncSubscriptionType,
    value: Union[str, List[str]] = None,
    combined_conditions: Dict[str, Any] = None
) -> None:
    """
    Register an async subscriber that can subscribe by event_code, or event_group.
    
    Args:
        handler: Async function to call when an event is received
        subscription_type: Subscription type
        value: Value or list of values to subscribe to
        combined_conditions: Dictionary of combined conditions for COMBINED type
    """
    # Global subscription (receives all events)
    if subscription_type == AsyncSubscriptionType.ALL:
        _async_subscribers[AsyncSubscriptionType.ALL.value].append(handler)
        return
        
    # Combined conditions subscription
    if subscription_type == AsyncSubscriptionType.COMBINED and combined_conditions:
        # Generate hash for combined conditions
        conditions_hash = _generate_combined_hash(combined_conditions)
        combined = _async_subscribers[AsyncSubscriptionType.COMBINED.value]
        
        # If conditions hash doesn't exist, create new entry
        if conditions_hash not in combined:
            combined[conditions_hash] = {
                'conditions': combined_conditions,
                'handlers': []
            }
        
        # Add handler to combined conditions (avoid duplicates)
        if handler not in combined[conditions_hash]['handlers']:
            combined[conditions_hash]['handlers'].append(handler)
        return

    # Simple condition subscription (event_code, event_group, event_id)
    # Handle enum values
    def normalize_value(v):
        return v.name if isinstance(v, Enum) else v

    # Handle multiple values
    if isinstance(value, list):
        for v in value:
            normalized = normalize_value(v)
            _register_single_async_value(handler, subscription_type, normalized)
    else:
        normalized = normalize_value(value)
        _register_single_async_value(handler, subscription_type, normalized)


def _register_single_async_value(
    handler: Callable[[AsyncEvent], Coroutine[Any, Any, Any]],
    subscription_type: AsyncSubscriptionType,
    value: str
) -> None:
    """
    Register an async subscriber for a single value.
    
    Args:
        handler: Async handler function
        subscription_type: Subscription type
        value: Subscription value
    """
    subscriptions = _async_subscribers[subscription_type.value]
    
    if value not in subscriptions:
        subscriptions[value] = []
        
    if handler not in subscriptions[value]:
        subscriptions[value].append(handler)


def unregister_async_subscriber(
    handler: Callable[[AsyncEvent], Coroutine[Any, Any, Any]],
    subscription_type: AsyncSubscriptionType,
    value: Union[str, List[str]] = None,
    combined_conditions: Dict[str, Any] = None
) -> bool:
    """
    Unregister an async subscriber.
    
    Args:
        handler: Async function to unregister
        subscription_type: Subscription type
        value: Value or list of values to unsubscribe from
        combined_conditions: Dictionary of combined conditions for COMBINED type
        
    Returns:
        True if unregistration was successful, False otherwise
    """
    # If wildcard subscription
    if subscription_type == AsyncSubscriptionType.ALL:
        if handler in _async_subscribers[AsyncSubscriptionType.ALL.value]:
            _async_subscribers[AsyncSubscriptionType.ALL.value].remove(handler)
            return True
        return False
        
    # If combined conditions
    if subscription_type == AsyncSubscriptionType.COMBINED and combined_conditions:
        conditions_hash = _generate_combined_hash(combined_conditions)
        combined = _async_subscribers[AsyncSubscriptionType.COMBINED.value]
        
        if conditions_hash in combined and handler in combined[conditions_hash]['handlers']:
            combined[conditions_hash]['handlers'].remove(handler)
            
            # Clean up empty entries
            if not combined[conditions_hash]['handlers']:
                del combined[conditions_hash]
                
            return True
        return False

    # Handle enum values
    def normalize_value(v):
        return v.name if isinstance(v, Enum) else v
        
    # Handle multiple values
    if isinstance(value, list):
        result = False
        for v in value:
            normalized = normalize_value(v)
            if _unregister_single_async_value(handler, subscription_type, normalized):
                result = True
        return result
    else:
        normalized = normalize_value(value)
        return _unregister_single_async_value(handler, subscription_type, normalized)


def _unregister_single_async_value(
    handler: Callable[[AsyncEvent], Coroutine[Any, Any, Any]],
    subscription_type: AsyncSubscriptionType,
    value: str
) -> bool:
    """
    Unregister an async subscriber for a single value.
    
    Args:
        handler: Async handler function
        subscription_type: Subscription type
        value: Subscription value
        
    Returns:
        True if unregistration was successful, False otherwise
    """
    subscriptions = _async_subscribers[subscription_type.value]
    
    if value in subscriptions and handler in subscriptions[value]:
        subscriptions[value].remove(handler)
        # Clean up empty lists
        if not subscriptions[value]:
            del subscriptions[value]
        return True
    return False


async def publish_async_event(event: AsyncEvent) -> None:
    """
    Publish an event to all relevant async subscribers.
    
    Any handler that has subscribed to any of the following will receive the event:
    - The specific event_code
    - The event's event_group
    - The event's event_id (if specified)
    - Combined conditions that match all specified criteria
    - All events (wildcard)
    
    Args:
        event: The async event to publish
    """
    handlers_called = set()  # Track which handlers have been called to avoid duplicates
    
    # 1. Process combined conditions - subscriptions with multiple constraints
    await _call_async_handlers_for_combined_conditions(event, handlers_called)
    
    # 2. Process specific event code subscriptions
    if hasattr(event, 'event_code') and event.event_code:
        await _call_async_handlers_for_subscription(
            event, 
            AsyncSubscriptionType.EVENT_CODE, 
            event.event_code, 
            handlers_called
        )
    
    # 3. Process event group subscriptions
    if hasattr(event, 'event_group') and event.event_group:
        await _call_async_handlers_for_subscription(
            event, 
            AsyncSubscriptionType.EVENT_GROUP, 
            event.event_group, 
            handlers_called
        )
    
    # 4. Process event ID subscriptions (if needed)
    if hasattr(event, 'event_id') and event.event_id:
        await _call_async_handlers_for_subscription(
            event, 
            AsyncSubscriptionType.EVENT_ID, 
            event.event_id, 
            handlers_called
        )
    
    # 5. Process wildcard subscriptions
    for handler in _async_subscribers[AsyncSubscriptionType.ALL.value]:
        if handler not in handlers_called:  # Prevent duplicate calls
            handlers_called.add(handler)
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Error in async event handler: {str(e)}")


async def _call_async_handlers_for_subscription(
    event: AsyncEvent,
    subscription_type: AsyncSubscriptionType, 
    value: str,
    handlers_called: set
) -> None:
    """
    Call all async handlers for a specific subscription value.
    
    Args:
        event: The event being published
        subscription_type: Type of subscription
        value: Subscription value to match
        handlers_called: Set of handlers already called
    """
    subscriptions = _async_subscribers[subscription_type.value]
    
    if value in subscriptions:
        # Create tasks for all handlers
        coros = []
        for handler in subscriptions[value]:
            if handler not in handlers_called:  # Prevent duplicate calls
                handlers_called.add(handler)
                try:
                    coros.append(handler(event))
                except Exception as e:
                    logger.error(f"Error in async event handler: {str(e)}")
        
        # Execute all handlers concurrently
        if coros:
            await asyncio.gather(*coros, return_exceptions=True)


async def _call_async_handlers_for_combined_conditions(
    event: AsyncEvent,
    handlers_called: set
) -> None:
    """
    Call all async handlers for combined conditions that match the event.
    
    Args:
        event: The event being published
        handlers_called: Set of handlers already called
    """
    combined = _async_subscribers[AsyncSubscriptionType.COMBINED.value]
    
    # Check each combined condition
    coros = []
    for conditions_hash, entry in combined.items():
        conditions = entry['conditions']
        
        # Check if all conditions match
        match = True
        for condition_key, condition_value in conditions.items():
            if not hasattr(event, condition_key) or getattr(event, condition_key) != condition_value:
                match = False
                break
                
        # If all conditions match, call all handlers
        if match:
            for handler in entry['handlers']:
                if handler not in handlers_called:  # Prevent duplicate calls
                    handlers_called.add(handler)
                    try:
                        coros.append(handler(event))
                    except Exception as e:
                        logger.error(f"Error in async event handler: {str(e)}")
    
    # Execute all matching combined handlers concurrently
    if coros:
        await asyncio.gather(*coros, return_exceptions=True)


def clear_async_subscribers(
    subscription_type: AsyncSubscriptionType = None,
    value: str = None
) -> None:
    """
    Clear async subscribers.
    
    Args:
        subscription_type: Type of subscription to clear, or all if None
        value: Specific value to clear, or all values of the type if None
    """
    # If subscription_type is None, clear all
    if subscription_type is None:
        for sub_type in AsyncSubscriptionType:
            _async_subscribers[sub_type.value] = {} if sub_type != AsyncSubscriptionType.ALL else []
        return
    
    # Clear specific subscription type
    if value is None:
        # Clear all values for this type
        if subscription_type == AsyncSubscriptionType.ALL:
            _async_subscribers[subscription_type.value] = []
        else:
            _async_subscribers[subscription_type.value] = {}
    else:
        # Clear only specific value
        if subscription_type != AsyncSubscriptionType.ALL and value in _async_subscribers[subscription_type.value]:
            del _async_subscribers[subscription_type.value][value] 