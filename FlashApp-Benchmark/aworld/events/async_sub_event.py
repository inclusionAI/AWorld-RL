"""
Async Event subscription functionality.
"""
import functools
from typing import Callable, Any, Union, List, Optional, Dict, Coroutine

from aworld.events.async_event import AsyncEvent
from aworld.events.async_pub_event import (
    register_async_subscriber,
    unregister_async_subscriber,
    AsyncSubscriptionType
)


def sub_async_event(
    event_code: Union[str, List[str]] = None,
    event_group: Union[str, List[str]] = None,
    event_id: Union[str, List[str]] = None
):
    """
    Decorator for subscribing to async events. 
    Supports subscription by event_code, event_group, or event_id.
    
    When multiple parameters are specified, they act as AND conditions.
    For example, @sub_async_event(event_group="critical")
    will only receive events that have event_group="critical".
    
    If no parameters are specified, subscribes to all events.
    
    Args:
        event_code: Event code(s) to subscribe to
        event_group: Event group(s) to subscribe to
        event_id: Event ID(s) to subscribe to
    
    Returns:
        Decorator function that registers the async handler
    """
    def decorator(async_func: Callable[[AsyncEvent], Coroutine[Any, Any, Any]]):
        @functools.wraps(async_func)
        async def wrapper(*args, **kwargs):
            return await async_func(*args, **kwargs)
        
        # Record subscription information (for unsubscribing)
        subscriptions = []
        
        # Check if there are any event parameters
        has_event_params = any(x is not None for x in [event_code, event_group, event_id])
        
        # If no parameters: subscribe to all events
        if not has_event_params:
            register_async_subscriber(async_func, AsyncSubscriptionType.ALL)
            subscriptions.append((AsyncSubscriptionType.ALL, None, None))
        
        # If multiple event parameters: combined condition subscription
        elif sum(1 for p in [event_code, event_group, event_id] if p is not None) > 1:
            conditions = {}
            if event_code is not None:
                conditions['event_code'] = event_code if not isinstance(event_code, list) else event_code[0]
            if event_group is not None:
                conditions['event_group'] = event_group if not isinstance(event_group, list) else event_group[0]
            if event_id is not None:
                conditions['event_id'] = event_id if not isinstance(event_id, list) else event_id[0]
            
            register_async_subscriber(async_func, AsyncSubscriptionType.COMBINED, combined_conditions=conditions)
            subscriptions.append((AsyncSubscriptionType.COMBINED, None, conditions))
        
        # If only one event parameter: single condition subscription
        else:
            if event_code is not None:
                register_async_subscriber(async_func, AsyncSubscriptionType.EVENT_CODE, event_code)
                subscriptions.append((AsyncSubscriptionType.EVENT_CODE, event_code, None))
            elif event_group is not None:
                register_async_subscriber(async_func, AsyncSubscriptionType.EVENT_GROUP, event_group)
                subscriptions.append((AsyncSubscriptionType.EVENT_GROUP, event_group, None))
            elif event_id is not None:
                register_async_subscriber(async_func, AsyncSubscriptionType.EVENT_ID, event_id)
                subscriptions.append((AsyncSubscriptionType.EVENT_ID, event_id, None))
        
        wrapper._async_subscriptions = subscriptions
        return wrapper
    return decorator


def unsubscribe_async(async_func: Callable) -> bool:
    """
    Unsubscribe a decorated async function.
    
    Args:
        async_func: Async function to unsubscribe
        
    Returns:
        True if the function was unsubscribed successfully
    """
    if not hasattr(async_func, '_async_subscriptions'):
        # If function has no subscription information, can't unsubscribe
        return False
    
    result = False
    for subscription_type, value, combined_conditions in async_func._async_subscriptions:
        if subscription_type == AsyncSubscriptionType.COMBINED:
            if unregister_async_subscriber(async_func, subscription_type, combined_conditions=combined_conditions):
                result = True
        else:
            if unregister_async_subscriber(async_func, subscription_type, value):
                result = True
    
    return result 