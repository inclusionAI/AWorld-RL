# coding: utf-8
# Copyright (c) 2025 inclusionAI.
import os
from pickle import GLOBAL
from aworld.trace.context_manager import TraceManager, trace_configure
from aworld.logs.log import set_log_provider, get_log_provider
from aworld.logs.util import logger

trace_configure(
    backends=["logfire"],
    write_token=os.getenv("LOGFIRE_WRITE_TOKEN")
)

set_log_provider(provider="otlp", backend="logfire", write_token=os.getenv("LOGFIRE_WRITE_TOKEN"))
get_log_provider().instrumentLogging(logger)

GLOBAL_TRACE_MANAGER: TraceManager = TraceManager()
span = GLOBAL_TRACE_MANAGER.span
func_span = GLOBAL_TRACE_MANAGER.func_span
auto_tracing = GLOBAL_TRACE_MANAGER.auto_tracing
get_current_span = GLOBAL_TRACE_MANAGER.get_current_span
new_manager = GLOBAL_TRACE_MANAGER.get_current_span