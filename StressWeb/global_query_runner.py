


import argparse
import asyncio
import datetime
import json
import logging
import multiprocessing as mp
import os
import queue
import re
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)




PROJECT_CONFIGS = [
    {
        "base_dir": "./websites/PRD_20260126_155324_ecom",
        "query_file": "./queries/ecom_query.md",
    },
    {
        "base_dir": "./websites/PRD_20260115_175319_note",
        "query_file": "./queries/note_query.md",
    },
    {
        "base_dir": "./websites/PRD_20260128_190659_reservation",
        "query_file": "./queries/reservation_query.md",
    },
    {
        "base_dir": "./websites/PRD_20260123_171546_food",
        "query_file": "./queries/food_query.md",
    },
    {
        "base_dir": "./websites/PRD_20260128_221759_calendar",
        "query_file": "./queries/calendar_query.md",
    },
    {
        "base_dir": "./websites/PRD_20260129_035820_file",
        "query_file": "./queries/file_query.md",
    },
    {
        "base_dir": "./websites/PRD_20260128_233231_transportation",
        "query_file": "./queries/transportation_query.md",
    },
    {
        "base_dir": "./websites/PRD_20260128_205609_network",
        "query_file": "./queries/network_query.md",
    },
    {
        "base_dir": "./websites/PRD_20260112_174927_email",
        "query_file": "./queries/email_query.md",
    },
    {
        "base_dir": "./websites/PRD_20260128_185040_management",
        "query_file": "./queries/management_query.md",
    },
]

ALL_MODES = ["clean", "chaos", "failed", "perturbed", "semanticE", "semanticS", "dom"]


MODEL_GROUPS = {
    "g1": [
        "gemini-2.5-flash",
    ],
    "g2": [
        "gpt-5.2",
        "claude-opus-4-5-20251124",
    ],
}

from batch_run_queries import (
    extract_pname,
    get_model_short_name,
    get_website_config_for_mode,
    load_queries_from_file,
    _is_env_startup_error,
)

RESULTS_DIR = os.path.join(current_dir, "test_results")


def find_all_batch_dirs(pname: str, model_short: str, mode: str) -> List[str]:
    
    pattern = f"batch_*_{pname}_{model_short}_{mode}"
    import glob
    matches = sorted(glob.glob(os.path.join(RESULTS_DIR, pattern)), reverse=True)
    return matches


def find_batch_result_dir(pname: str, model_short: str, mode: str) -> Optional[str]:
    
    dirs = find_all_batch_dirs(pname, model_short, mode)
    return dirs[0] if dirs else None


def _scan_completed_query_results(pname: str, model_short: str, mode: str) -> Dict[int, Dict]:
    
    all_dirs = find_all_batch_dirs(pname, model_short, mode)
    completed = {}  
    
    for batch_dir in all_dirs:
        
        try:
            for entry in os.listdir(batch_dir):
                if not entry.startswith("query_"):
                    continue
                try:
                    qid = int(entry.split("_", 1)[1])
                except (ValueError, IndexError):
                    continue
                
                result_file = os.path.join(batch_dir, entry, "result.json")
                if not os.path.isfile(result_file):
                    continue
                
                try:
                    with open(result_file, "r", encoding="utf-8") as f:
                        result = json.load(f)
                    
                    
                    mtime = os.path.getmtime(result_file)
                    if qid not in completed or mtime > completed[qid][0]:
                        completed[qid] = (mtime, result)
                except Exception:
                    continue
        except OSError:
            continue
    
    
    return {qid: result for qid, (_, result) in completed.items()}


def check_batch_complete(pname: str, model_short: str, mode: str, total_queries: int = 0) -> bool:
    
    completed = _scan_completed_query_results(pname, model_short, mode)
    if not completed:
        return False
    
    if total_queries > 0:
        
        return all(qid in completed for qid in range(1, total_queries + 1))
    
    
    latest_dir = find_batch_result_dir(pname, model_short, mode)
    if not latest_dir:
        return False
    summary_file = os.path.join(latest_dir, "batch_summary.json")
    if not os.path.isfile(summary_file):
        return False
    try:
        with open(summary_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        failed = data.get("failed")
        if failed is not None:
            return failed == 0
        return str(data.get("success_rate", "")).startswith("100")
    except Exception:
        return False


def get_completed_query_ids(pname: str, model_short: str, mode: str) -> set:
    
    completed = _scan_completed_query_results(pname, model_short, mode)
    return set(completed.keys())

def _global_query_worker(
    task_queue: "mp.Queue",
    result_queue: "mp.Queue",
    worker_id: int,
    max_steps: int,
    use_dom: bool,
    headless: bool,
    detailed_log_path: str = None,
):
    
    import atexit
    import psutil

    
    log_handler = None
    if detailed_log_path:
        log_handler = logging.FileHandler(detailed_log_path, encoding="utf-8")
        log_handler.setLevel(logging.DEBUG)
        log_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(log_handler)

        
        def cleanup_log_handler():
            try:
                if log_handler:
                    log_handler.close()
                    logging.getLogger().removeHandler(log_handler)
            except:
                pass
        atexit.register(cleanup_log_handler)

    
    def cleanup_browser_processes():
        
        try:
            current_process = psutil.Process()
            children = current_process.children(recursive=True)
            for child in children:
                try:
                    name = child.name().lower()
                    if 'chrome' in name or 'chromium' in name or 'playwright' in name:
                        logger.debug(f"[Worker-{worker_id}] Killing leaked browser process: {child.pid}")
                        child.kill()
                        child.wait(timeout=3)
                except:
                    pass
        except:
            pass
    atexit.register(cleanup_browser_processes)

    from batch_run_queries import run_single_query_with_isolated_env, _is_env_startup_error

    while True:
        task = task_queue.get()
        if task is None:
            break

        task_info = task  
        query_id = task_info["query_id"]
        query = task_info["query"]
        website_path = task_info["website_path"]
        model = task_info["model"]
        batch_result_dir = task_info["batch_result_dir"]
        dom_filter_selectors = task_info.get("dom_filter_selectors")
        dom_perturbation_modes = task_info.get("dom_perturbation_modes")
        pname = task_info["pname"]
        model_short = task_info["model_short"]
        mode = task_info["mode"]

        task_label = f"{pname}/{model_short}/{mode}/Q{query_id}"
        logger.info(f"[Worker-{worker_id}] ▶️  Starting: {task_label}")
        
        result_queue.put({
            "_event": "task_started",
            "worker_id": worker_id,
            "query_id": query_id,
            "pname": pname,
            "model_short": model_short,
            "mode": mode,
            "started_at": time.time(),
        })

        max_start_retries = 2
        attempt = 0
        while True:
            attempt += 1
            
            loop = None
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                result = loop.run_until_complete(
                    run_single_query_with_isolated_env(
                        query_id=query_id,
                        query=query,
                        website_path=website_path,
                        model=model,
                        max_steps=max_steps,
                        use_dom=use_dom,
                        batch_result_dir=batch_result_dir,
                        dom_filter_selectors=dom_filter_selectors,
                        dom_perturbation_modes=dom_perturbation_modes,
                        headless=headless,
                        worker_id=worker_id,
                    )
                )

                
                result["pname"] = pname
                result["model_short"] = model_short
                result["mode"] = mode
                result["model_full"] = model

                error_hint = f"{result.get('reason', '')} {result.get('error', '')}"
                if not result.get("success", False) and _is_env_startup_error(error_hint) and attempt <= max_start_retries:
                    logger.warning(f"[Worker-{worker_id}] {task_label} startup failed, retrying {attempt}/{max_start_retries}...")
                    time.sleep(2)
                    continue

                status = "✅" if result.get("success") else "❌"
                logger.info(f"[Worker-{worker_id}] {status} Done: {task_label} ({result.get('execution_time', 0):.1f}s)")
                result_queue.put(result)
                break

            except Exception as e:
                error_msg = str(e)
                if _is_env_startup_error(error_msg) and attempt <= max_start_retries:
                    logger.warning(f"[Worker-{worker_id}] {task_label} startup exception, retrying {attempt}/{max_start_retries}...")
                    time.sleep(2)
                    continue

                logger.error(f"[Worker-{worker_id}] ❌ Exception: {task_label}: {error_msg}")
                result_queue.put({
                    "query_id": query_id,
                    "query": query,
                    "success": False,
                    "reason": f"Exception: {error_msg}",
                    "steps": 0,
                    "execution_time": 0,
                    "error": error_msg,
                    "worker_id": worker_id,
                    "pname": pname,
                    "model_short": model_short,
                    "mode": mode,
                    "model_full": model,
                })
                break

            finally:
                
                if loop is not None:
                    try:
                        
                        pending = asyncio.all_tasks(loop)
                        for task in pending:
                            task.cancel()
                        if pending:
                            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                    except Exception as cleanup_error:
                        logger.debug(f"[Worker-{worker_id}] Loop cleanup warning: {cleanup_error}")
                    finally:
                        try:
                            loop.close()
                        except:
                            pass


def aggregate_results(
    all_results: List[Dict[str, Any]],
    batch_dirs: Dict[str, str],
    total_time: float,
    total_queries_map: Dict[str, int] = None,
):
    
    from batch_run_queries import generate_summary

    if total_queries_map is None:
        total_queries_map = {}

    
    grouped = {}
    for r in all_results:
        key = (r["pname"], r["model_short"], r["mode"])
        grouped.setdefault(key, []).append(r)

    summaries = {}

    
    all_group_keys = set(batch_dirs.keys())
    for r in all_results:
        all_group_keys.add(f"{r['pname']}_{r['model_short']}_{r['mode']}")

    for group_key in sorted(all_group_keys):
        batch_result_dir = batch_dirs.get(group_key)
        if not batch_result_dir:
            continue

        
        
        
        parts_key = (None, None, None)
        new_results = grouped.get(None, [])  
        for key, results in grouped.items():
            if f"{key[0]}_{key[1]}_{key[2]}" == group_key:
                parts_key = key
                new_results = results
                break
        
        if parts_key[0] is None:
            
            config_file = os.path.join(batch_result_dir, "batch_config.json")
            if os.path.isfile(config_file):
                try:
                    with open(config_file, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                    parts_key = (cfg["pname"], cfg["model_short"], cfg["mode"])
                    new_results = []
                except Exception:
                    continue
            else:
                continue

        pname, model_short, mode = parts_key

        
        historical_completed = _scan_completed_query_results(pname, model_short, mode)

        
        merged = dict(historical_completed)  
        for r in new_results:
            qid = r.get("query_id")
            if qid is not None:
                merged[qid] = r

        merged_results = sorted(merged.values(), key=lambda r: r.get("query_id", 0))
        
        if not merged_results:
            continue

        group_time = sum(r.get("execution_time", 0) for r in new_results) if new_results else 0
        merged_summary = generate_summary(merged_results, group_time)
        
        
        total_q = total_queries_map.get(group_key, 0)
        if total_q > 0:
            merged_summary["expected_total_queries"] = total_q
            merged_summary["coverage"] = f"{len(merged_results)}/{total_q}"
        
        merged_summary["pname"] = pname
        merged_summary["model_short"] = model_short
        merged_summary["mode"] = mode
        
        wp = ""
        if new_results:
            wp = new_results[0].get("website_path", "")
        if not wp:
            config_file = os.path.join(batch_result_dir, "batch_config.json")
            if os.path.isfile(config_file):
                try:
                    with open(config_file, "r", encoding="utf-8") as f:
                        wp = json.load(f).get("website_path", "")
                except Exception:
                    pass
        merged_summary["website_path"] = wp

        summary_file = os.path.join(batch_result_dir, "batch_summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(merged_summary, f, indent=2, ensure_ascii=False)

        summaries[group_key] = merged_summary
        logger.info(f"📊 {group_key}: {merged_summary['success_rate']} ({merged_summary['successful']}/{merged_summary['total_queries']})")

    return summaries





def build_task_list(
    models: List[str],
    modes: List[str],
    projects: List[Dict],
    max_steps: int,
    skip_completed_batches: bool = True,
    skip_completed_queries: bool = True,
    query_ids_filter: List[int] = None,
) -> Tuple[List[Dict], Dict[str, str], int, int, Dict[str, int]]:
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tasks = []
    batch_dirs = {}  
    total_queries_map = {}  
    total_skipped_batches = 0
    total_skipped_queries = 0

    for model in models:
        model_short = get_model_short_name(model)

        for proj in projects:
            base_dir = proj["base_dir"]
            query_file = proj["query_file"]
            pname = extract_pname(base_dir)

            
            queries = load_queries_from_file(query_file)
            if not queries:
                logger.warning(f"⚠️  No queries loaded from {query_file}, skipping {pname}")
                continue

            for mode in modes:
                group_key = f"{pname}_{model_short}_{mode}"
                total_queries_map[group_key] = len(queries)

                
                if skip_completed_batches and check_batch_complete(pname, model_short, mode, total_queries=len(queries)):
                    total_skipped_batches += 1
                    logger.info(f"⏭️  Skip (batch complete): {group_key}")
                    continue

                
                try:
                    website_path, filters, perturbations, description = get_website_config_for_mode(base_dir, mode)
                except (ValueError, FileNotFoundError) as e:
                    logger.warning(f"⚠️  Skip {group_key}: {e}")
                    continue

                
                completed_qids = set()
                if skip_completed_queries:
                    completed_qids = get_completed_query_ids(pname, model_short, mode)
                    if completed_qids:
                        logger.info(f"   📋 {group_key}: {len(completed_qids)} queries already completed, will skip them")

                
                group_tasks = []
                for i, query in enumerate(queries):
                    query_id = i + 1

                    
                    if query_ids_filter and query_id not in query_ids_filter:
                        continue

                    
                    if query_id in completed_qids:
                        total_skipped_queries += 1
                        continue

                    group_tasks.append({
                        "query_id": query_id,
                        "query": query,
                        "website_path": website_path,
                        "model": model,
                        "model_short": model_short,
                        "pname": pname,
                        "mode": mode,
                        "batch_result_dir": None,  
                        "dom_filter_selectors": filters,
                        "dom_perturbation_modes": perturbations,
                    })

                if not group_tasks:
                    
                    logger.info(f"⏭️  Skip (all queries done): {group_key}")
                    total_skipped_batches += 1
                    continue

                
                existing_dir = find_batch_result_dir(pname, model_short, mode)
                if existing_dir:
                    batch_result_dir = existing_dir
                    logger.info(f"   📂 Reusing existing dir: {os.path.basename(existing_dir)}")
                else:
                    batch_result_dir = os.path.join(
                        RESULTS_DIR, f"batch_{timestamp}_{group_key}"
                    )
                    os.makedirs(batch_result_dir, exist_ok=True)
                batch_dirs[group_key] = batch_result_dir

                
                batch_config = {
                    "website_path": str(website_path),
                    "pname": pname,
                    "model": model,
                    "model_short": model_short,
                    "mode": mode,
                    "max_steps": max_steps,
                    "use_dom": True,
                    "dom_filter_selectors": filters or [],
                    "dom_perturbation_modes": perturbations or [],
                    "total_queries": len(queries),
                    "skipped_queries": sorted(completed_qids),
                    "new_queries": [t["query_id"] for t in group_tasks],
                    "scheduler": "global_query_runner",
                }
                config_file = os.path.join(batch_result_dir, "batch_config.json")
                with open(config_file, "w", encoding="utf-8") as f:
                    json.dump(batch_config, f, indent=2)

                
                for t in group_tasks:
                    t["batch_result_dir"] = batch_result_dir
                tasks.extend(group_tasks)

    return tasks, batch_dirs, total_skipped_batches, total_skipped_queries, total_queries_map


def run_global_scheduler(
    tasks: List[Dict],
    batch_dirs: Dict[str, str],
    max_workers: int,
    max_steps: int,
    use_dom: bool = True,
    headless: bool = True,
    total_queries_map: Dict[str, int] = None,
    detailed_log_path: str = None,
):
    
    if not tasks:
        logger.info("✅ No tasks to run!")
        return {}

    total_tasks = len(tasks)
    logger.info(f"\n{'='*80}")
    logger.info(f"🚀 Global Query Scheduler")
    logger.info(f"📊 Total tasks: {total_tasks}")
    logger.info(f"👷 Max workers: {max_workers}")
    logger.info(f"{'='*80}\n")

    
    distribution = {}
    for t in tasks:
        key = f"{t['pname']}/{t['model_short']}/{t['mode']}"
        distribution[key] = distribution.get(key, 0) + 1
    logger.info("📋 Task distribution:")
    for key in sorted(distribution.keys()):
        logger.info(f"   {key}: {distribution[key]} queries")
    logger.info("")

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()

    for task in tasks:
        task_queue.put(task)

    
    for _ in range(max_workers):
        task_queue.put(None)

    
    start_time = time.time()
    processes = []
    workers = {}
    for wid in range(1, max_workers + 1):
        p = ctx.Process(
            target=_global_query_worker,
            args=(task_queue, result_queue, wid, max_steps, use_dom, headless, detailed_log_path),
        )
        p.start()
        processes.append(p)
        workers[wid] = p

    
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    progress_log_path = os.path.join(RESULTS_DIR, f"progress_{ts}.log")
    progress_fh = None
    all_results = []
    completed = 0
    success_count = 0
    worker_inflight = {}  
    max_idle_seconds = 900  
    last_result_time = time.time()
    try:
        progress_fh = open(progress_log_path, "w", encoding="utf-8")
        progress_fh.write(f"# Global Query Runner Progress Log\n")
        progress_fh.write(f"# Started: {datetime.datetime.now().isoformat()}\n")
        progress_fh.write(f"# Total tasks: {total_tasks}, Workers: {max_workers}\n")
        progress_fh.write(f"# Format: [timestamp] status completed/total pname/model/mode/Q#\n\n")
        progress_fh.flush()
        logger.info(f"📝 Progress log: {progress_log_path}")

        
        while completed < total_tasks:
            try:
                result = result_queue.get(timeout=15)
            except queue.Empty:
                dead_workers = []
                for wid, proc in workers.items():
                    if not proc.is_alive() and wid in worker_inflight:
                        dead_workers.append(wid)

                for wid in dead_workers:
                    started = worker_inflight.pop(wid)
                    synthetic = {
                        "query_id": started.get("query_id"),
                        "query": started.get("query", ""),
                        "success": False,
                        "reason": "Worker exited unexpectedly while task was running",
                        "steps": 0,
                        "execution_time": max(0.0, time.time() - started.get("started_at", time.time())),
                        "error": "worker_process_exited",
                        "worker_id": wid,
                        "pname": started.get("pname"),
                        "model_short": started.get("model_short"),
                        "mode": started.get("mode"),
                        "model_full": started.get("model_full"),
                        "website_path": started.get("website_path", ""),
                    }
                    all_results.append(synthetic)
                    completed += 1
                    elapsed = time.time() - start_time
                    rate = completed / elapsed * 3600 if elapsed > 0 else 0
                    status = "❌"
                    task_label = f"{synthetic.get('pname','?')}/{synthetic.get('model_short','?')}/{synthetic.get('mode','?')}/Q{synthetic.get('query_id','?')}"
                    now_str = datetime.datetime.now().strftime("%H:%M:%S")
                    progress_fh.write(f"[{now_str}] {status} {completed}/{total_tasks} {task_label}\n")
                    progress_fh.flush()
                    logger.error(f"[Worker-{wid}] ❌ Marked failed due to unexpected worker exit: {task_label}")
                    logger.info(
                        f"📈 Progress: {completed}/{total_tasks} "
                        f"(✅{success_count} ❌{completed - success_count}) "
                        f"| Elapsed: {elapsed:.0f}s | Rate: {rate:.1f} queries/hr"
                    )

                alive_count = sum(1 for proc in workers.values() if proc.is_alive())
                if alive_count == 0:
                    
                    for wid, proc in workers.items():
                        logger.error(f"   Worker-{wid} exit code: {proc.exitcode}")
                    remaining = total_tasks - completed
                    logger.error(f"❌ All workers have exited! {remaining} tasks were never processed. Stopping collection.")
                    
                    progress_fh.write(f"# ⚠️ All workers exited unexpectedly, {remaining} tasks unprocessed\n")
                    progress_fh.flush()
                    break

                idle_seconds = time.time() - last_result_time
                
                
                inflight_count = len(worker_inflight)
                if idle_seconds >= max_idle_seconds and inflight_count == 0 and alive_count > 0:
                    
                    logger.error(f"❌ No results for {idle_seconds:.0f}s and no inflight tasks. Possible deadlock, stopping.")
                    logger.error(f"   completed={completed}/{total_tasks}, alive_workers={alive_count}")
                    for wid, proc in workers.items():
                        status = 'alive' if proc.is_alive() else f'exited({proc.exitcode})'
                        logger.error(f"   Worker-{wid}: {status}")
                    break
                elif inflight_count > 0 and idle_seconds >= 60:
                    
                    if int(idle_seconds) % 120 < 16:  
                        logger.info(f"⏳ Waiting for {inflight_count} inflight tasks... ({idle_seconds:.0f}s since last result, {alive_count} workers alive)")
                        for wid, started in worker_inflight.items():
                            task_label = f"{started.get('pname','?')}/{started.get('model_short','?')}/{started.get('mode','?')}/Q{started.get('query_id','?')}"
                            elapsed_task = time.time() - started.get("started_at", time.time())
                            logger.info(f"   Worker-{wid}: {task_label} ({elapsed_task:.0f}s)")

                continue

            last_result_time = time.time()

            if result.get("_event") == "task_started":
                
                worker_inflight[result.get("worker_id")] = {
                    "query_id": result.get("query_id"),
                    "pname": result.get("pname"),
                    "model_short": result.get("model_short"),
                    "mode": result.get("mode"),
                    "started_at": result.get("started_at", time.time()),
                }

                
                if len(worker_inflight) > max_workers * 2:
                    now = time.time()
                    stale_workers = [
                        wid for wid, info in worker_inflight.items()
                        if now - info.get("started_at", now) > 7200  
                    ]
                    for wid in stale_workers:
                        stale_info = worker_inflight.pop(wid, None)
                        if stale_info:
                            logger.warning(
                                f"⚠️  Cleaned stale inflight task from Worker-{wid}: "
                                f"{stale_info.get('pname')}/{stale_info.get('model_short')}/"
                                f"{stale_info.get('mode')}/Q{stale_info.get('query_id')}"
                            )

                task_label = f"{result.get('pname','?')}/{result.get('model_short','?')}/{result.get('mode','?')}/Q{result.get('query_id','?')}"
                logger.debug(f"[Worker-{result.get('worker_id')}] 📥 Task started: {task_label}")
                continue

            worker_inflight.pop(result.get("worker_id"), None)
            all_results.append(result)
            completed += 1
            if result.get("success"):
                success_count += 1
            elapsed = time.time() - start_time
            rate = completed / elapsed * 3600 if elapsed > 0 else 0

            
            status = "✅" if result.get("success") else "❌"
            task_label = f"{result.get('pname','?')}/{result.get('model_short','?')}/{result.get('mode','?')}/Q{result.get('query_id','?')}"
            now_str = datetime.datetime.now().strftime("%H:%M:%S")
            progress_fh.write(f"[{now_str}] {status} {completed}/{total_tasks} {task_label}\n")
            progress_fh.flush()

            logger.info(
                f"📈 Progress: {completed}/{total_tasks} "
                f"(✅{success_count} ❌{completed - success_count}) "
                f"| Elapsed: {elapsed:.0f}s | Rate: {rate:.1f} queries/hr"
            )

        unprocessed = total_tasks - completed
        progress_fh.write(f"\n# Finished: {datetime.datetime.now().isoformat()}\n")
        progress_fh.write(f"# Processed: {completed}/{total_tasks} (✅{success_count} ❌{completed - success_count})\n")
        if unprocessed > 0:
            progress_fh.write(f"# ⚠️ Unprocessed: {unprocessed} tasks (workers crashed or timed out)\n")

    except Exception as e:
        logger.error(f"❌ Error in result collection: {e}")
    finally:
        
        if progress_fh is not None:
            try:
                progress_fh.close()
            except Exception as e:
                logger.error(f"Failed to close progress log: {e}")

    
    logger.info("⏳ Waiting for all worker processes to finish...")
    process_timeout = 300  
    for i, p in enumerate(processes, 1):
        p.join(timeout=process_timeout)
        if p.is_alive():
            logger.warning(f"⚠️  Worker {i} still alive after {process_timeout}s, terminating...")
            p.terminate()
            p.join(timeout=10)
            if p.is_alive():
                logger.error(f"❌ Worker {i} failed to terminate, killing...")
                p.kill()
                p.join()

    
    try:
        import psutil
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        if children:
            logger.warning(f"⚠️  Found {len(children)} leaked child processes, cleaning up...")
            for child in children:
                try:
                    logger.debug(f"   Killing leaked process: {child.pid} ({child.name()})")
                    child.kill()
                except:
                    pass
    except Exception as e:
        logger.debug(f"Child process cleanup: {e}")

    
    try:
        while not task_queue.empty():
            task_queue.get_nowait()
        while not result_queue.empty():
            result_queue.get_nowait()
    except Exception as e:
        logger.debug(f"Queue cleanup: {e}")
    finally:
        try:
            task_queue.close()
            task_queue.join_thread()
        except Exception:
            pass
        try:
            result_queue.close()
            result_queue.join_thread()
        except Exception:
            pass

    total_time = time.time() - start_time

    unprocessed = total_tasks - completed
    logger.info(f"\n{'='*80}")
    logger.info(f"🏁 Finished in {total_time:.0f}s: processed {completed}/{total_tasks} tasks")
    logger.info(f"   ✅ Success: {success_count}")
    logger.info(f"   ❌ Failed: {completed - success_count}")
    if unprocessed > 0:
        logger.warning(f"   ⚠️  Unprocessed: {unprocessed} tasks (workers crashed or idle timeout)")
    logger.info(f"{'='*80}\n")

    
    summaries = aggregate_results(all_results, batch_dirs, total_time, total_queries_map=total_queries_map)

    
    global_summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "total_tasks": total_tasks,
        "processed": completed,
        "successful": success_count,
        "failed": completed - success_count,
        "unprocessed": total_tasks - completed,
        "total_time_seconds": total_time,
        "rate_queries_per_hour": completed / total_time * 3600 if total_time > 0 else 0,
        "group_summaries": {
            k: {
                "success_rate": v.get("success_rate"),
                "successful": v.get("successful"),
                "failed": v.get("failed"),
                "total_queries": v.get("total_queries"),
            }
            for k, v in summaries.items()
        },
    }
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    global_summary_file = os.path.join(RESULTS_DIR, f"global_summary_{ts}.json")
    with open(global_summary_file, "w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2, ensure_ascii=False)
    logger.info(f"📂 Global summary saved: {global_summary_file}")

    
    logger.info(f"\n{'='*80}")
    logger.info("📊 Results by group:")
    logger.info(f"{'='*80}")
    for key in sorted(summaries.keys()):
        s = summaries[key]
        logger.info(f"  {key}: {s.get('success_rate', 'N/A')} ({s.get('successful', 0)}/{s.get('total_queries', 0)})")
    logger.info(f"{'='*80}\n")

    return summaries


def main():
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
  python global_query_runner.py --group g1 --max-workers 8

  python global_query_runner.py --group g2 --max-workers 6

    python global_query_runner.py --group g3 --max-workers 6

  python global_query_runner.py --models "gpt-5.2,claude-sonnet-4-5-20250929" --max-workers 4

  python global_query_runner.py --group g1 --modes "clean,chaos" --max-workers 8

  python global_query_runner.py --group g1 --queries "1,3,5-7" --max-workers 4

  python global_query_runner.py --group g1 --no-skip --max-workers 8

  python global_query_runner.py --group g1 --no-skip-queries --max-workers 8
        """,
    )

    parser.add_argument(
        "--group",
        type=str,
        choices=["g1", "g2", "g3", "g4"],
        help="Use predefined model groups (g1/g2/g3)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Custom model list, comma-separated (takes precedence over --group)",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default=None,
        help=f"Modes to run, comma-separated (default: all). Options: {','.join(ALL_MODES)}",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Global max concurrent workers (default: 8)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Max steps per query (default: 100)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Use headless mode (default: True)",
    )
    parser.add_argument(
        "--no-dom",
        action="store_true",
        help="Disable DOM mode",
    )
    parser.add_argument(
        "--queries",
        type=str,
        default=None,
        help="Only run specified query IDs, comma-separated (e.g.: 1,3,5-7)",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Don't skip any completed tasks (force rerun all)",
    )
    parser.add_argument(
        "--no-skip-queries",
        action="store_true",
        help="Don't skip individual completed queries (only skip fully completed batches)",
    )
    parser.add_argument(
        "--projects",
        type=str,
        default=None,
        help="Only run specified project pnames, comma-separated (e.g.: food,ecom,note)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List tasks only, don't execute",
    )

    args = parser.parse_args()

    
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    elif args.group:
        models = MODEL_GROUPS[args.group]
    else:
        parser.error("Must specify --group or --models")

    
    if args.modes:
        modes = [m.strip() for m in args.modes.split(",")]
        for m in modes:
            if m not in ALL_MODES:
                parser.error(f"Unknown mode: {m}. Options: {', '.join(ALL_MODES)}")
    else:
        modes = ALL_MODES

    
    projects = PROJECT_CONFIGS
    if args.projects:
        selected_pnames = set(p.strip() for p in args.projects.split(","))
        projects = [
            p for p in PROJECT_CONFIGS
            if extract_pname(p["base_dir"]) in selected_pnames
        ]
        if not projects:
            parser.error(f"No matching projects found: {args.projects}")

    
    query_ids_filter = None
    if args.queries:
        query_ids_filter = []
        for part in args.queries.split(","):
            if "-" in part:
                start, end = map(int, part.split("-"))
                query_ids_filter.extend(range(start, end + 1))
            else:
                query_ids_filter.append(int(part))
        query_ids_filter = sorted(set(query_ids_filter))

    
    group_label = args.group or "custom"
    detail_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    detailed_log_path = os.path.join(RESULTS_DIR, f"detailed_{group_label}_{detail_ts}.log")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    detail_file_handler = logging.FileHandler(detailed_log_path, encoding="utf-8")
    detail_file_handler.setLevel(logging.DEBUG)
    detail_file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    
    logging.getLogger().addHandler(detail_file_handler)
    logger.info(f"📝 Detailed log: {detailed_log_path}")

    logger.info(f"{'='*80}")
    logger.info(f"🔧 Configuration:")
    logger.info(f"   Models: {models}")
    logger.info(f"   Modes: {modes}")
    logger.info(f"   Projects: {[extract_pname(p['base_dir']) for p in projects]}")
    logger.info(f"   Max workers: {args.max_workers}")
    logger.info(f"   Max steps: {args.max_steps}")
    if query_ids_filter:
        logger.info(f"   Query IDs filter: {query_ids_filter}")
    logger.info(f"   Skip completed batches: {not args.no_skip}")
    logger.info(f"   Skip completed queries: {not args.no_skip and not args.no_skip_queries}")
    logger.info(f"{'='*80}\n")

    
    skip_batches = not args.no_skip
    skip_queries = not args.no_skip and not args.no_skip_queries

    tasks, batch_dirs, skipped_batches, skipped_queries, total_queries_map = build_task_list(
        models=models,
        modes=modes,
        projects=projects,
        max_steps=args.max_steps,
        skip_completed_batches=skip_batches,
        skip_completed_queries=skip_queries,
        query_ids_filter=query_ids_filter,
    )

    logger.info(f"📊 Task summary:")
    logger.info(f"   Tasks to run: {len(tasks)}")
    logger.info(f"   Batches skipped (fully complete): {skipped_batches}")
    logger.info(f"   Queries skipped (individually complete): {skipped_queries}")
    logger.info("")

    if args.dry_run:
        logger.info("🔍 Dry run - listing tasks:")
        for i, t in enumerate(tasks):
            logger.info(f"  [{i+1}] {t['pname']}/{t['model_short']}/{t['mode']}/Q{t['query_id']}")
        logger.info(f"\nTotal: {len(tasks)} tasks")
        return

    if not tasks:
        logger.info("✅ All tasks completed, nothing to run!")
        return

    
    try:
        
        try:
            import psutil
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024  
            logger.info(f"📊 Main process memory before execution: {mem_before:.1f} MB")
        except:
            pass

        run_global_scheduler(
            tasks=tasks,
            batch_dirs=batch_dirs,
            max_workers=args.max_workers,
            max_steps=args.max_steps,
            use_dom=not args.no_dom,
            headless=args.headless,
            total_queries_map=total_queries_map,
            detailed_log_path=detailed_log_path,
        )

        
        try:
            import psutil
            process = psutil.Process()
            mem_after = process.memory_info().rss / 1024 / 1024  
            logger.info(f"📊 Main process memory after execution: {mem_after:.1f} MB (diff: {mem_after - mem_before:+.1f} MB)")
        except:
            pass

    except KeyboardInterrupt:
        logger.warning("\n⚠️  Interrupted by user (Ctrl+C)")
        logger.info("Scheduler will attempt graceful shutdown of worker processes...")
    except Exception as e:
        logger.error(f"\n❌ Scheduler error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Program interrupted")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)
