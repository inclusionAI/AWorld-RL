import asyncio
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import threading
import fcntl
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Set
import tempfile

logger = logging.getLogger(__name__)


class PortManager:
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._allocated_ports: Set[int] = set()
                    cls._instance._port_lock = threading.Lock()
                    cls._instance._registry_path = Path(tempfile.gettempdir()) / "simweb_ports.json"
        return cls._instance

    def _load_registry(self) -> Set[int]:
        if not self._registry_path.exists():
            return set()
        try:
            data = json.loads(self._registry_path.read_text())
            if isinstance(data, list):
                return set(int(p) for p in data)
        except Exception:
            return set()
        return set()

    def _save_registry(self, ports: Set[int]) -> None:
        self._registry_path.write_text(json.dumps(sorted(ports)))
    
    def allocate_port(self, start_port: int = 3000, max_attempts: int = 1000) -> int:
        
        with self._port_lock:
            self._registry_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._registry_path, "a+") as lock_file:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                ports_in_use = self._load_registry()
                for port in range(start_port, start_port + max_attempts):
                    if port in ports_in_use:
                        continue

                    try:
                        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                            s.settimeout(0.1)
                            result = s.connect_ex(('localhost', port))
                            if result == 0:
                                continue
                    except Exception:
                        pass

                    try:
                        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                            s.bind(('localhost', port))
                        ports_in_use.add(port)
                        self._allocated_ports.add(port)
                        self._save_registry(ports_in_use)
                        logger.info(f"Port {port} allocated (total allocated: {len(ports_in_use)})")
                        return port
                    except OSError:
                        continue

                raise RuntimeError(f"No free port found in range {start_port}-{start_port + max_attempts}")
    
    def release_port(self, port: int):
        
        with self._port_lock:
            if not self._registry_path.exists():
                return
            with open(self._registry_path, "a+") as lock_file:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                ports_in_use = self._load_registry()
                if port in ports_in_use:
                    ports_in_use.discard(port)
                    self._save_registry(ports_in_use)
                if port in self._allocated_ports:
                    self._allocated_ports.discard(port)
                logger.info(f"Port {port} released (total allocated: {len(ports_in_use)})")
    
    def release_all_ports(self):
        
        with self._port_lock:
            if not self._registry_path.exists():
                self._allocated_ports.clear()
                return
            with open(self._registry_path, "a+") as lock_file:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                ports_in_use = self._load_registry()
                count = len(ports_in_use)
                ports_in_use.clear()
                self._save_registry(ports_in_use)
                self._allocated_ports.clear()
                logger.info(f"All {count} ports released")


_port_manager = PortManager()


class EnvironmentManager:
    
    
    def __init__(self, website_path: str, reset_data: bool = True, isolate_workspace: bool = True):
        
        self.original_website_path = Path(website_path).resolve()
        self.reset_data = reset_data
        self.isolate_workspace = isolate_workspace
        
        
        self.workspace_temp_dir: Optional[Path] = None
        self.website_path = self.original_website_path  
        
        self.frontend_path = self.website_path / "frontend"
        self.backend_path = self.website_path / "backend"
        
        
        if not self.frontend_path.exists():
            raise ValueError(f"Frontend path not found: {self.frontend_path}")
        if not self.backend_path.exists():
            raise ValueError(f"Backend path not found: {self.backend_path}")
        
        
        self.frontend_process: Optional[subprocess.Popen] = None
        self.backend_process: Optional[subprocess.Popen] = None
        
        
        self.frontend_port: Optional[int] = None
        self.backend_port: Optional[int] = None
        
        
        self.data_backup_path: Optional[Path] = None
        
        
        self.original_env_content: Optional[str] = None
        self.original_server_content: Optional[str] = None
    
    def _create_isolated_workspace(self):
        
        import uuid
        
        
        temp_base = Path(tempfile.gettempdir()) / "simweb_workspaces"
        temp_base.mkdir(exist_ok=True)
        
        workspace_id = f"{self.original_website_path.name}_{uuid.uuid4().hex[:8]}"
        self.workspace_temp_dir = temp_base / workspace_id
        
        logger.info(f"📁 Creating isolated workspace: {self.workspace_temp_dir.name}")
        
        
        
        frontend_src = self.original_website_path / "frontend"
        backend_src = self.original_website_path / "backend"
        
        frontend_dst = self.workspace_temp_dir / "frontend"
        backend_dst = self.workspace_temp_dir / "backend"
        
        
        self._copy_dir_exclude_node_modules(frontend_src, frontend_dst)
        
        
        self._copy_dir_exclude_node_modules(backend_src, backend_dst)
        
        
        self.website_path = self.workspace_temp_dir
        self.frontend_path = frontend_dst
        self.backend_path = backend_dst
        
        logger.info(f"✅ Isolated workspace created")
    
    def _copy_dir_exclude_node_modules(self, src: Path, dst: Path):
        
        dst.mkdir(parents=True, exist_ok=True)
        
        for item in src.iterdir():
            src_item = src / item.name
            dst_item = dst / item.name
            
            if item.name == "node_modules":
                
                if src_item.exists():
                    os.symlink(src_item, dst_item)
            elif item.is_dir():
                shutil.copytree(src_item, dst_item)
            else:
                shutil.copy2(src_item, dst_item)
        
    @staticmethod
    def find_free_port(start_port: int = 3000, max_attempts: int = 1000) -> int:
        
        return _port_manager.allocate_port(start_port, max_attempts)
    
    @staticmethod
    def release_port(port: int):
        
        _port_manager.release_port(port)
    
    def _backup_data(self):
        
        data_path = self.backend_path / "data"
        if data_path.exists():
            
            backup_dir = Path(tempfile.mkdtemp(prefix="simweb_data_backup_"))
            self.data_backup_path = backup_dir / "data"
            shutil.copytree(data_path, self.data_backup_path)
            logger.info(f"📦 Data backed up to: {self.data_backup_path}")
    
    def _restore_data(self):
        
        if self.data_backup_path and self.data_backup_path.exists():
            data_path = self.backend_path / "data"
            if data_path.exists():
                shutil.rmtree(data_path)
            shutil.copytree(self.data_backup_path, data_path)
            logger.info(f"♻️ Data restored from backup")
    
    def _cleanup_backup(self):
        
        if self.data_backup_path:
            backup_parent = self.data_backup_path.parent
            if backup_parent.exists():
                shutil.rmtree(backup_parent)
                logger.debug(f"🧹 Backup cleaned up: {backup_parent}")
    
    def _update_frontend_env(self, backend_port: int, frontend_port: int):
        
        env_path = self.frontend_path / ".env"
        
        
        if env_path.exists():
            self.original_env_content = env_path.read_text()
        
        
        env_content = f"""REACT_APP_API_URL=http://localhost:{backend_port}
PORT={frontend_port}
BROWSER=none
DANGEROUSLY_DISABLE_HOST_CHECK=true
"""
        env_path.write_text(env_content)
        logger.debug(f"📝 Updated frontend .env: PORT={frontend_port}, API_URL=http://localhost:{backend_port}")
    
    def _patch_frontend_api_urls(self, backend_port: int):
        
        import re
        
        src_path = self.frontend_path / "src"
        if not src_path.exists():
            logger.warning("⚠️ Frontend src directory not found, skipping API URL patching")
            return
        
        
        if not hasattr(self, '_original_src_files'):
            self._original_src_files = {}
        
        
        hardcoded_ports = [5000, 5001, 5002, 5003, 8000, 8080, 3001]
        
        
        for ext in ['*.js', '*.jsx', '*.ts', '*.tsx']:
            for file_path in src_path.rglob(ext):
                try:
                    content = file_path.read_text(encoding='utf-8')
                    original_content = content
                    modified = False
                    
                    
                    for port in hardcoded_ports:
                        pattern = f'http://localhost:{port}'
                        if pattern in content:
                            content = content.replace(pattern, f'http://localhost:{backend_port}')
                            modified = True
                    
                    if modified:
                        
                        self._original_src_files[str(file_path)] = original_content
                        
                        file_path.write_text(content, encoding='utf-8')
                        logger.debug(f"📝 Patched API URLs in: {file_path.name}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Failed to patch {file_path}: {e}")
        
        if self._original_src_files:
            logger.info(f"📝 Patched {len(self._original_src_files)} frontend files with backend port {backend_port}")
    
    def _restore_frontend_src_files(self):
        
        
        if self.isolate_workspace:
            return
            
        if hasattr(self, '_original_src_files') and self._original_src_files:
            restored_count = 0
            for file_path, content in self._original_src_files.items():
                try:
                    path = Path(file_path)
                    if path.parent.exists():
                        path.write_text(content, encoding='utf-8')
                        restored_count += 1
                except Exception as e:
                    logger.debug(f"⚠️ Could not restore {file_path}: {e}")
            if restored_count > 0:
                logger.debug(f"♻️ Restored {restored_count} frontend source files")
            self._original_src_files = {}
    
    def _update_backend_port(self, port: int):
        
        server_path = self.backend_path / "server.js"
        
        if server_path.exists():
            content = server_path.read_text()
            self.original_server_content = content
            
            
            import re
            
            new_content = re.sub(
                r'const\s+PORT\s*=\s*(?:process\.env\.PORT\s*\|\|\s*)?\d+',
                f'const PORT = {port}',
                content
            )
            server_path.write_text(new_content)
            logger.debug(f"📝 Updated backend server.js: PORT={port}")
    
    def _restore_configs(self):
        
        
        if self.isolate_workspace:
            return
            
        if self.original_env_content:
            env_path = self.frontend_path / ".env"
            try:
                if env_path.parent.exists():
                    env_path.write_text(self.original_env_content)
                    logger.debug("♻️ Restored frontend .env")
            except Exception as e:
                logger.debug(f"⚠️ Could not restore .env: {e}")
        
        if self.original_server_content:
            server_path = self.backend_path / "server.js"
            try:
                if server_path.parent.exists():
                    server_path.write_text(self.original_server_content)
                    logger.debug("♻️ Restored backend server.js")
            except Exception as e:
                logger.debug(f"⚠️ Could not restore server.js: {e}")
    
    async def _wait_for_service(self, port: int, service_name: str, timeout: float = 60.0) -> bool:
        
        import urllib.request
        import urllib.error
        
        start_time = time.time()
        url = f"http://localhost:{port}"
        last_error = None
        
        while time.time() - start_time < timeout:
            try:
                req = urllib.request.Request(url, method='GET')
                with urllib.request.urlopen(req, timeout=2) as resp:
                    if resp.status < 500:
                        logger.info(f"✅ {service_name} is ready on port {port}")
                        return True
            except urllib.error.HTTPError as e:
                
                if e.code < 500:
                    logger.info(f"✅ {service_name} is ready on port {port} (HTTP {e.code})")
                    return True
                last_error = f"HTTP {e.code}"
            except urllib.error.URLError as e:
                last_error = str(e.reason)
            except Exception as e:
                last_error = str(e)
            
            
            if service_name == "Frontend" and self.frontend_process:
                if self.frontend_process.poll() is not None:
                    
                    logger.error(f"❌ {service_name} process exited unexpectedly")
                    log_file = self.workspace_temp_dir / "frontend.log" if self.workspace_temp_dir else Path(tempfile.gettempdir()) / f"frontend_{port}.log"
                    if log_file.exists():
                        try:
                            with open(log_file, 'r') as f:
                                log_content = f.read()
                                if log_content:
                                    logger.error(f"   Frontend log (last 1000 chars): \n{log_content[-1000:]}")
                        except Exception as e:
                            logger.debug(f"   Could not read log file: {e}")
                    return False
            elif service_name == "Backend" and self.backend_process:
                if self.backend_process.poll() is not None:
                    logger.error(f"❌ {service_name} process exited unexpectedly")
                    return False
            
            await asyncio.sleep(1)
        
        logger.error(f"❌ {service_name} failed to start within {timeout}s (last error: {last_error})")
        
        if service_name == "Frontend":
            log_file = self.workspace_temp_dir / "frontend.log" if self.workspace_temp_dir else Path(tempfile.gettempdir()) / f"frontend_{port}.log"
            if log_file.exists():
                try:
                    with open(log_file, 'r') as f:
                        log_content = f.read()
                        if log_content:
                            logger.error(f"   Frontend log (last 2000 chars): \n{log_content[-2000:]}")
                        else:
                            logger.error(f"   Frontend log file is empty")
                except Exception as e:
                    logger.debug(f"   Could not read log file: {e}")
        return False

    
    def _start_backend(self) -> subprocess.Popen:
        
        
        if self.isolate_workspace:
            node_modules = self.backend_path / "node_modules"
            if not node_modules.exists():
                logger.info(f"📦 Installing backend dependencies...")
                install_process = subprocess.Popen(
                    ['npm', 'install'],
                    cwd=str(self.backend_path),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                install_stdout, install_stderr = install_process.communicate(timeout=120)
                if install_process.returncode != 0:
                    error_msg = install_stderr.decode('utf-8', errors='ignore')
                    logger.error(f"❌ npm install failed for backend: {error_msg[:500]}")
                    raise RuntimeError(f"Backend npm install failed: {error_msg[:500]}")
                logger.info(f"✅ Backend dependencies installed")
        
        env = os.environ.copy()
        env['PORT'] = str(self.backend_port)
        
        
        process = subprocess.Popen(
            ['npm', 'start'],
            cwd=str(self.backend_path),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid  
        )
        
        logger.info(f"🚀 Backend starting on port {self.backend_port} (PID: {process.pid})")
        return process
    
    def _start_frontend(self) -> subprocess.Popen:
        
        
        if self.isolate_workspace:
            node_modules = self.frontend_path / "node_modules"
            if not node_modules.exists():
                logger.info(f"📦 Installing frontend dependencies...")
                install_process = subprocess.Popen(
                    ['npm', 'install'],
                    cwd=str(self.frontend_path),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                install_stdout, install_stderr = install_process.communicate(timeout=120)
                if install_process.returncode != 0:
                    error_msg = install_stderr.decode('utf-8', errors='ignore')
                    logger.error(f"❌ npm install failed for frontend: {error_msg[:500]}")
                    raise RuntimeError(f"Frontend npm install failed: {error_msg[:500]}")
                logger.info(f"✅ Frontend dependencies installed")
        
        env = os.environ.copy()
        env['PORT'] = str(self.frontend_port)
        env['BROWSER'] = 'none'  
        env['REACT_APP_API_URL'] = f'http://localhost:{self.backend_port}'
        env['GENERATE_SOURCEMAP'] = 'false'  
        env['FAST_REFRESH'] = 'false'  
        env['NODE_ENV'] = 'development'  
        
        env['NODE_OPTIONS'] = '--max-old-space-size=512'
        
        
        log_file = self.workspace_temp_dir / "frontend.log" if self.workspace_temp_dir else Path(tempfile.gettempdir()) / f"frontend_{self.frontend_port}.log"
        
        
        
        logger.info(f"📦 Starting frontend in dev mode (supports dynamic ports)")
        logger.info(f"   Using NODE_OPTIONS: {env.get('NODE_OPTIONS', 'default')}")
        
        try:
            with open(log_file, 'w') as log:
                process = subprocess.Popen(
                    ['npm', 'start'],
                    cwd=str(self.frontend_path),
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid
                )
        except Exception as e:
            logger.error(f"❌ Failed to start frontend process: {e}")
            raise
        
        logger.info(f"🚀 Frontend starting on port {self.frontend_port} (PID: {process.pid}, logs: {log_file})")
        return process
    
    async def start(self) -> Tuple[str, int, int]:
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🌐 Starting environment for: {self.original_website_path.name}")
        logger.info(f"{'='*60}")
        
        try:
            
            if self.isolate_workspace:
                self._create_isolated_workspace()
            
            
            self.backend_port = self.find_free_port(5000)
            self.frontend_port = self.find_free_port(3000)
            logger.info(f"📍 Allocated ports: Frontend={self.frontend_port}, Backend={self.backend_port}")
            
            
            if self.reset_data:
                self._backup_data()
            
            
            self._update_backend_port(self.backend_port)
            self._update_frontend_env(self.backend_port, self.frontend_port)
            
            
            self._patch_frontend_api_urls(self.backend_port)
            
            
            self.backend_process = self._start_backend()
            backend_ready = await self._wait_for_service(
                self.backend_port, "Backend", timeout=30.0
            )
            if not backend_ready:
                raise RuntimeError("Backend failed to start")
            
            
            self.frontend_process = self._start_frontend()
            frontend_ready = await self._wait_for_service(
                self.frontend_port, "Frontend", timeout=90.0  
            )
            if not frontend_ready:
                raise RuntimeError("Frontend failed to start")
            
            frontend_url = f"http://localhost:{self.frontend_port}"
            logger.info(f"\n✅ Environment ready!")
            logger.info(f"   Frontend: {frontend_url}")
            logger.info(f"   Backend:  http://localhost:{self.backend_port}")
            
            return frontend_url, self.frontend_port, self.backend_port
            
        except Exception as e:
            logger.error(f"❌ Failed to start environment: {e}")
            await self.stop()  
            raise
    
    async def reset(self):
        
        logger.info("🔄 Resetting environment data...")
        
        if self.data_backup_path:
            self._restore_data()
            logger.info("✅ Data reset to initial state")
        else:
            logger.warning("⚠️ No backup available, cannot reset data")
    
    async def stop(self):
        
        logger.info("\n🛑 Stopping environment...")
        
        
        if self.frontend_process:
            try:
                
                os.killpg(os.getpgid(self.frontend_process.pid), signal.SIGTERM)
                self.frontend_process.wait(timeout=5)
                logger.info("   ✅ Frontend stopped")
            except subprocess.TimeoutExpired:
                
                try:
                    os.killpg(os.getpgid(self.frontend_process.pid), signal.SIGKILL)
                    logger.info("   ✅ Frontend force killed")
                except:
                    pass
            except Exception as e:
                try:
                    os.killpg(os.getpgid(self.frontend_process.pid), signal.SIGKILL)
                except:
                    pass
                logger.debug(f"   Frontend cleanup error: {e}")
            finally:
                self.frontend_process = None
        
        
        if self.backend_process:
            try:
                
                os.killpg(os.getpgid(self.backend_process.pid), signal.SIGTERM)
                self.backend_process.wait(timeout=5)
                logger.info("   ✅ Backend stopped")
            except subprocess.TimeoutExpired:
                
                try:
                    os.killpg(os.getpgid(self.backend_process.pid), signal.SIGKILL)
                    logger.info("   ✅ Backend force killed")
                except:
                    pass
            except Exception as e:
                try:
                    os.killpg(os.getpgid(self.backend_process.pid), signal.SIGKILL)
                except:
                    pass
                logger.debug(f"   Backend cleanup error: {e}")
            finally:
                self.backend_process = None
        
        
        try:
            import subprocess as sp
            
            if self.frontend_path and self.frontend_path.exists():
                sp.run(['pkill', '-f', f'npm start.*{self.frontend_path.name}'], 
                      stdout=sp.DEVNULL, stderr=sp.DEVNULL)
        except Exception as e:
            logger.debug(f"   Could not cleanup npm processes: {e}")
        
        
        if self.frontend_port:
            self.release_port(self.frontend_port)
            self.frontend_port = None
        if self.backend_port:
            self.release_port(self.backend_port)
            self.backend_port = None
        
        
        if self.isolate_workspace and self.workspace_temp_dir:
            self._cleanup_isolated_workspace()
        else:
            
            self._restore_frontend_src_files()
            
            self._restore_configs()
        
        
        self._cleanup_backup()
        
        logger.info("✅ Environment stopped and cleaned up")

    
    def _cleanup_isolated_workspace(self):
        
        if self.workspace_temp_dir and self.workspace_temp_dir.exists():
            try:
                shutil.rmtree(self.workspace_temp_dir, ignore_errors=True)
                logger.debug(f"🧹 Cleaned up isolated workspace: {self.workspace_temp_dir.name}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to cleanup workspace: {e}")
            self.workspace_temp_dir = None
    
    def get_url(self) -> str:
        
        if self.frontend_port:
            return f"http://localhost:{self.frontend_port}"
        raise RuntimeError("Environment not started")
    
    async def __aenter__(self):
        
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        
        await self.stop()
        return False


class MultiEnvironmentManager:
    
    
    def __init__(self, website_path: str, max_instances: int = 5):
        
        self.website_path = Path(website_path).resolve()
        self.max_instances = max_instances
        self.instances: Dict[int, EnvironmentManager] = {}
        self._lock = asyncio.Lock()
        self._instance_counter = 0
    
    async def acquire(self) -> Tuple[int, EnvironmentManager]:
        
        async with self._lock:
            if len(self.instances) >= self.max_instances:
                raise RuntimeError(f"Maximum instances ({self.max_instances}) reached")
            
            self._instance_counter += 1
            instance_id = self._instance_counter
            
            env = EnvironmentManager(str(self.website_path), reset_data=True)
            self.instances[instance_id] = env
            
            return instance_id, env
    
    async def release(self, instance_id: int):
        
        async with self._lock:
            if instance_id in self.instances:
                env = self.instances.pop(instance_id)
                await env.stop()
    
    async def stop_all(self):
        
        async with self._lock:
            for instance_id, env in list(self.instances.items()):
                try:
                    await env.stop()
                except Exception as e:
                    logger.error(f"Error stopping instance {instance_id}: {e}")
            self.instances.clear()




async def run_with_environment(
    website_path: str,
    task_func,
    reset_before_run: bool = True
):
    
    env = EnvironmentManager(website_path, reset_data=reset_before_run)
    
    try:
        url, _, _ = await env.start()
        result = await task_func(url, env)
        return result
    finally:
        await env.stop()




async def test_environment():
    
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python environment_manager.py <website_path>")
        print("Example: python environment_manager.py /path/to/generated-website-query-1")
        return
    
    website_path = sys.argv[1]
    
    logging.basicConfig(level=logging.INFO)
    
    async with EnvironmentManager(website_path) as env:
        print(f"\n🌐 Website running at: {env.get_url()}")
        print("Press Ctrl+C to stop...")
        
        try:
            
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\n\nStopping...")


if __name__ == "__main__":
    asyncio.run(test_environment())
