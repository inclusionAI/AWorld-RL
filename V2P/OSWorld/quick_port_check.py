#!/usr/bin/env python3
"""
Quick script to check Docker port usage
"""

import docker
import subprocess
import json

def quick_docker_ports():
    """Quickly get port information using Docker command"""
    try:
        # Method 1: Use docker ps command
        result = subprocess.run(['docker', 'ps', '--format', 'table {{.Names}}\t{{.Image}}\t{{.Ports}}\t{{.Status}}'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("🐳 Docker container port information (via docker ps):")
            print("=" * 100)
            print(result.stdout)
        else:
            print(f"Docker command execution failed: {result.stderr}")
            
    except FileNotFoundError:
        print("❌ Docker command not found, please ensure Docker is installed")

def docker_api_ports():
    """Get port information using Docker API"""
    try:
        client = docker.from_env()
        containers = client.containers.list()
        
        print(f"\n🔍 Detailed port mapping information ({len(containers)} containers):")
        print("=" * 100)
        
        for container in containers:
            print(f"\n📦 Container: {container.name}")
            print(f"   Image: {container.image.tags[0] if container.image.tags else 'N/A'}")
            print(f"   Status: {container.status}")
            
            # Get port mappings
            ports = container.attrs['NetworkSettings']['Ports']
            if ports:
                print(f"   Port mappings:")
                for internal_port, mappings in ports.items():
                    if mappings:
                        for mapping in mappings:
                            host_ip = mapping['HostIp'] or '0.0.0.0'
                            host_port = mapping['HostPort']
                            print(f"     {host_ip}:{host_port} → {internal_port}")
                    else:
                        print(f"     {internal_port} (not mapped)")
            else:
                print(f"   Port mappings: None")
                
            # Highlight OSWorld containers
            if 'osworld' in container.image.tags[0].lower() if container.image.tags else False:
                print(f"   🖥️  This is an OSWorld container")
                
    except Exception as e:
        print(f"❌ Docker API connection failed: {e}")

def get_used_ports_summary():
    """Get summary of all used ports"""
    try:
        client = docker.from_env()
        containers = client.containers.list()
        
        all_host_ports = []
        osworld_containers = []
        
        for container in containers:
            ports = container.attrs['NetworkSettings']['Ports']
            container_ports = []
            
            if ports:
                for internal_port, mappings in ports.items():
                    if mappings:
                        for mapping in mappings:
                            host_port = int(mapping['HostPort'])
                            all_host_ports.append(host_port)
                            container_ports.append((internal_port, host_port))
            
            # Check if it's an OSWorld container
            image_name = container.image.tags[0] if container.image.tags else ''
            if 'osworld' in image_name.lower():
                osworld_containers.append({
                    'name': container.name,
                    'ports': container_ports
                })
        
        print(f"\n📈 Port usage summary:")
        print(f"Total host ports used: {len(all_host_ports)}")
        if all_host_ports:
            print(f"Port range: {min(all_host_ports)} - {max(all_host_ports)}")
            print(f"Used ports: {sorted(all_host_ports)}")
        
        if osworld_containers:
            print(f"\n🖥️  OSWorld container port allocation:")
            for container in osworld_containers:
                print(f"  {container['name']}:")
                for internal_port, host_port in container['ports']:
                    port_type = "Unknown"
                    
                    # Determine port type based on port range
                    # VNC: 8006-8079, VLC: 8080-8199, Server: 5000-5999, Chromium: 9222-9999
                    if 8006 <= host_port <= 8079:
                        port_type = "VNC"
                    elif 8080 <= host_port <= 8199:
                        port_type = "VLC"
                    elif 5000 <= host_port <= 5999:
                        port_type = "Server"
                    elif 9222 <= host_port <= 9999:
                        port_type = "Chromium"
                    print(f"    {port_type}: {host_port}")
                    
    except Exception as e:
        print(f"❌ Failed to get port summary: {e}")

if __name__ == "__main__":
    # Use Docker command for quick check
    quick_docker_ports()
    
    # Use Docker API to get detailed information
    docker_api_ports()
    
    # Get port usage summary
    get_used_ports_summary()
