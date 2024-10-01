import psutil
import time
from typing import Dict, Any, Optional, List
import logging
import threading
import asyncio

class NetworkProfiler:
    def __init__(self, interval: float = 0.1, detailed: bool = True):
        self.interval = interval
        self.detailed = detailed
        self.start_data: Optional[psutil._common.snetio] = None
        self.end_data: Optional[psutil._common.snetio] = None
        self.is_profiling = False
        self.thread: Optional[threading.Thread] = None
        self.network_usage: List[Dict[str, int]] = []
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        self.async_mode = False
        self.task: Optional[asyncio.Task] = None

    def start(self):
        self.start_data = psutil.net_io_counters()
        self.is_profiling = True
        self.thread = threading.Thread(target=self._profile_network, daemon=True)
        self.thread.start()
        self.logger.info("Network profiling started")

    async def start_async(self):
        self.start_data = psutil.net_io_counters()
        self.is_profiling = True
        self.async_mode = True
        self.task = asyncio.create_task(self._profile_network_async())
        self.logger.info("Async network profiling started")

    def stop(self) -> Dict[str, int]:
        self.is_profiling = False
        if self.thread:
            self.thread.join()
        self.end_data = psutil.net_io_counters()
        total_usage = self._calculate_usage(self.start_data, self.end_data)
        self.logger.info(f"Network profiling stopped. Total usage: {total_usage}")
        return total_usage

    async def stop_async(self) -> Dict[str, int]:
        self.is_profiling = False
        if self.task:
            await self.task
        self.end_data = psutil.net_io_counters()
        total_usage = self._calculate_usage(self.start_data, self.end_data)
        self.logger.info(f"Async network profiling stopped. Total usage: {total_usage}")
        return total_usage

    def _profile_network(self):
        while self.is_profiling:
            current_data = psutil.net_io_counters()
            with self.lock:
                self.network_usage.append(self._calculate_usage(self.start_data, current_data))
            time.sleep(self.interval)

    async def _profile_network_async(self):
        while self.is_profiling:
            current_data = psutil.net_io_counters()
            with self.lock:
                self.network_usage.append(self._calculate_usage(self.start_data, current_data))
            await asyncio.sleep(self.interval)

    def _calculate_usage(self, start: psutil._common.snetio, end: psutil._common.snetio) -> Dict[str, int]:
        basic_usage = {
            'bytes_sent': end.bytes_sent - start.bytes_sent,
            'bytes_recv': end.bytes_recv - start.bytes_recv,
            'packets_sent': end.packets_sent - start.packets_sent,
            'packets_recv': end.packets_recv - start.packets_recv,
        }
        if self.detailed:
            basic_usage.update({
                'errin': end.errin - start.errin,
                'errout': end.errout - start.errout,
                'dropin': end.dropin - start.dropin,
                'dropout': end.dropout - start.dropout
            })
        return basic_usage

    def get_current_usage(self) -> Dict[str, int]:
        with self.lock:
            if self.network_usage:
                return self.network_usage[-1]
            return {'bytes_sent': 0, 'bytes_recv': 0, 'packets_sent': 0, 'packets_recv': 0}

    def get_average_usage(self) -> Dict[str, float]:
        with self.lock:
            if not self.network_usage:
                return {'bytes_sent': 0.0, 'bytes_recv': 0.0, 'packets_sent': 0.0, 'packets_recv': 0.0}
            avg_usage = {k: sum(usage[k] for usage in self.network_usage) / len(self.network_usage)
                         for k in self.network_usage[0].keys()}
            return avg_usage

    def get_total_usage(self) -> Dict[str, int]:
        with self.lock:
            if not self.network_usage:
                return {'bytes_sent': 0, 'bytes_recv': 0, 'packets_sent': 0, 'packets_recv': 0}
            total_usage = {k: sum(usage[k] for usage in self.network_usage)
                           for k in self.network_usage[0].keys()}
            return total_usage

    def get_usage_over_time(self) -> List[Dict[str, int]]:
        with self.lock:
            return self.network_usage.copy()

    def reset(self):
        with self.lock:
            self.network_usage.clear()
        self.start_data = None
        self.end_data = None
        self.logger.info("Network profiler reset")

    def get_network_stats(self) -> Dict[str, Any]:
        stats = psutil.net_io_counters()
        return {
            'bytes_sent': stats.bytes_sent,
            'bytes_recv': stats.bytes_recv,
            'packets_sent': stats.packets_sent,
            'packets_recv': stats.packets_recv,
            'errin': stats.errin,
            'errout': stats.errout,
            'dropin': stats.dropin,
            'dropout': stats.dropout
        }

    def get_network_connections(self) -> List[Dict[str, Any]]:
        connections = psutil.net_connections()
        return [
            {
                'fd': conn.fd,
                'family': conn.family,
                'type': conn.type,
                'laddr': conn.laddr,
                'raddr': conn.raddr,
                'status': conn.status,
                'pid': conn.pid
            }
            for conn in connections
        ]

    def get_network_interfaces(self) -> Dict[str, Any]:
        interfaces = psutil.net_if_addrs()
        stats = psutil.net_if_stats()
        result = {}
        for interface, addrs in interfaces.items():
            result[interface] = {
                'addresses': [
                    {
                        'family': addr.family,
                        'address': addr.address,
                        'netmask': addr.netmask,
                        'broadcast': addr.broadcast,
                        'ptp': addr.ptp
                    }
                    for addr in addrs
                ],
                'stats': {
                    'isup': stats[interface].isup,
                    'duplex': stats[interface].duplex,
                    'speed': stats[interface].speed,
                    'mtu': stats[interface].mtu
                }
            }
        return result

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    async def __aenter__(self):
        await self.start_async()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop_async()