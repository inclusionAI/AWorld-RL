from typing import Optional
from aworld.memory.base import MemoryBase, MemoryItem, MemoryStore, InMemoryMemoryStore


class Memory(MemoryBase):

    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store


    @classmethod
    def from_config(cls, config: dict) -> "Memory":
        """
        Initialize a Memory instance from a configuration dictionary.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            Memory: Memory instance.
        """
        if config.get("memory_store") == "inmemory":    
            return cls(memory_store=InMemoryMemoryStore())
        else:
            raise ValueError(f"Invalid memory store type: {config.get('memory_store')}")
        

    def add(self, memory_item: MemoryItem):
        self.memory_store.add(memory_item)

    def update(self, memory_item: MemoryItem):
        self.memory_store.update(memory_item)

    def delete(self, memory_id):
        self.memory_store.delete(memory_id)

    def get(self, memory_id) -> Optional[MemoryItem]:
        return self.memory_store.get(memory_id)

    def get_all(self) -> list[MemoryItem]:
        return self.memory_store.get_all()

    def get_last_n(self, last_rounds) -> list[MemoryItem]:
        """
        Get last n memories.

        Args:
            last_rounds (int): Number of memories to retrieve.

        Returns:
            list[MemoryItem]: List of latest memories.
        """
        memory_items = self.memory_store.get_last_n(last_rounds)
        while len(memory_items) > 0 and memory_items[0].metadata and "tool_call_id" in memory_items[0].metadata and memory_items[0].metadata["tool_call_id"]:
            last_rounds = last_rounds + 1
            memory_items = self.memory_store.get_last_n(last_rounds)

        return memory_items

    def retrieve(self, query, filters: dict) -> list[MemoryItem]:
        return self.memory_store.retrieve(query, filters)

    def history(self, memory_id) -> list[MemoryItem]:
        return self.memory_store.history(memory_id)