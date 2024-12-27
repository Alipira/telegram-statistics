class Node:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.next = None


class HashTable:
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.buckets = [None] * self.capacity
        self.size = 0

    def _hash(self, key):
        return hash(key) % self.capacity

    def __setitem__(self, key, value):
        index = self._hash(key)
        head = self.buckets[index]
        current = head
        while current:
            if current.key == key:
                current.value = value  # Update if key exists
                return
            current = current.next
        new_node = Node(key, value)
        new_node.next = head
        self.buckets[index] = new_node
        self.size += 1

    def __getitem__(self, key):
        index = self._hash(key)
        current = self.buckets[index]
        while current:
            if current.key == key:
                return current.value
            current = current.next
        raise KeyError(f"Key {key} not found.")

    def __delitem__(self, key):
        index = self._hash(key)
        current = self.buckets[index]
        prev = None
        while current:
            if current.key == key:
                if prev:
                    prev.next = current.next
                else:
                    self.buckets[index] = current.next
                self.size -= 1
                return
            prev, current = current, current.next
        raise KeyError(f"Key {key} not found.")

    def __contains__(self, key):
        index = self._hash(key)
        current = self.buckets[index]

        while current:
            if current.key == key:
                return True
            current = current.next
        return False

    def get(self, key):
        try:
            return self[key]  # Uses __getitem__
        except KeyError:
            return None

    def set(self, key, value):
        self[key] = value  # Uses __setitem__

    def delete(self, key):
        try:
            del self[key]  # Uses __delitem__
        except KeyError:
            pass  # FIXME:

    def __repr__(self):
        items = []
        for head in self.buckets:
            current = head
            while current:
                items.append(f"{current.key}: {current.value}")
                current = current.next
        return "{" + ", ".join(items) + "}"

    def __len__(self):
        return self.size
