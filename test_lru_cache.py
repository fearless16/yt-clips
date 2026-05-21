import unittest
from lru_cache import LRUCache

class TestLRUCache(unittest.TestCase):
    def setUp(self):
        self.cache = LRUCache(capacity=2, ttl=1)

    def test_basic_operations(self):
        self.cache.put('a', 1)
        self.assertEqual(self.cache.get('a'), 1)
        self.cache.put('b', 2)
        self.assertEqual(self.cache.get('a'), 1)
        self.cache.put('c', 3)
        self.assertIsNone(self.cache.get('a'))
        self.assertEqual(self.cache.get('b'), 2)

    def test_ttl(self):
        self.cache.put('d', 4)
        self.assertEqual(self.cache.get('d'), 4)
        time.sleep(2)
        self.assertIsNone(self.cache.get('d'))

    def test_delete(self):
        self.cache.put('e', 5)
        self.cache.delete('e')
        self.assertIsNone(self.cache.get('e'))

if __name__ == '__main__':
    unittest.main()