import json
import math
import numpy as np

class Buckets:
    def __init__(self, mn, mx, num_buckets=None, bucket_size=None):
        assert (num_buckets is not None) ^ (bucket_size is not None)
        self.mn = mn
        self.mx = mx
        self.bucket_size = bucket_size
        self.num_buckets = num_buckets
        if bucket_size is None:
            self.bucket_size = (self.mx - self.mn)/ float(num_buckets)
        if num_buckets is None:
            self.num_buckets = int(math.ceil((mx-mn)/float(bucket_size)))

    def get_bucket(self, val):
        return min(max(0,int((val - self.mn)/self.bucket_size)), self.num_buckets-1)

