"""HTTP load test against a running server.

    INDEX_DIR=index locust -f bench/locustfile.py --headless -u 64 -r 16 -t 60s \
        --host http://127.0.0.1:8080 --csv bench/results/load
"""

import os
import random

from locust import FastHttpUser, constant, task
from queries import sample_queries

from newssearch.index import load_index

QUERIES = sample_queries(load_index(os.environ.get("INDEX_DIR", "index")), 5000)


class Searcher(FastHttpUser):
    wait_time = constant(0)  # closed loop: each user fires as fast as responses return

    @task(8)
    def search(self):
        self.client.get("/api/search", params={"q": random.choice(QUERIES)}, name="/api/search")

    @task(2)
    def suggest(self):
        word = random.choice(QUERIES).split()[0]
        self.client.get("/api/suggest", params={"prefix": word[:3]}, name="/api/suggest")
