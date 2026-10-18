"""Reliable-queue behaviour: claimed tasks survive a worker crash."""

import json
import os
import unittest
from unittest.mock import patch
from uuid import uuid4

import redis

from app.controllers.manager import redis_manager
from app.controllers.manager.redis_manager import RedisTaskManager
from app.models import const
from app.services import state as sm


def _payload(task_id):
    return json.dumps({"func": "start", "args": [], "kwargs": {"task_id": task_id}})


class TestRedisReliableQueue(unittest.TestCase):
    def setUp(self):
        host = os.getenv("MPT_TEST_REDIS_HOST", "127.0.0.1")
        port = int(os.getenv("MPT_TEST_REDIS_PORT", "6379"))
        db = int(os.getenv("MPT_TEST_REDIS_DB", "15"))
        self.queue = f"test-reliable:{uuid4()}"
        self.manager = RedisTaskManager(0, f"redis://{host}:{port}/{db}")
        try:
            self.manager.redis_client.ping()
        except redis.RedisError as exc:
            self.skipTest(f"Redis is unavailable: {exc}")
        self.manager.queue = self.queue
        self.client = self.manager.redis_client

        def cleanup():
            for key in self.client.scan_iter(match=f"{self.queue}*"):
                self.client.delete(key)
            self.manager.close()
            self.client.close()

        self.addCleanup(cleanup)
        func_map = patch.object(
            redis_manager, "FUNC_MAP", {"start": lambda task_id=None: None}
        )
        func_map.start()
        self.addCleanup(func_map.stop)

    def test_dequeue_moves_task_to_processing_list_until_acked(self):
        self.client.rpush(self.queue, _payload("t1"))
        task = self.manager.dequeue()
        self.assertEqual(task["kwargs"]["task_id"], "t1")
        self.assertEqual(self.client.llen(self.queue), 0)
        self.assertEqual(self.client.llen(self.manager.processing_key), 1)

        self.manager._ack(self.manager._forget_inflight("t1"))
        self.assertEqual(self.client.llen(self.manager.processing_key), 0)

    def test_run_task_acks_on_completion_and_on_failure(self):
        for task_id, boom in (("ok", False), ("bad", True)):
            self.client.rpush(self.queue, _payload(task_id))
            self.manager.dequeue()

            def func(task_id=None, boom=boom):
                if boom:
                    raise RuntimeError("fail")

            self.manager.current_tasks = 1
            try:
                self.manager.run_task(func, task_id=task_id)
            except RuntimeError:
                pass
        self.assertEqual(self.client.llen(self.manager.processing_key), 0)

    def test_discarded_unusable_item_is_not_left_in_processing(self):
        self.client.rpush(self.queue, "not json")
        self.client.rpush(self.queue, _payload("good"))
        task = self.manager.dequeue()
        self.assertEqual(task["kwargs"]["task_id"], "good")
        self.assertEqual(self.client.llen(self.manager.processing_key), 1)

    def test_orphaned_items_of_dead_worker_are_requeued_once(self):
        dead = f"{self.queue}:processing:otherhost:1:deadbeef"
        self.client.rpush(dead, _payload("a"), _payload("b"))
        self.assertEqual(self.manager.requeue_orphaned_tasks(), 2)
        self.assertEqual(self.client.llen(dead), 0)
        self.assertEqual(self.client.llen(self.queue), 2)
        # Idempotent: a second scan finds nothing to move.
        self.assertEqual(self.manager.requeue_orphaned_tasks(), 0)
        self.assertEqual(self.client.llen(self.queue), 2)

    def test_live_worker_items_are_left_alone(self):
        live = f"{self.queue}:processing:otherhost:1:cafe"
        self.client.rpush(live, _payload("a"))
        self.client.set(
            f"{self.queue}:worker:otherhost:1:cafe",
            redis_manager._PROCESS_INCARNATION,
            ex=60,
        )
        self.assertEqual(self.manager.requeue_orphaned_tasks(), 0)
        self.assertEqual(self.client.llen(live), 1)

    def test_restarted_process_reusing_host_pid_is_detected(self):
        previous = f"{self.manager._host_pid}:oldold01"
        self.client.rpush(f"{self.queue}:processing:{previous}", _payload("a"))
        self.client.set(
            f"{self.queue}:worker:{previous}", "previous-incarnation", ex=60
        )
        self.assertEqual(self.manager.requeue_orphaned_tasks(), 1)
        self.assertEqual(self.client.llen(self.queue), 1)

    def test_resume_queued_tasks_recovers_then_dispatches(self):
        dead = f"{self.queue}:processing:gone:2:feedface"
        self.client.rpush(dead, _payload("lost"))
        self.manager.max_concurrent_tasks = 1
        with (
            patch.object(self.manager, "execute_task") as execute,
            patch.object(redis_manager, "FUNC_MAP", {"start": lambda **kw: None}),
        ):
            self.manager.resume_queued_tasks()
        execute.assert_called_once()
        self.assertEqual(execute.call_args.kwargs["task_id"], "lost")

    def test_failure_state_error_is_redacted(self):
        bad = json.dumps({"func": "start", "args": 5, "kwargs": {"task_id": "leak"}})
        self.client.rpush(self.queue, bad)
        with patch.object(sm, "state", sm.MemoryState()) as state:
            state.update_task("leak", state=const.TASK_STATE_PROCESSING)
            with patch.object(
                redis_manager,
                "redact_secrets_in_text",
                side_effect=lambda t: "REDACTED:" + t,
            ):
                self.assertIsNone(self.manager.dequeue())
            self.assertTrue(state.get_task("leak")["error"].startswith("REDACTED:"))
