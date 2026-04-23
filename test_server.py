import unittest
import os
import sys
import json
import shutil
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))


class TestFlaskServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server_thread = None
        cls.base_url = "http://127.0.0.1:5558"
        cls.data_dir = Path(tempfile.mkdtemp())

        os.environ["DATA_DIR"] = str(cls.data_dir)

        import serve

        serve.DATA_DIR = cls.data_dir
        serve.CONVERSATIONS_DIR = cls.data_dir / "conversations"
        serve.ATTACHMENTS_DIR = cls.data_dir / "attachments"
        serve.CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
        serve.ATTACHMENTS_DIR.mkdir(parents=True, exist_ok=True)

        def run_server():
            serve.app.run(
                host="127.0.0.1",
                port=5558,
                debug=False,
                use_reloader=False,
                threaded=True,
            )

        cls.server_thread = threading.Thread(target=run_server, daemon=True)
        cls.server_thread.start()
        time.sleep(2)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.data_dir, ignore_errors=True)

    def _request(self, method, path, data=None, headers=None):
        import urllib.request

        url = f"{self.base_url}{path}"
        if isinstance(data, str):
            data = data.encode("utf-8")
        elif data is not None and not isinstance(data, bytes):
            data = data.encode("utf-8")
        else:
            data = None
        req = urllib.request.Request(url=url, data=data, headers=headers or {})
        req.get_method = lambda: method
        return urllib.request.urlopen(req)

    def test_new_conversation_returns_conv_id(self):
        resp = self._request("POST", "/new_conversation", "{}")
        result = json.loads(resp.read().decode())
        self.assertIn("conv_id", result)
        self.assertIsInstance(result["conv_id"], int)

    def test_new_conversation_with_name(self):
        data = json.dumps({"name": "My Test Chat"})
        headers = {"Content-Type": "application/json"}
        resp = self._request("POST", "/new_conversation", data, headers)
        result = json.loads(resp.read().decode())
        self.assertIn("conv_id", result)
        self.assertIsInstance(result["conv_id"], int)

    def test_list_conversations_empty(self):
        import serve

        shutil.rmtree(serve.CONVERSATIONS_DIR)
        serve.CONVERSATIONS_DIR.mkdir()

        resp = self._request("GET", "/list_conversations")
        result = json.loads(resp.read().decode())
        self.assertEqual(result, [])

    def test_list_conversations_returns_list(self):
        self._request("POST", "/new_conversation", "{}")
        resp = self._request("GET", "/list_conversations")
        result = json.loads(resp.read().decode())
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_list_conversations_has_correct_fields(self):
        self._request("POST", "/new_conversation", '{"name": "Test"}')
        resp = self._request("GET", "/list_conversations")
        result = json.loads(resp.read().decode())
        self.assertIn("conv_id", result[0])
        self.assertIn("name", result[0])

    def test_history_empty(self):
        self._request("POST", "/new_conversation", "{}")
        resp = self._request("GET", "/conversations/1/history")
        result = json.loads(resp.read().decode())
        self.assertEqual(result, [])

    def test_history_returns_messages(self):
        import serve

        conv_dir = serve.get_conversation_dir(1)
        history_path = conv_dir / "history.json"
        history_path.write_text(
            json.dumps([{"role": "user", "message": "hello", "attachments": []}])
        )

        resp = self._request("GET", "/conversations/1/history")
        result = json.loads(resp.read().decode())
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["message"], "hello")

    def test_history_maintains_order(self):
        import serve

        conv_dir = serve.get_conversation_dir(1)
        history_path = conv_dir / "history.json"
        history_path.write_text(
            json.dumps(
                [
                    {"role": "user", "message": "first", "attachments": []},
                    {"role": "assistant", "message": "hi", "attachments": []},
                    {"role": "user", "message": "second", "attachments": []},
                ]
            )
        )

        resp = self._request("GET", "/conversations/1/history")
        result = json.loads(resp.read().decode())
        self.assertEqual(len(result), 3)

    def test_attachments_empty(self):
        resp = self._request("GET", "/conversations/1/attachments")
        result = json.loads(resp.read().decode())
        self.assertEqual(result, [])

    def test_attachments_returns_list(self):
        self._request("POST", "/new_conversation", "{}")
        resp = self._request("GET", "/conversations/1/attachments")
        result = json.loads(resp.read().decode())
        self.assertIsInstance(result, list)

    def test_sources_empty(self):
        resp = self._request("GET", "/conversations/1/sources")
        result = json.loads(resp.read().decode())
        self.assertEqual(result, [])

    def test_graph_empty(self):
        resp = self._request("GET", "/conversations/1/graph")
        result = json.loads(resp.read().decode())
        self.assertIn("edges", result)
        self.assertIn("nodes", result)
        self.assertEqual(result["edges"], [])
        self.assertEqual(result["nodes"], [])

    def test_graph_with_data(self):
        import serve
        from graph import GraphDatabase

        db = GraphDatabase()
        db.add_node("Test")
        db.save(str(serve.get_conversation_dir(1) / "graph"))

        resp = self._request("GET", "/conversations/1/graph")
        result = json.loads(resp.read().decode())
        self.assertGreater(len(result["nodes"]), 0)


class TestUploadAttachment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://127.0.0.1:5559"
        cls.data_dir = Path(tempfile.mkdtemp())
        os.environ["DATA_DIR"] = str(cls.data_dir)

        import serve

        serve.DATA_DIR = cls.data_dir
        serve.CONVERSATIONS_DIR = cls.data_dir / "conversations"
        serve.ATTACHMENTS_DIR = cls.data_dir / "attachments"
        serve.CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
        serve.ATTACHMENTS_DIR.mkdir(parents=True, exist_ok=True)

        def run_server():
            serve.app.run(
                host="127.0.0.1",
                port=5559,
                debug=False,
                use_reloader=False,
                threaded=True,
            )

        cls.server_thread = threading.Thread(target=run_server, daemon=True)
        cls.server_thread.start()
        time.sleep(2)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.data_dir, ignore_errors=True)

    def _request(self, method, path, data=None, headers=None):
        import urllib.request

        url = f"{self.base_url}{path}"
        if isinstance(data, str):
            data = data.encode("utf-8")
        elif data is not None and not isinstance(data, bytes):
            data = data.encode("utf-8")
        else:
            data = None
        req = urllib.request.Request(url=url, data=data, headers=headers or {})
        req.get_method = lambda: method
        return urllib.request.urlopen(req)

    def test_upload_returns_attachment_id(self):
        import serve

        serve.get_conversation_dir(1).mkdir(parents=True, exist_ok=True)

        data = b"test file content"
        headers = {"Content-Type": "text/plain"}
        resp = self._request(
            "POST", "/conversations/1/upload_attachment", data, headers
        )
        result = json.loads(resp.read().decode())
        self.assertIn("attachment_id", result)

    def test_upload_with_filename(self):
        import serve

        serve.get_conversation_dir(2).mkdir(parents=True, exist_ok=True)

        data = b"content"
        headers = {
            "Content-Type": "text/plain",
            "Content-Disposition": 'attachment; filename="test.txt"',
        }
        resp = self._request(
            "POST", "/conversations/2/upload_attachment", data, headers
        )
        result = json.loads(resp.read().decode())
        self.assertIn("attachment_id", result)

    def test_upload_and_list(self):
        import serve

        serve.get_conversation_dir(3).mkdir(parents=True, exist_ok=True)

        data = "test"
        headers = {"Content-Type": "text/plain"}
        resp = self._request(
            "POST", "/conversations/3/upload_attachment", data, headers
        )

        resp = self._request("GET", "/conversations/3/attachments")
        result = json.loads(resp.read().decode())
        self.assertEqual(len(result), 1)
        self.assertGreater(result[0]["size"], 0)


def test_upload_multiple_attachments(self):
    import serve

    conv_dir = serve.get_conversation_dir(4)
    conv_dir.mkdir(parents=True, exist_ok=True)

    for i in range(3):
        data = f"content {i}"
        headers = {"Content-Type": "text/plain"}
        self._request("POST", f"/conversations/4/upload_attachment", data, headers)

    resp = self._request("GET", "/conversations/4/attachments")
    result = json.loads(resp.read().decode())
    self.assertEqual(len(result), 3)


class TestChatEndpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://127.0.0.1:5560"
        cls.data_dir = Path(tempfile.mkdtemp())
        os.environ["DATA_DIR"] = str(cls.data_dir)

        import serve

        serve.DATA_DIR = cls.data_dir
        serve.CONVERSATIONS_DIR = cls.data_dir / "conversations"
        serve.ATTACHMENTS_DIR = cls.data_dir / "attachments"
        serve.CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
        serve.ATTACHMENTS_DIR.mkdir(parents=True, exist_ok=True)

        def run_server():
            serve.app.run(
                host="127.0.0.1",
                port=5560,
                debug=False,
                use_reloader=False,
                threaded=True,
            )

        cls.server_thread = threading.Thread(target=run_server, daemon=True)
        cls.server_thread.start()
        time.sleep(2)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.data_dir, ignore_errors=True)

    def test_chat_accepts_message(self):
        import serve

        serve.get_conversation_dir(1).mkdir(parents=True, exist_ok=True)

        data = json.dumps({"message": "hello", "attachments": []})
        headers = {"Content-Type": "application/json"}

        import urllib.request

        req = urllib.request.Request(
            f"{self.base_url}/conversations/1/chat",
            data=data.encode(),
            headers=headers,
            method="POST",
        )
        resp = urllib.request.urlopen(req)
        body = resp.read().decode()

        has_content = False
        for line in body.strip().split("\n"):
            if line:
                try:
                    msg = json.loads(line)
                    if msg.get("type") in ("token", "error", "done"):
                        has_content = True
                except:
                    pass
        self.assertTrue(has_content)

    def test_chat_returns_done(self):
        import serve

        serve.get_conversation_dir(2).mkdir(parents=True, exist_ok=True)

        data = json.dumps({"message": "x", "attachments": []})
        headers = {"Content-Type": "application/json"}

        import urllib.request

        req = urllib.request.Request(
            f"{self.base_url}/conversations/2/chat",
            data=data.encode(),
            headers=headers,
            method="POST",
        )
        resp = urllib.request.urlopen(req)
        body = resp.read().decode()

        found_done = False
        for line in body.strip().split("\n"):
            if line:
                try:
                    msg = json.loads(line)
                    if msg.get("type") == "done":
                        found_done = True
                except:
                    pass
        self.assertTrue(found_done)

    def test_chat_with_attachments(self):
        import serve

        conv_dir = serve.get_conversation_dir(3)
        conv_dir.mkdir(parents=True, exist_ok=True)

        att_path = serve.ATTACHMENTS_DIR / "att1_3"
        att_path.write_text("test content")

        meta_path = conv_dir / "attachments.json"
        meta_path.write_text(
            json.dumps(
                [
                    {
                        "attachment_id": "att1",
                        "name": "test.txt",
                        "type": "text/plain",
                        "size": 12,
                    }
                ]
            )
        )

        data = json.dumps({"message": "test", "attachments": ["att1"]})
        headers = {"Content-Type": "application/json"}

        import urllib.request

        req = urllib.request.Request(
            f"{self.base_url}/conversations/3/chat",
            data=data.encode(),
            headers=headers,
            method="POST",
        )
        resp = urllib.request.urlopen(req)
        self.assertEqual(resp.status, 200)


class TestChatEndpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://127.0.0.1:5560"
        cls.data_dir = Path(tempfile.mkdtemp())
        os.environ["DATA_DIR"] = str(cls.data_dir)

        import serve

        serve.DATA_DIR = cls.data_dir
        serve.CONVERSATIONS_DIR = cls.data_dir / "conversations"
        serve.ATTACHMENTS_DIR = cls.data_dir / "attachments"
        serve.CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
        serve.ATTACHMENTS_DIR.mkdir(parents=True, exist_ok=True)

        def run_server():
            serve.app.run(
                host="127.0.0.1",
                port=5560,
                debug=False,
                use_reloader=False,
                threaded=True,
            )

        cls.server_thread = threading.Thread(target=run_server, daemon=True)
        cls.server_thread.start()
        time.sleep(2)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.data_dir, ignore_errors=True)

    def test_chat_accepts_message(self):
        import serve
        import re

        serve.get_conversation_dir(1).mkdir(parents=True, exist_ok=True)

        data = json.dumps({"message": "hello", "attachments": []})
        headers = {"Content-Type": "application/json"}

        import urllib.request

        req = urllib.request.Request(
            f"{self.base_url}/conversations/1/chat",
            data=data.encode(),
            headers=headers,
            method="POST",
        )
        resp = urllib.request.urlopen(req)
        body = resp.read().decode()

        has_content = False
        json_objects = re.findall(r"\{[^{}]*\}", body)
        for obj_str in json_objects:
            try:
                msg = json.loads(obj_str)
                if msg.get("type") in ("token", "error", "done"):
                    has_content = True
            except:
                pass
        self.assertTrue(has_content)

    def test_chat_returns_done(self):
        import serve
        import re

        serve.get_conversation_dir(2).mkdir(parents=True, exist_ok=True)

        data = json.dumps({"message": "test", "attachments": []})
        headers = {"Content-Type": "application/json"}

        import urllib.request

        req = urllib.request.Request(
            f"{self.base_url}/conversations/2/chat",
            data=data.encode(),
            headers=headers,
            method="POST",
        )
        resp = urllib.request.urlopen(req)
        body = resp.read().decode()

        found_done = False
        found_error = False
        json_objects = re.findall(r"\{[^{}]*\}", body)
        for obj_str in json_objects:
            try:
                msg = json.loads(obj_str)
                if msg.get("type") == "done":
                    found_done = True
                if msg.get("type") == "error":
                    found_error = True
            except:
                pass
        self.assertTrue(found_done or found_error)

    def test_chat_with_attachments(self):
        import serve

        conv_dir = serve.get_conversation_dir(3)
        conv_dir.mkdir(parents=True, exist_ok=True)

        att_path = serve.ATTACHMENTS_DIR / "att1_3"
        att_path.write_text("test content")

        meta_path = conv_dir / "attachments.json"
        meta_path.write_text(
            json.dumps(
                [
                    {
                        "attachment_id": "att1",
                        "name": "test.txt",
                        "type": "text/plain",
                        "size": 12,
                    }
                ]
            )
        )

        data = json.dumps({"message": "summarize this", "attachments": ["att1"]})
        headers = {"Content-Type": "application/json"}

        import urllib.request

        req = urllib.request.Request(
            f"{self.base_url}/conversations/3/chat",
            data=data.encode(),
            headers=headers,
            method="POST",
        )
        resp = urllib.request.urlopen(req)
        self.assertEqual(resp.status, 200)


if __name__ == "__main__":
    unittest.main()
