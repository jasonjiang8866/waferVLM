"""Real local HTTP exchange, including a proxy-style lost acknowledgement."""
import hashlib
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from runpod import dinov3_controller as c, dinov3_worker as w


class TransferTests(unittest.TestCase):
    def exchange(self, handler, action):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, cache, output = [root/n for n in ('source','cache','output')]
            source.mkdir()
            # More than three maximum chunks; not a tiny one-request fake.
            data = {'config.json': b'{"model_type":"dinov3_vit"}',
                    'model.safetensors': b'fixture' * (2 * 1024**2)}
            paths = {}
            for name, content in data.items():
                paths[name] = source/name
                paths[name].write_bytes(content)
            with patch.object(w,'TOKEN','local-test-only'), patch.object(w,'cache_root',return_value=cache), \
                    patch.object(w,'WEIGHTS',{'files_sha256':{n:hashlib.sha256(v).hexdigest() for n,v in data.items()}}), \
                    patch.object(w,'STATE',{}), patch.object(w,'READY',threading.Event()), patch.object(c,'OUT',output):
                server = ThreadingHTTPServer(('127.0.0.1',0),handler)
                thread = threading.Thread(target=server.serve_forever,daemon=True)
                thread.start()
                url='http://127.0.0.1:'+str(server.server_port)
                try:
                    action(url,paths,cache,output)
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join()

    def test_multichunk_roundtrip_with_lost_ack_and_no_overwrite(self):
        class Proxy(w.Handler):
            failed=False
            sizes=[]
            def do_PUT(self):
                if self.headers.get('User-Agent') != 'waferVLM-actions/0.1':
                    self.send_error(403)
                    return
                type(self).sizes.append(int(self.headers['Content-Length']))
                super().do_PUT()
            def send_response(self,code,message=None):
                # Worker committed bytes, but proxy lost its first response.
                if code==204 and not type(self).failed:
                    type(self).failed=True
                    code=502
                super().send_response(code,message)
        def action(url,paths,cache,output):
            with patch.object(c.time,'sleep'):
                c.upload_weights(url,'local-test-only',paths,{})
            self.assertTrue(w.READY.is_set())
            self.assertGreater(len(Proxy.sizes),4)
            self.assertLessEqual(max(Proxy.sizes),c.UPLOAD_CHUNK_BYTES)
            for name,path in paths.items():
                self.assertEqual(path.read_bytes(),(cache/name).read_bytes())
            self.assertEqual(json.loads((output/'transfer.json').read_text())['status'],'acknowledged')
        self.exchange(Proxy,action)

    def test_http_failure_is_diagnostic_without_token_or_response_body(self):
        class Reject(w.Handler):
            def do_PUT(self): self.send_error(413,'unsafe response body')
        def action(url,paths,cache,output):
            with self.assertRaises(urllib.error.HTTPError):
                c.upload_weights(url,'local-test-only',paths,{})
            text=(output/'transfer.json').read_text()
            self.assertEqual(json.loads(text)['http_status'],413)
            self.assertNotIn('local-test-only',text)
            self.assertNotIn('unsafe response body',text)
            self.assertFalse(w.READY.is_set())
        self.exchange(Reject,action)

    def test_gaps_oversized_chunks_and_bad_hashes_never_mark_ready(self):
        def action(url,paths,cache,output):
            for span,chunk_hash in [('bytes 1-1/2',hashlib.sha256(b'x').hexdigest()),
                                    ('bytes 0-0/1','0'*64),
                                    ('bytes 0-0/999999999',hashlib.sha256(b'x').hexdigest())]:
                req=urllib.request.Request(url+'/weights/model.safetensors',data=b'x',method='PUT',
                    headers={'Authorization':'Bearer local-test-only','Content-Range':span,
                             'X-Chunk-SHA256':chunk_hash})
                with self.assertRaises(urllib.error.HTTPError): urllib.request.urlopen(req)
            self.assertFalse(w.READY.is_set())
            self.assertFalse((cache/'model.safetensors').exists())
        self.exchange(w.Handler,action)

    def test_archive_download_uses_the_same_explicit_client_header(self):
        import io
        from unittest.mock import Mock
        payload=b'archive fixture'
        response=io.BytesIO(payload)
        response.status=200
        opener=Mock()
        opener.open.return_value=response
        with tempfile.TemporaryDirectory() as tmp, patch.object(c,'OUT',Path(tmp)), \
                patch.object(c.urllib.request,'build_opener',return_value=opener):
            archive=c.download('https://example.invalid','test-only',
                               {'size':len(payload),'sha256':hashlib.sha256(payload).hexdigest()})
            self.assertEqual(archive.read_bytes(),payload)
        req=opener.open.call_args.args[0]
        self.assertEqual(req.get_header('User-agent'),'waferVLM-actions/0.1')


if __name__=='__main__': unittest.main()
