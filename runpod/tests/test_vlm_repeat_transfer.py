import json
import os
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from unittest.mock import patch

from runpod import vlm_controller as controller, vlm_repeat_inputs as inputs
from runpod import vlm_repeat_worker as worker
from runpod.provider import RemoteHTTPError, request


class FreshInputDelivery(unittest.TestCase):
    def setUp(self):
        self.env=patch.dict(os.environ,WAFER_DEADLINE=str(time.time()+300))
        self.env.start();self.addCleanup(self.env.stop)
        self.result=patch.object(worker.w,'RESULT',dict(status='running',phase='inputs'))
        self.result.start();self.addCleanup(self.result.stop)
        self.token=patch.object(worker.w,'TOKEN','test-result-token')
        self.token.start();self.addCleanup(self.token.stop)
        self.server=ThreadingHTTPServer(('127.0.0.1',0),worker.InputHandler)
        self.server_thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.server_thread.start();self.addCleanup(self.close)
        self.url='http://127.0.0.1:'+str(self.server.server_port)
        self.answer=[]
        self.waiter=threading.Thread(target=lambda:self.answer.append(worker.request_input_url('seed42')),daemon=True)
        self.waiter.start()
        deadline=time.monotonic()+2
        while 'input_request' not in worker.w.RESULT and time.monotonic()<deadline:time.sleep(.01)
        self.desired=request(self.url+'/result',token='test-result-token')['input_request']

    def close(self):
        worker.INPUT_READY.set();self.waiter.join(2)
        self.server.shutdown();self.server.server_close();self.server_thread.join(2)

    def test_link_is_minted_after_worker_requests_it_and_never_returned(self):
        scoped='https://example.blob.core.windows.net/seed42?test-scope=one-minute'
        with patch.object(inputs,'inputs',return_value={'seed42':{}}), \
             patch.object(inputs,'signed_input_url',return_value=scoped) as mint,patch('builtins.print'):
            controller.deliver_input_url(self.url,'test-result-token',self.desired)
        self.waiter.join(2);self.assertEqual(self.answer,[scoped]);mint.assert_called_once_with('seed42')
        result=request(self.url+'/result',token='test-result-token')
        self.assertNotIn('input_request',result);self.assertNotIn(scoped,json.dumps(result))
        with self.assertRaises(RemoteHTTPError) as raised:
            request(self.url+'/input-url',token='test-result-token',method='POST',body={**self.desired,'url':scoped})
        self.assertEqual(raised.exception.status,409)

    def test_auth_nonce_destination_and_size_gates(self):
        valid={**self.desired,'url':'https://example.blob.core.windows.net/seed42'}
        for token,body,status in [('wrong',valid,403),('test-result-token',dict(valid,nonce='b'*32),409),
                                 ('test-result-token',dict(valid,url='https://evil.example/data'),400),
                                 ('test-result-token',dict(valid,url=valid['url']+'x'*16000),400)]:
            with self.subTest(status=status),self.assertRaises(RemoteHTTPError) as raised:
                request(self.url+'/input-url',token=token,method='POST',body=body)
            self.assertEqual(raised.exception.status,status)
        self.assertEqual(self.answer,[])


if __name__=='__main__':unittest.main()
