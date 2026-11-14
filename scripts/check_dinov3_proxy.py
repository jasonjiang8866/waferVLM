"""Two anonymous GETs to a verified deleted Pod route; no compute or secrets."""
import json
import urllib.error
import urllib.request

URL = 'https://vvwvq9qu6267as-8888.proxy.runpod.net/result'


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        return None


def main():
    rows = []
    for label, headers in [('python-default', {}), ('project-client', {'User-Agent': 'waferVLM-actions/0.1'})]:
        req = urllib.request.Request(URL, headers=headers, method='GET')
        try:
            response = urllib.request.build_opener(NoRedirect).open(req, timeout=30)
        except urllib.error.HTTPError as exc:
            response = exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            rows.append({'client': label, 'error_type': type(exc).__name__})
            continue
        with response:
            raw = response.read(16000)
            rows.append({'client': label, 'http_status': response.code,
                         'mentions_cloudflare': b'cloudflare' in raw.lower()})
    print(json.dumps({'scope': 'deleted Pod proxy route; not a GPU or authenticated transfer test',
                      'requests': rows}, indent=2))


if __name__ == '__main__':
    main()
