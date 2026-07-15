"""Unified lexical URL feature extractor - works on a raw URL string alone
(no HTML/DNS/WHOIS lookups), so it can be applied identically to any source.
"""
import re
import math
from urllib.parse import urlparse
from collections import Counter

SUSPICIOUS_WORDS = ['login', 'secure', 'account', 'update', 'confirm', 'signin',
                    'bank', 'verify', 'webscr', 'password', 'ebayisapi']
IP_RE = re.compile(r'^(\d{1,3}\.){3}\d{1,3}$')


def _entropy(s):
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def extract_url_features(raw_url):
    u = str(raw_url).strip()
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9+.\-]*://', u):
        u_parse = 'http://' + u
    else:
        u_parse = u
    try:
        p = urlparse(u_parse)
    except Exception:
        p = urlparse('http://invalid')
    host = p.netloc.split('@')[-1].split(':')[0] if p.netloc else ''
    path = p.path or ''
    query = p.query or ''
    full = u

    digits = sum(c.isdigit() for c in full)
    letters = sum(c.isalpha() for c in full)
    length = len(full)

    feats = {
        'url_length': length,
        'hostname_length': len(host),
        'path_length': len(path),
        'query_length': len(query),
        'qty_dot': full.count('.'),
        'qty_hyphen': full.count('-'),
        'qty_underline': full.count('_'),
        'qty_slash': full.count('/'),
        'qty_questionmark': full.count('?'),
        'qty_equal': full.count('='),
        'qty_at': full.count('@'),
        'qty_and': full.count('&'),
        'qty_exclamation': full.count('!'),
        'qty_space': full.count(' '),
        'qty_tilde': full.count('~'),
        'qty_comma': full.count(','),
        'qty_plus': full.count('+'),
        'qty_asterisk': full.count('*'),
        'qty_hashtag': full.count('#'),
        'qty_percent': full.count('%'),
        'qty_digits': digits,
        'qty_letters': letters,
        'digit_ratio': round(digits / length, 4) if length else 0.0,
        'qty_dot_host': host.count('.'),
        'qty_hyphen_host': host.count('-'),
        'qty_subdomains': max(host.count('.') - 1, 0),
        'has_ip_host': int(bool(IP_RE.match(host))),
        'has_https': int(p.scheme == 'https'),
        'has_http_token_in_path': int('http' in path.lower() or 'https' in path.lower()),
        'has_at_symbol': int('@' in full),
        'has_double_slash_redirect': int('//' in (path + query)),
        'has_www': int(host.lower().startswith('www.')),
        'tld_length': len(host.split('.')[-1]) if '.' in host else 0,
        'shortest_word_path': (min((len(w) for w in re.split(r'[\/\-_.?=&]', path) if w), default=0)),
        'longest_word_path': (max((len(w) for w in re.split(r'[\/\-_.?=&]', path) if w), default=0)),
        'url_entropy': round(_entropy(full), 4),
        'suspicious_word_count': sum(w in full.lower() for w in SUSPICIOUS_WORDS),
    }
    return feats


FEATURE_NAMES = list(extract_url_features('http://example.com/path?q=1').keys())
