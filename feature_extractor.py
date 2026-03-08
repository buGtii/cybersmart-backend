"""
CyberSmart - Feature Extractor v5
Outputs exactly 56 URL-structure features matching the Kaggle dataset columns 1-56.
NO page fetching, NO WHOIS, NO DNS — pure URL string analysis only.
This matches what the model was trained on perfectly.
"""

import re
import math
import urllib.parse
import ipaddress

SHORTENERS = {
    "bit.ly","tinyurl.com","goo.gl","ow.ly","t.co","is.gd","buff.ly",
    "adf.ly","shorte.st","clck.ru","rebrand.ly","cutt.ly","rb.gy",
    "shorturl.at","tiny.cc","bit.do","bc.vc"
}

BRANDS = [
    "paypal","amazon","google","microsoft","apple","netflix","ebay",
    "facebook","instagram","twitter","linkedin","dropbox","spotify",
    "chase","wellsfargo","bankofamerica","citibank","visa","mastercard",
    "dhl","fedex","ups","usps","irs","steam","discord","whatsapp",
    "yahoo","outlook","office365","binance","coinbase"
]

PHISH_KEYWORDS = [
    "login","signin","verify","update","secure","account","banking",
    "confirm","password","credential","support","helpdesk","suspend",
    "locked","unusual","urgent","alert","free","winner","prize",
    "webscr","wallet","crypto","bitcoin","lucky","bonus","click",
    "verify","validation","authenticate","recover","restore","unlock"
]

SUSPICIOUS_TLDS = {
    "xyz","top","club","work","gq","ml","cf","ga","tk","pw",
    "click","link","online","site","website","space","fun",
    "bid","win","racing","stream","party","trade","date","review",
    "loan","download","cricket","accountant","science","faith"
}


def _norm(url):
    url = url.strip()
    if not url.startswith(("http://","https://","ftp://")):
        url = "http://" + url
    return url

def _entropy(s):
    if not s: return 0.0
    freq = {}
    for c in s: freq[c] = freq.get(c,0)+1
    n = len(s)
    return -sum((f/n)*math.log2(f/n) for f in freq.values())


class FeatureExtractor:
    """
    Extracts 56 URL-structure features.
    Column names and order exactly match Kaggle dataset columns 1-56.
    """

    FEATURE_NAMES = [
        "length_url","length_hostname","ip","nb_dots","nb_hyphens",
        "nb_at","nb_qm","nb_and","nb_or","nb_eq",
        "nb_underscore","nb_tilde","nb_percent","nb_slash","nb_star",
        "nb_colon","nb_comma","nb_semicolumn","nb_dollar","nb_space",
        "nb_www","nb_com","nb_dslash","http_in_path","https_token",
        "ratio_digits_url","ratio_digits_host","punycode","port",
        "tld_in_path","tld_in_subdomain","abnormal_subdomain",
        "nb_subdomains","prefix_suffix","random_domain","shortening_service",
        "path_extension","nb_redirection","nb_external_redirection",
        "length_words_raw","char_repeat","shortest_words_raw",
        "shortest_word_host","shortest_word_path","longest_words_raw",
        "longest_word_host","longest_word_path","avg_words_raw",
        "avg_word_host","avg_word_path","phish_hints",
        "domain_in_brand","brand_in_subdomain","brand_in_path",
        "suspecious_tld","statistical_report",
    ]

    def __init__(self, timeout=4):
        self.timeout = timeout   # kept for API compatibility, unused here

    def extract(self, raw_url):
        url    = _norm(raw_url)
        p      = urllib.parse.urlparse(url)
        host   = (p.hostname or "").lower()
        path   = p.path  or ""
        query  = p.query or ""
        full   = url
        fl     = full.lower()

        # ── helpers ───────────────────────────────────────────────────────────
        parts      = host.split(".")
        tld        = parts[-1] if parts else ""
        domain     = parts[-2] if len(parts)>=2 else host
        subs       = parts[:-2] if len(parts)>2 else []
        sub_str    = ".".join(subs)
        path_l     = path.lower()

        # strip scheme for double-slash check
        after_scheme = full[full.find("//")+2:] if "//" in full else full

        # word tokenisation
        raw_words  = [w for w in re.split(r'[\W_]+', fl)       if w]
        host_words = [w for w in re.split(r'[\W_]+', host)     if w]
        path_words = [w for w in re.split(r'[\W_]+', path_l)   if w]

        def wlen(lst): return [len(w) for w in lst] if lst else [0]

        f = []

        # 1  length_url
        f.append(float(len(full)))
        # 2  length_hostname
        f.append(float(len(host)))
        # 3  ip
        try:
            ipaddress.ip_address(host)
            f.append(1.0)
        except ValueError:
            f.append(1.0if re.match(r'^\d{1,3}(\.\d{1,3}){3}$',host) else 0.0)
        # 4  nb_dots
        f.append(float(full.count(".")))
        # 5  nb_hyphens
        f.append(float(full.count("-")))
        # 6  nb_at
        f.append(float(full.count("@")))
        # 7  nb_qm
        f.append(float(full.count("?")))
        # 8  nb_and
        f.append(float(full.count("&")))
        # 9  nb_or
        f.append(float(full.count("|")))
        # 10 nb_eq
        f.append(float(full.count("=")))
        # 11 nb_underscore
        f.append(float(full.count("_")))
        # 12 nb_tilde
        f.append(float(full.count("~")))
        # 13 nb_percent
        f.append(float(full.count("%")))
        # 14 nb_slash
        f.append(float(full.count("/")))
        # 15 nb_star
        f.append(float(full.count("*")))
        # 16 nb_colon
        f.append(float(full.count(":")))
        # 17 nb_comma
        f.append(float(full.count(",")))
        # 18 nb_semicolumn  ← dataset spells it this way
        f.append(float(full.count(";")))
        # 19 nb_dollar
        f.append(float(full.count("$")))
        # 20 nb_space
        f.append(float(full.count(" ") + full.count("%20")))
        # 21 nb_www
        f.append(float(fl.count("www")))
        # 22 nb_com
        f.append(float(fl.count(".com")))
        # 23 nb_dslash  (//)  after scheme
        f.append(float(after_scheme.count("//")))
        # 24 http_in_path
        f.append(1.0 if "http" in path_l else 0.0)
        # 25 https_token
        f.append(1.0 if "https" in path_l or "https" in query.lower() else 0.0)
        # 26 ratio_digits_url
        f.append(sum(c.isdigit() for c in full) / max(len(full),1))
        # 27 ratio_digits_host
        f.append(sum(c.isdigit() for c in host) / max(len(host),1))
        # 28 punycode
        f.append(1.0 if "xn--" in host else 0.0)
        # 29 port
        f.append(0.0 if p.port in (80,443,8080,8443,None) else 1.0)
        # 30 tld_in_path
        f.append(1.0 if tld and tld in path_l else 0.0)
        # 31 tld_in_subdomain
        f.append(1.0 if tld and tld in sub_str else 0.0)
        # 32 abnormal_subdomain
        f.append(1.0 if any(re.search(r'\d',s) or len(s)>15 for s in subs) else 0.0)
        # 33 nb_subdomains
        f.append(float(len(subs)))
        # 34 prefix_suffix  (hyphen in domain part)
        f.append(1.0 if "-" in domain else 0.0)
        # 35 random_domain  (high entropy)
        f.append(1.0 if _entropy(domain) > 3.5 else 0.0)
        # 36 shortening_service
        f.append(1.0 if host in SHORTENERS else 0.0)
        # 37 path_extension
        f.append(1.0 if re.search(r'\.(php|asp|aspx|jsp|cgi|exe|bat|sh)(\?|$)',path_l) else 0.0)
        # 38 nb_redirection  (//)
        f.append(float(after_scheme.count("//")))
        # 39 nb_external_redirection
        f.append(float(len(re.findall(r'https?://', path+query))))
        # 40 length_words_raw
        f.append(float(len(raw_words)))
        # 41 char_repeat
        f.append(1.0 if re.search(r'(.)\1{3,}', full) else 0.0)
        # 42 shortest_words_raw
        f.append(float(min(wlen(raw_words))))
        # 43 shortest_word_host
        f.append(float(min(wlen(host_words))))
        # 44 shortest_word_path
        f.append(float(min(wlen(path_words))))
        # 45 longest_words_raw
        f.append(float(max(wlen(raw_words))))
        # 46 longest_word_host
        f.append(float(max(wlen(host_words))))
        # 47 longest_word_path
        f.append(float(max(wlen(path_words))))
        # 48 avg_words_raw
        wl = wlen(raw_words); f.append(sum(wl)/max(len(wl),1))
        # 49 avg_word_host
        wl = wlen(host_words); f.append(sum(wl)/max(len(wl),1))
        # 50 avg_word_path
        wl = wlen(path_words); f.append(sum(wl)/max(len(wl),1))
        # 51 phish_hints
        f.append(float(sum(1 for k in PHISH_KEYWORDS if k in fl)))
        # 52 domain_in_brand
        f.append(1.0 if domain in BRANDS else 0.0)
        # 53 brand_in_subdomain
        f.append(1.0 if any(b in sub_str for b in BRANDS) else 0.0)
        # 54 brand_in_path
        f.append(1.0 if any(b in path_l for b in BRANDS) else 0.0)
        # 55 suspecious_tld
        f.append(1.0 if tld in SUSPICIOUS_TLDS else 0.0)
        # 56 statistical_report
        try:
            ipaddress.ip_address(host)
            is_ip = True
        except ValueError:
            is_ip = bool(re.match(r'^\d{1,3}(\.\d{1,3}){3}$', host))
        f.append(1.0 if is_ip or any(b in host for b in BRANDS) else 0.0)

        assert len(f) == 56, f"Expected 56 features, got {len(f)}"
        return [float(x) for x in f]

    def extract_with_names(self, raw_url):
        return dict(zip(self.FEATURE_NAMES, self.extract(raw_url)))


if __name__ == "__main__":
    fe = FeatureExtractor()
    urls = [
        ("https://www.google.com",                            "SAFE"),
        ("https://stackoverflow.com/questions/tagged/python", "SAFE"),
        ("http://paypal-secure-login.xyz/verify?user=admin",  "PHISHING"),
        ("http://192.168.1.1/admin/login.php",                "PHISHING"),
    ]
    print(f"{'URL':<55} {'FEATS':<6} {'phish_hints':<13} {'susp_tld':<10} {'prefix_suf'}")
    print("-"*90)
    for url, label in urls:
        f = fe.extract(url)
        d = fe.extract_with_names(url)
        print(f"{url[:54]:<55} {len(f):<6} {d['phish_hints']:<13} "
              f"{d['suspecious_tld']:<10} {d['prefix_suffix']}  [{label}]")
