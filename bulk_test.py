import requests

API = "http://localhost:5000/predict"

tests = [
    # ── OBVIOUS PHISHING (25) ──────────────────────────────────────────────
    ("http://paypal-secure-login.xyz/verify?user=admin&token=abc123",       "phishing"),
    ("http://amazon-account-suspended.tk/confirm/login.php",                "phishing"),
    ("http://192.168.1.100/admin/login.php",                                "phishing"),
    ("http://microsoft-security-alert.top/update/account",                  "phishing"),
    ("http://apple-id-locked.gq/unlock?session=xyz123",                     "phishing"),
    ("http://secure-banking-update.ml/account/verify",                      "phishing"),
    ("http://www.paypal.com.secure-login.xyz/webscr",                       "phishing"),
    ("http://netflix-billing-update.pw/payment/confirm",                    "phishing"),
    ("http://login-facebook-secure.club/checkpoint/verify",                 "phishing"),
    ("http://appleid.apple.com.phishing-test.xyz/signin",                   "phishing"),
    ("http://google.com.account-verify.top/login?redirect=gmail",           "phishing"),
    ("http://secure.amazon.com.update-billing.ml/signin",                   "phishing"),
    ("http://10.0.0.1/setup/admin/login",                                   "phishing"),
    ("http://172.16.254.1/cgi-bin/luci",                                    "phishing"),
    ("http://instagram-verify-account.xyz/confirm?user=victim",             "phishing"),
    ("http://irs-tax-refund-2024.tk/claim?ssn=required",                    "phishing"),
    ("http://dhl-delivery-failed.ml/reschedule?parcel=12345",               "phishing"),
    ("http://covid-relief-payment.gq/apply?id=citizen123",                  "phishing"),
    ("http://steam-free-games-offer.pw/claim?user=gamer",                   "phishing"),
    ("http://bankofamerica-secure-alert.xyz/signin/verify",                 "phishing"),
    ("http://bit.ly/3xSecureLoginNow",                                      "phishing"),
    ("http://tinyurl.com/verify-your-account",                              "phishing"),
    ("http://www.wellsfargo.com.account-locked.club/restore",               "phishing"),
    ("http://chase-bank-security-alert.top/update?account=checking",        "phishing"),
    ("http://discount-winner-prize-claim.xyz/free?voucher=WIN100",          "phishing"),

    # ── SAFE — MAJOR WEBSITES (20) ────────────────────────────────────────
    ("https://www.google.com",                                               "safe"),
    ("https://www.github.com",                                               "safe"),
    ("https://stackoverflow.com/questions/tagged/python",                    "safe"),
    ("https://www.wikipedia.org/wiki/Phishing",                              "safe"),
    ("https://www.amazon.com/dp/B08N5WRWNW",                                 "safe"),
    ("https://www.youtube.com/watch?v=dQw4w9WgXcQ",                         "safe"),
    ("https://docs.python.org/3/library/urllib.html",                        "safe"),
    ("https://www.linkedin.com/in/username",                                 "safe"),
    ("https://www.reddit.com/r/netsec",                                      "safe"),
    ("https://www.microsoft.com/en-us/windows",                              "safe"),
    ("https://www.apple.com/iphone",                                         "safe"),
    ("https://www.netflix.com/browse",                                       "safe"),
    ("https://www.twitter.com/home",                                         "safe"),
    ("https://www.instagram.com/explore",                                    "safe"),
    ("https://www.facebook.com",                                             "safe"),
    ("https://www.dropbox.com/login",                                        "safe"),
    ("https://www.spotify.com/us/account/overview",                         "safe"),
    ("https://www.paypal.com/signin",                                        "safe"),
    ("https://accounts.google.com/signin",                                   "safe"),
    ("https://login.microsoftonline.com",                                    "safe"),

    # ── TRICKY EDGE CASES (10) ────────────────────────────────────────────
    ("http://www.amazon.com",                                                "safe"),   # HTTP but real
    ("https://www.paypal.com/us/webapps/mpp/home",                          "safe"),   # has webapps in path
    ("https://support.google.com/accounts/answer/1066447",                  "safe"),   # support subdomain
    ("https://mail.google.com/mail/u/0/#inbox",                             "safe"),   # mail subdomain
    ("https://developer.apple.com/documentation/swift",                     "safe"),   # developer subdomain
    ("https://aws.amazon.com/ec2/pricing",                                  "safe"),   # aws subdomain
    ("https://docs.microsoft.com/en-us/azure/",                             "safe"),   # docs subdomain
    ("https://www.chase.com/personal/banking",                              "safe"),   # real bank
    ("https://www.wellsfargo.com/mortgage/",                                "safe"),   # real bank
    ("https://store.steampowered.com/app/1091500",                          "safe"),   # steam store
]

# ── Run tests ────────────────────────────────────────────────────────────────
correct   = 0
phish_ok  = 0
safe_ok   = 0
phish_tot = sum(1 for _, e in tests if e == "phishing")
safe_tot  = sum(1 for _, e in tests if e == "safe")

print(f"\n{'URL':<58} {'GOT':<10} {'EXP':<10} {'PROB':<8} RISK")
print("=" * 100)

for url, expected in tests:
    try:
        r    = requests.post(API, json={"url": url}, timeout=15).json()
        got  = r.get("label", "error")
        risk = r.get("risk_level", "?")
        prob = r.get("probability", 0)
        ok   = "✅" if got == expected else "❌"

        if got == expected:
            correct += 1
            if expected == "phishing": phish_ok += 1
            else:                      safe_ok  += 1

        print(f"{url[:57]:<58} {got:<10} {expected:<10} {prob:<8.3f} {risk}  {ok}")

    except Exception as e:
        print(f"{url[:57]:<58} ERROR: {e}")

# ── Summary ──────────────────────────────────────────────────────────────────
total = len(tests)
print("=" * 100)
print(f"\n  Overall  : {correct}/{total} ({correct/total*100:.0f}%)")
print(f"  Phishing : {phish_ok}/{phish_tot} caught  ({phish_ok/phish_tot*100:.0f}%)")
print(f"  Safe     : {safe_ok}/{safe_tot} correct ({safe_ok/safe_tot*100:.0f}%)")

if correct/total >= 0.90:
    print("\n  🟢 EXCELLENT — Model is production ready!")
elif correct/total >= 0.80:
    print("\n  🟡 GOOD — Minor tuning may help on edge cases")
elif correct/total >= 0.65:
    print("\n  🟠 FAIR — Consider retraining with more epochs")
else:
    print("\n  🔴 NEEDS WORK — Retrain with train_on_dataset.py")
