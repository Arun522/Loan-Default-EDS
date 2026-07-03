"""Embed dashboard_data.json into the dashboard template -> dashboard.html (self-contained)."""
import json

data = open("dashboard_data.json").read()
tpl = open("dashboard_template.html").read()
out = tpl.replace("/*__DATA__*/{}", data)
open("dashboard.html", "w").write(out)
print("dashboard.html written,", len(out) // 1024, "KB")
