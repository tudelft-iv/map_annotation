import json
import os

template_dir = os.path.join(os.environ["MA_DIR"], "templates")

lanes_template = ["element_id"]

with open(os.path.join(template_dir, "lanes_attributes.json"), "w") as f:
    json.dump(lanes_template, f)
