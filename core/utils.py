import os
import yaml


class Utility:
    @staticmethod
    def load_template(filepath="templates.yml", template="prompt"):
        base_path = os.path.dirname(__file__)  # folder where utils.py is
        full_path = os.path.join(base_path, filepath)
        with open(full_path, "r") as f:
            data = yaml.safe_load(f)
        if template == "napkin":
            return data["NAPKIN_TEMPLATE"]
        else:
            return data["PROMPT_TEMPLATE"]
