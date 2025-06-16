import json
import os

class Translator:
    def __init__(self, language="en"):
        self.language = language
        self.resources = self._load_resources()
    
    def _load_resources(self):
        resource_path = os.path.join("resources", f"{self.language}.json")
        try:
            with open(resource_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            # Fallback to English if the language file doesn't exist
            with open(os.path.join("resources", "en.json"), 'r', encoding='utf-8') as f:
                return json.load(f)
            
    # Get a translation by dot-separated key path (e.g., 'sidebar.title')
    def get(self, key_path):
        keys = key_path.split('.')
        value = self.resources
        for key in keys:
            value = value.get(key, {})
        return value if isinstance(value, str) else key_path 