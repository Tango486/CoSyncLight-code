import argparse
import argparse
import json

def parse_filters(filters_str):
    try:
        filters = json.loads(filters_str)
        return filters
 
    except json.JSONDecodeError:
        raise ValueError("Filters must be in valid JSON format.")

class AdjGen_Config():
    def __init__(self):
        self.hidden_state_features_size = 64
 
adjgen_get_config = AdjGen_Config()
 