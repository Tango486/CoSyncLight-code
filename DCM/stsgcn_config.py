import json

def parse_filters(filters_str):
    try:
        filters = json.loads(filters_str)
        return filters
 
    except json.JSONDecodeError:
        raise ValueError("Filters must be in valid JSON format.")

class STSGCN_Config():
    def __init__(self):
        self.module_type = 'individual'
        self.act_type = 'GLU'
        self.temporal_emb = True
        self.spatial_emb = True
        self.first_layer_embedding_size = 64
        self.filters = [[64, 64, 64], [64, 64, 64]]
        self.batch_size = 1
        self.epochs = 200
        self.max_update_factor = 1
        self.device = 'cuda'
        self.num_for_predict = 1
 
stsgcn_get_config = STSGCN_Config()
 