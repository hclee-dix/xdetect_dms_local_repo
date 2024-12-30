import yaml,copy, ast

class PipelineConfig():
    def __init__(self, kwargs):
        self.cfg = {
            'config':'',
            **kwargs,
            'opt':{}
        }

def get_pipeline_cfg(kwargs):
    a = PipelineConfig(kwargs)
    return a.cfg
    
def merge_cfg(args):
    # load config
    with open(args['config']) as f:
        pred_config = yaml.safe_load(f)

    def merge(cfg, arg):
        # update cfg from arg directly
        merge_cfg = copy.deepcopy(cfg)
        for k, v in cfg.items():
            if k in arg:
                merge_cfg[k] = arg[k]
            else:
                if isinstance(v, dict):
                    merge_cfg[k] = merge(v, arg)
            
            if(k == 'model_dir'):
                merge_cfg[k] = '/'.join([args['model_path'],v]) # type: ignore

        return merge_cfg

    def merge_opt(cfg, arg):
        merge_cfg = copy.deepcopy(cfg)
        # merge opt
        if 'opt' in arg.keys() and arg['opt']:
            for name, value in arg['opt'].items(
            ):  # example: {'MOT': {'batch_size': 3}}
                if name not in merge_cfg.keys():
                    print("No", name, "in config file!")
                    continue
                for sub_k, sub_v in value.items():
                    if sub_k not in merge_cfg[name].keys():
                        print("No", sub_k, "in config file of", name, "!")
                        continue
                    merge_cfg[name][sub_k] = sub_v

        return merge_cfg

    args_dict = copy.deepcopy(args)
    pred_config = merge(pred_config, args_dict)
    pred_config = merge_opt(pred_config, args_dict)
    
    return pred_config

def print_arguments(cfg):
    print('-----------  Running Arguments -----------')
    args = copy.deepcopy(cfg)
    buffer = yaml.dump(args)
    print(buffer)
    print('------------------------------------------')
