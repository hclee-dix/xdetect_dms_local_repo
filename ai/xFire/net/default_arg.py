from types import SimpleNamespace

DEFAULT_ARGS = SimpleNamespace(**{
    'shard_id':0,
    'num_shards':1,
    'init_method':'tcp://localhost:9999',
    'opts':None,
})