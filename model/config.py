from easydict import EasyDict as edict

CONFIGS = {
    "pfnn":
    edict({
        "seed": 10,
        "gpu":True,
        "wandb":False,
        "batch_size": 32,
        "lr":0.003,
        "lasso_coef":0.01,
        "shuffle": True,
        "epoch": 20,
        "test_epoch":1,
        "num_joint": 31,
        "window_size": 60,
        "hidden_dim": 512, #hiddenlayer dimension
        "drop_prob": 0.7, #dropout prob
        "log_interval":10,#1 log per nth batch
        "save_dir":"./data/pfnn/nn/"
    })
}
