import yaml

with open('./wmodel/config.yaml', "r") as file:
    config = yaml.safe_load(file)

model = config["model"]
optimizer = config["optimizer"]
pipelines = config["pipelines"]
pipeline_cat_wrapper = config["pipelines"]["pipeline_cat_wrapper"]
pipeline_homotopy = config["pipelines"]["pipeline_homotopy"]
pipeline_quantile = config["pipelines"]["pipeline_quantile"]
pipeline_smallnans = config["pipelines"]["pipeline_smallnans"]
pipeline_tree = config["pipelines"]["pipeline_tree"]
selector_first = config["selector_first"]
selector_last = config["selector_last"]
utilities = config["utilities"]
woe = config["woe"]
