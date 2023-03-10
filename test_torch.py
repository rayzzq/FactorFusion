from model import TorchModel, TorchModelCV
import hydra
from utils import create_fake_data

@hydra.main(config_path="./model/torch_module", config_name="sample_config")
def test_model_cv(cfg):
    train_df, test_df, Xcols, regYcols, clsYcols = create_fake_data()
    
    # cfg.params["Xcols"] = Xcols
    # cfg.params["regYcols"] = regYcols
    # # params["regYcols"] = []
    # # params["clsYcols"] = clsYcols
    # cfg.params["clsYcols"] = []

    
    # model = TorchModelCV.create_cv_with_single_param(cfg, cv=2)
    # model.fit(train_df)
    # res = model.predict(test_df)
    # print("")
    # print("fitted predict print")
    # print(res.head(10))
    # model.save_model(r"C:\Users\Peihan.li\Desktop\FactorFusion\save_models")
    
    model = TorchModelCV.load_model(r"C:\Users\Peihan.li\Desktop\FactorFusion\save_models")
    res = model.predict(test_df)
    print("reload predict print") 
    print(res.head(10))
    
    print("reload model print")
    print(model.params)
    print("")
    print(model.models)

@hydra.main(config_path="./model/torch_module", config_name="sample_config")
def test_model(cfg):
    train_df, test_df, Xcols, regYcols, clsYcols = create_fake_data()
    
    net_name = cfg.net_name
    params = cfg.params
    params["Xcols"] = Xcols
    params["regYcols"] = regYcols
    # params["regYcols"] = []
    # params["clsYcols"] = clsYcols
    params["clsYcols"] = []

    # print("-" * 10, "Configuation", "-" * 10)
    # for k, v in params.items():
    #     print('{:<20}  {:<20}'.format(str(k), str(v)))
    
    model = TorchModel(params, net_name)
    model.fit(train_df)
    res = model.predict(test_df)
    print("")
    print("fitted predict print")
    print(res.head(10))
    model.save_model("./test_torch_model.pkl")
    model = TorchModel.load_model("./test_torch_model.pkl")
    res = model.predict(test_df)
    print("reload predict print") 
    print(res.head(10))
    print("reload model print")
    print("")
    for k, v in model.params.items():
        print('{:<20}  {:<20}'.format(str(k), str(v)))
    
    print("")
    print(model.model)
    
    
    print("")
    print(model.Xcols)
    print(model.regYcols)
    print(model.clsYcols)
    
    print("")
    print(model.scaler)
    
if __name__ == "__main__":
    test_model_cv()
