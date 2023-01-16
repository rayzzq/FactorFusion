from model import SklearnRegressor

from utils import create_fake_data


def test_model():
    train_df, test_df, Xcols, regYcols, clsYcols = create_fake_data()
    params = {"n_jobs":20,
              "eps":1e-5,
              "max_iter":1000,}
    
    model = SklearnRegressor(params)
    model.set_cols(Xcols, regYcols)

    model.fit(train_df[Xcols], train_df[regYcols[0]])

    Y_pred = model.predict(test_df[Xcols])

    print(Y_pred.shape)

    model.save_model("test_sklearn_model.joblib")

    model = SklearnRegressor.load_model("test_lgb_model.joblib")

    Y_pred = model.predict(test_df[Xcols])

    print(Y_pred.shape)


if __name__ == "__main__":
    test_model()
