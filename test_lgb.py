from model import LgbRegressor
from utils import create_fake_data


def test_model():
    train_df, test_df, Xcols, regYcols, clsYcols = create_fake_data()

    model = LgbRegressor(params={})
    model.set_cols(Xcols, regYcols)

    model.fit(train_df[Xcols], train_df[regYcols[0]])

    Y_pred = model.predict(test_df[Xcols])

    print(Y_pred.shape)

    model.save_model("test_lgb_model.joblib")

    model = LgbRegressor.load_model("test_lgb_model.joblib")

    Y_pred = model.predict(test_df[Xcols])

    print(Y_pred.shape)


if __name__ == "__main__":
    test_model()
