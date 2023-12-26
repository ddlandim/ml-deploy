import pytest
import model as model_pkg


@pytest.fixture
def model():
    return model_pkg.load_model()

def test_inference(model):
    model_pkg.validate_model(model, model_pkg.TEST_X, model_pkg.TEST_Y)