import numpy as np
from PIL import Image
from fred.io import load_prompt_depth_as_tensor, save_depth_u16_qmm_png

def test_save_filename_without_parent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    save_depth_u16_qmm_png(np.array([[1.0]], dtype=np.float32), 'depth.png')
    assert np.asarray(Image.open('depth.png'))[0, 0] == 4000

def test_qmm_round_trip(tmp_path):
    path = tmp_path / 'depth.png'
    source = np.array([[0.25, 1.5]], dtype=np.float32)
    save_depth_u16_qmm_png(source, path)
    loaded = load_prompt_depth_as_tensor(str(path), 'qmm', 'cpu')
    np.testing.assert_allclose(loaded.numpy()[0, 0], source, atol=0.00025)
