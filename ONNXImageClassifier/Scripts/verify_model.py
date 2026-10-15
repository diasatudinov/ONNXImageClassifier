"""Run with: python3 -m pip install onnx onnxruntime pillow numpy; python3 Scripts/verify_model.py"""
from pathlib import Path
import hashlib
import json
import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image, ImageOps

root = Path(__file__).resolve().parents[1]
path = root / 'ONNXImageClassifier/Resources/Models/mobilenetv2.onnx'
model = onnx.load(path)
onnx.checker.check_model(model)
session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
inputs, outputs = session.get_inputs(), session.get_outputs()
assert [(x.name, x.shape, x.type) for x in inputs] == [('data', [1, 3, 224, 224], 'tensor(float)')]
assert [(x.name, x.shape, x.type) for x in outputs] == [('mobilenetv20_output_flatten0_reshape0', [1, 1000], 'tensor(float)')]
assert model.graph.node[-1].op_type == 'Reshape'
assert not any(n.op_type == 'Softmax' for n in model.graph.node)
labels = (root / 'ONNXImageClassifier/Resources/Labels/imagenet_labels.txt').read_text().splitlines()
assert len(labels) == 1000
image = ImageOps.exif_transpose(Image.open(root / 'ONNXImageClassifierTests/Fixtures/dog.jpg')).convert('RGB')
ratio = 256 / min(image.size)
image = image.resize(tuple(round(d * ratio) for d in image.size), Image.Resampling.BILINEAR)
x, y = image.width // 2 - 112, image.height // 2 - 112
image = image.crop((x, y, x + 224, y + 224))
tensor = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255
for c, (mean, std) in enumerate(zip([.485,.456,.406], [.229,.224,.225])):
    tensor[c] = (tensor[c] - mean) / std
scores = session.run(None, {inputs[0].name: tensor[None]})[0][0]
probabilities = np.exp(scores - scores.max()); probabilities /= probabilities.sum()
top = probabilities.argsort()[-3:][::-1]
assert top[0] == 258
report = {
    'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    'inputs': [{'name': x.name, 'shape': x.shape, 'type': x.type} for x in inputs],
    'outputs': [{'name': x.name, 'shape': x.shape, 'type': x.type} for x in outputs],
    'opset': model.opset_import[0].version,
    'final_operator': model.graph.node[-1].op_type,
    'top3': [{'index': int(i), 'label': labels[i], 'confidence': float(probabilities[i])} for i in top]
}
print(json.dumps(report, indent=2))
(root / 'Scripts/model-verification.json').write_text(json.dumps(report, indent=2) + '\n')
