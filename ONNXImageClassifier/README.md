# ONNXImageClassifier — Image Lens

A Swift 6 / SwiftUI app for iOS 17+ that classifies photos entirely on device using real MobileNetV2 inference through Microsoft's ONNX Runtime. Select an image with the system photo picker, analyze it, and see the three most likely ImageNet classes with confidence scores. No server, account, analytics, or mock predictions.

## Architecture

- `App/`: SwiftUI entry point.
- `Features/ImageClassification/Views/`: adaptive native UI, photo picker, bounded asynchronous preview, loading/error states, ranked results, and reset.
- `Features/ImageClassification/ViewModels/`: main-actor observable state and selection/analysis cancellation. `ImageClassifying` supports dependency injection.
- `Services/MachineLearning/ImagePreprocessor`: orientation-aware decoding, resizing, cropping, RGB extraction, and Float32 NCHW conversion.
- `Services/MachineLearning/ONNXModelService`: actor that owns and reuses the ORT environment/session. Preprocessing and synchronous inference execute on the actor executor, outside the main actor. Names are read from the loaded session; output shape/type and buffer size are checked on every run.
- `Services/MachineLearning/ModelOutputProcessor`: validates logits, computes stable softmax, and returns the top three classes with deterministic tie ordering.
- `Resources/`: the actual model, 1,000 ordered labels, and model license.

Only one inference runs at a time per service. ORT borrows an explicitly retained mutable input buffer for the synchronous run; output floats are copied before ORT values leave scope. Reset cancels work and uses a generation token to prevent stale results updating the UI.

## Dependencies

The [official Objective-C integration](https://onnxruntime.ai/docs/get-started/with-obj-c.html) provides a Swift-importable API through CocoaPods. This project uses `onnxruntime-objc` **1.30.0**, with its matching `onnxruntime-c` binary, pinned in both the Podfile and lockfile. CPU execution is used for predictable operator coverage; two intra-op threads and full graph optimization are enabled. No third-party UI dependencies are used.

## Model provenance and verified contract

[ONNX Model Zoo MobileNetV2 opset 7](https://github.com/onnx/models/tree/main/validated/vision/classification/mobilenet), distributed under Apache 2.0. The bundled model is downloaded directly from [the official repository's LFS media](https://media.githubusercontent.com/media/onnx/models/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx).

| Property | Verified value |
| --- | --- |
| Input name | `data` |
| Input type and shape | Float32 `[1, 3, 224, 224]`, NCHW |
| Output name | `mobilenetv20_output_flatten0_reshape0` |
| Output type and shape | Float32 `[1, 1000]` |
| Output semantics | Unnormalized scores; graph has no Softmax |
| Opset | 7 |
| SHA-256 | `c1c513582d56afceff8516c73804e484c81c6a830712ab6d682253f4a3cd042f` |

Names, dimensions, types, final graph operators, and integrity were inspected with ONNX and ONNX Runtime, rather than inferred from the filename. `Scripts/verify_model.py` reproduces inspection, graph validation, and real sample inference. Its recorded results are in `Scripts/model-verification.json`.

Labels preserve the exact line order of [Model Zoo synset.txt](https://github.com/onnx/models/blob/main/validated/vision/classification/synset.txt), with only the synset identifier removed. The source model documentation and reference preprocessing are saved in `Scripts/ModelSource.md` and `Scripts/reference_preprocess.py`. The latter is a provenance reference, not an application dependency.

## Preprocessing and output

1. ImageIO decodes a thumbnail bounded to a maximum side of 2,048 pixels and applies EXIF orientation, including mirrored orientations.
2. Resize proportionally so the shorter side is 256; round dimensions and center crop 224×224, following the Model Zoo reference.
3. Render into an sRGB RGBX buffer, extract RGB, and divide each channel by 255.
4. Normalize with mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`, explicitly required by this model's documentation.
5. Pack channel planes into Float32 `[1, 3, 224, 224]`.
6. Run ONNX Runtime, validate Float32 `[1, 1000]`, subtract the maximum logit before softmax, and rank all classes before displaying three.

Core Image/ImageIO interpolation and bounded initial decoding can produce small differences from Pillow/MXNet reference pixels. Confidence values can consequently differ slightly across pipelines. The displayed preview shows the original photo rather than the model's center crop.

## Setup and launch

Requirements: macOS, Xcode with an installed iOS Simulator runtime, Ruby/CocoaPods, and network access for the initial dependency installation. The model is already included; inference needs no network connection. PhotosPicker uses system-granted selection access and does not require broad photo-library permission.

```sh
cd /Users/diasatudinov/Developer/ONNXImageClassifier
pod install
open ONNXImageClassifier.xcworkspace
```

Select the **ONNXImageClassifier** scheme, choose an iPhone Simulator, and press **⌘R**. Choose a photo, then tap **Analyze**. To use the included fixture, drag `ONNXImageClassifierTests/Fixtures/dog.jpg` into the running Simulator to add it to Photos. For a physical device, set your development team in Signing & Capabilities first.

The Xcode project and shared scheme are checked into the project; regeneration is unnecessary. `Scripts/generate_project.rb` documents initial generation; if you deliberately regenerate, run `pod install` afterward to restore integration.

## Build and test

```sh
xcodebuild -workspace ONNXImageClassifier.xcworkspace \
  -scheme ONNXImageClassifier \
  -destination 'platform=iOS Simulator,name=iPhone 18 Pro,OS=27.0' \
  -derivedDataPath build test
```

Choose a destination available on your machine using `xcrun simctl list devices available`.

Tests cover RGB normalization, input dimensions, EXIF rotation, center cropping, invalid images/outputs, stable softmax, ranked sorting/ties, session initialization, missing resources, and real inference on the [PyTorch Hub Samoyed fixture](https://github.com/pytorch/hub/blob/master/images/dog.jpg). The integration test requires class 258 (Samoyed) to rank first with confidence greater than 50%.

For independent Python verification, use a virtual environment with `onnx`, `onnxruntime`, `pillow`, and `numpy`, then run `python Scripts/verify_model.py`. These tools are not required to build or run the iOS app.

### Verification results

- Xcode 27.0: initial simulator `build-for-testing` succeeded.
- Python ONNX checker and real CPU inference passed: Samoyed 83.09%, Pomeranian 5.33%, Great Pyrenees 2.86%.
- Final `xcodebuild test` on iPhone 18 Pro / iOS 27.0: **10 tests passed, 0 failures**. The final test build reported no Swift concurrency warnings. Result bundle: `build/ClassificationTests.xcresult`.
- Installed and launched the app on the iPhone 18 Pro simulator; visually checked the initial light-appearance screen.
- The initialization and integration tests confirm the actual model and ordered labels are bundled in the application. Simulator Core Image emitted a pixel-buffer fallback diagnostic; inference and all pixel/normalization assertions passed.

## Known limitations

- ImageNet has a fixed set of 1,000 classes; this is classification, not object detection. A center crop can omit subjects near the edge. Scores are not calibrated guarantees.
- Cancellation is checked around preprocessing and inference. An already-running synchronous ORT call finishes before cancellation is observed; stale results are discarded.
- The app loads selected encoded photo data in memory, while decoded preview and inference buffers are bounded. Very large encoded files still consume memory.
- CPU inference and first-run model initialization vary by device. Core ML acceleration and quantization are not enabled.
- Physical-device signing, release distribution, VoiceOver walkthrough, and comprehensive device/appearance testing require further validation.
