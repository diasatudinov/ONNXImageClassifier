import Foundation
import onnxruntime_objc

/// The actor owns all non-Sendable ORT objects and serializes inference off the main actor.
actor ONNXModelService: ImageClassifying {
    private let bundle: Bundle
    private var environment: ORTEnv?
    private var session: ORTSession?
    private var inputName = ""
    private var outputName = ""
    private var labels: [String] = []
    private let preprocessor = ImagePreprocessor()
    private let outputProcessor = ModelOutputProcessor()

    init(bundle: Bundle = .main) { self.bundle = bundle }

    func initialize() throws {
        guard session == nil else { return }
        guard let modelURL = bundle.url(forResource: "mobilenetv2", withExtension: "onnx") else {
            throw ClassificationError.missingResource("MobileNetV2 model")
        }
        guard let labelsURL = bundle.url(forResource: "imagenet_labels", withExtension: "txt") else {
            throw ClassificationError.missingResource("ImageNet labels")
        }
        let loadedLabels = try String(contentsOf: labelsURL, encoding: .utf8)
            .split(whereSeparator: \.isNewline).map(String.init)
        guard loadedLabels.count == 1000 else { throw ClassificationError.missingResource("1,000 class labels") }
        let env = try ORTEnv(loggingLevel: .warning)
        let options = try ORTSessionOptions()
        try options.setIntraOpNumThreads(2)
        try options.setGraphOptimizationLevel(.all)
        let loadedSession = try ORTSession(env: env, modelPath: modelURL.path, sessionOptions: options)
        let inputs = try loadedSession.inputNames()
        let outputs = try loadedSession.outputNames()
        // Names are queried from ORT; expected shape/type was inspected from the bundled graph.
        guard inputs.count == 1, outputs.count == 1 else { throw ClassificationError.incompatibleModel }
        environment = env
        session = loadedSession
        inputName = inputs[0]; outputName = outputs[0]; labels = loadedLabels
    }

    func classify(imageData: Data) async throws -> [ClassificationResult] {
        try Task.checkCancellation()
        do {
            try initialize()
            let floats = try preprocessor.prepare(imageData: imageData)
            return try infer(floats)
        } catch is CancellationError {
            throw CancellationError()
        } catch let error as ClassificationError {
            throw error
        } catch {
            throw RuntimeFailure(detail: error.localizedDescription)
        }
    }

    private func infer(_ floats: [Float]) throws -> [ClassificationResult] {
        guard floats.count == ImagePreprocessor.shape.reduce(1, *) else { throw ClassificationError.invalidTensor }
        guard let session else { throw ClassificationError.incompatibleModel }
        try Task.checkCancellation()
        // ORT borrows this buffer. Keep it alive through run and output extraction.
        let buffer = floats.withUnsafeBytes { NSMutableData(bytes: $0.baseAddress!, length: $0.count) }
        let input = try ORTValue(tensorData: buffer, elementType: .float,
                                 shape: ImagePreprocessor.shape.map { NSNumber(value: $0) })
        let values = try session.run(withInputs: [inputName: input], outputNames: Set([outputName]), runOptions: nil)
        try Task.checkCancellation()
        guard let output = values[outputName] else { throw ClassificationError.invalidOutput }
        let info = try output.tensorTypeAndShapeInfo()
        guard info.elementType == .float, info.shape.map(\.intValue) == [1, 1000] else {
            throw ClassificationError.invalidOutput
        }
        let data = try output.tensorData() as Data
        guard data.count == 1000 * MemoryLayout<Float>.size else { throw ClassificationError.invalidOutput }
        let scores = data.withUnsafeBytes { bytes in
            (0..<1000).map { bytes.loadUnaligned(fromByteOffset: $0 * MemoryLayout<Float>.size, as: Float.self) }
        }
        return try outputProcessor.process(logits: scores, labels: labels)
    }
}

private struct RuntimeFailure: LocalizedError {
    let detail: String
    var errorDescription: String? { "The model could not analyze this photo. Please try again. (\(detail))" }
}
