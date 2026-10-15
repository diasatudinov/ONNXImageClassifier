import SwiftUI
import PhotosUI

protocol ImageClassifying: Sendable {
    func classify(imageData: Data) async throws -> [ClassificationResult]
}

@MainActor @Observable
final class ImageClassifierViewModel {
    private(set) var imageData: Data?
    private(set) var results: [ClassificationResult] = []
    private(set) var isLoading = false
    private(set) var isAnalyzing = false
    private(set) var errorMessage: String?
    private(set) var elapsedTime: Double?
    private let classifier: any ImageClassifying
    private var task: Task<Void, Never>?
    private var generation = UUID()
    var isBusy: Bool { isLoading || isAnalyzing }

    init(classifier: any ImageClassifying = ONNXModelService()) {
        self.classifier = classifier
    }

    func select(_ item: PhotosPickerItem?) {
        reset()
        guard let item else { return }
        isLoading = true
        let token = generation
        task = Task {
            defer { if generation == token { isLoading = false } }
            do {
                guard let data = try await item.loadTransferable(type: Data.self) else {
                    throw ClassificationError.invalidImage
                }
                try Task.checkCancellation()
                guard generation == token else { return }
                imageData = data
            } catch is CancellationError {
            } catch {
                if generation == token { errorMessage = error.localizedDescription }
            }
        }
    }

    func analyze() {
        guard let imageData, !isBusy else { return }
        results = []; errorMessage = nil; elapsedTime = nil; isAnalyzing = true
        let token = generation
        let classifier = classifier
        task = Task {
            defer { if generation == token { isAnalyzing = false } }
            do {
                let start = ContinuousClock.now
                let predictions = try await classifier.classify(imageData: imageData)
                try Task.checkCancellation()
                guard generation == token else { return }
                results = predictions
                let duration = start.duration(to: .now)
                elapsedTime = Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
            } catch is CancellationError {
            } catch {
                if generation == token { errorMessage = error.localizedDescription }
            }
        }
    }

    func reset() {
        task?.cancel(); task = nil; generation = UUID()
        imageData = nil; results = []; errorMessage = nil; elapsedTime = nil
        isLoading = false; isAnalyzing = false
    }
}
