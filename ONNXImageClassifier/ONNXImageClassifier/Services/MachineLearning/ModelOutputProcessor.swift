import Foundation

struct ModelOutputProcessor: Sendable {
    func process(logits: [Float], labels: [String]) throws -> [ClassificationResult] {
        guard logits.count == 1000, labels.count == logits.count,
              logits.allSatisfy(\.isFinite), let maximum = logits.max() else {
            throw ClassificationError.invalidOutput
        }
        let exponentials = logits.map { exp($0 - maximum) }
        let sum = exponentials.reduce(0, +)
        guard sum.isFinite, sum > 0 else { throw ClassificationError.invalidOutput }
        return logits.indices.sorted {
            logits[$0] == logits[$1] ? $0 < $1 : logits[$0] > logits[$1]
        }.prefix(3).map {
            ClassificationResult(classIndex: $0, label: labels[$0], confidence: exponentials[$0] / sum)
        }
    }
}
