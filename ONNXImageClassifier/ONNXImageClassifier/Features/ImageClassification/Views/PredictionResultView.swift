import SwiftUI

struct PredictionResultView: View {
    let result: ClassificationResult
    let rank: Int
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                Text(String(format: "%02d", rank)).font(.caption.monospaced()).foregroundStyle(.secondary)
                Text(result.label.components(separatedBy: ",").first ?? result.label)
                    .font(.headline).textSelection(.enabled)
                Spacer(minLength: 8)
                Text(Double(result.confidence), format: .percent.precision(.fractionLength(1)))
                    .font(.headline.monospacedDigit()).foregroundStyle(rank == 1 ? Color.teal : Color.secondary)
            }
            ProgressView(value: Double(result.confidence)).tint(rank == 1 ? .teal : .secondary)
                .accessibilityLabel("Confidence")
        }
        .padding(16)
        .background(.background, in: RoundedRectangle(cornerRadius: 18))
        .accessibilityElement(children: .combine)
    }
}
