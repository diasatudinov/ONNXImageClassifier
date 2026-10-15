import SwiftUI
import PhotosUI
import ImageIO

struct ImageClassifierView: View {
    @State private var model = ImageClassifierViewModel()
    @State private var selection: PhotosPickerItem?

    var body: some View {
        let photoButtonTitle = model.imageData == nil ? "Choose photo" : "Change photo"
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    VStack(alignment: .leading, spacing: 8) {
                        Label("ON-DEVICE INTELLIGENCE", systemImage: "sparkles")
                            .font(.caption.weight(.semibold)).tracking(1.5).foregroundStyle(.teal)
                        Text("A closer look.").font(.largeTitle.bold())
                        Text("Discover what’s in your photo with MobileNetV2. Your images stay on this device.")
                            .foregroundStyle(.secondary)
                    }
                    preview
                    HStack(spacing: 12) {
                        PhotosPicker(selection: $selection, matching: .images, photoLibrary: .shared()) {
                            Label(photoButtonTitle, systemImage: "photo.on.rectangle")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered).disabled(model.isBusy)
                        Button(action: model.analyze) {
                            HStack {
                                if model.isAnalyzing { ProgressView().tint(.white) }
                                Text(model.isAnalyzing ? "Analyzing…" : "Analyze")
                            }.frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.borderedProminent).tint(.teal)
                        .disabled(model.imageData == nil || model.isBusy)
                    }
                    .controlSize(.large)
                    if let error = model.errorMessage {
                        Label(error, systemImage: "exclamationmark.triangle")
                            .font(.callout).foregroundStyle(.red)
                            .accessibilityLabel("Error: \(error)")
                    }
                    if !model.results.isEmpty {
                        VStack(alignment: .leading, spacing: 12) {
                            HStack {
                                Text("Top matches").font(.title2.bold())
                                Spacer()
                                if let elapsed = model.elapsedTime {
                                    Text(String(format: "%.2f s", elapsed)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                                }
                            }
                            ForEach(Array(model.results.enumerated()), id: \.element.id) { index, result in
                                PredictionResultView(result: result, rank: index + 1)
                            }
                            Text("Model confidence is an estimate across 1,000 ImageNet classes, and can be wrong.")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    Label("Powered by ONNX Runtime · Works offline", systemImage: "lock.shield")
                        .font(.caption).foregroundStyle(.secondary).frame(maxWidth: .infinity)
                }.padding(24).frame(maxWidth: 680)
                    .frame(maxWidth: .infinity)
            }
            .background(Color(uiColor: .secondarySystemGroupedBackground))
            .navigationTitle("Image Lens").navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Reset") { selection = nil; model.reset() }
                        .disabled(model.imageData == nil && !model.isBusy && model.errorMessage == nil)
                }
            }
            .onChange(of: selection) { _, item in model.select(item) }
        }
    }

    private var preview: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 24).fill(.teal.opacity(0.08))
            if let data = model.imageData {
                ImagePreview(data: data)
            } else {
                VStack(spacing: 16) {
                    if model.isLoading {
                        ProgressView("Loading photo…")
                    } else {
                        Image(systemName: "viewfinder").font(.system(size: 52, weight: .light)).foregroundStyle(.teal)
                        Text("Start with a photo").font(.headline)
                        Text("Animals, everyday objects, and more")
                            .font(.subheadline).foregroundStyle(.secondary)
                    }
                }.padding()
            }
        }
        .frame(height: 300).clipShape(RoundedRectangle(cornerRadius: 24))
    }
}

private struct ImagePreview: View {
    let data: Data
    @State private var preview: UIImage?
    var body: some View {
        Group {
            if let preview {
                Image(uiImage: preview).resizable().scaledToFit().accessibilityLabel("Selected photo")
            } else { ProgressView() }
        }
        .task(id: data) {
            let decoded = await Task.detached(priority: .userInitiated) {
                guard let source = CGImageSourceCreateWithData(data as CFData, nil),
                      let image = CGImageSourceCreateThumbnailAtIndex(source, 0, [
                        kCGImageSourceCreateThumbnailFromImageAlways: true,
                        kCGImageSourceCreateThumbnailWithTransform: true,
                        kCGImageSourceThumbnailMaxPixelSize: 1000
                      ] as CFDictionary) else { return Optional<UIImage>.none }
                return UIImage(cgImage: image)
            }.value
            if !Task.isCancelled { preview = decoded }
        }
    }
}
