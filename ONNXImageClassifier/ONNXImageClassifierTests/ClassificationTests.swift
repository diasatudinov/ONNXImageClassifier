import XCTest
import UIKit
import ImageIO
import UniformTypeIdentifiers
@testable import ONNXImageClassifier

final class ClassificationTests: XCTestCase {
    func testShapeAndRGBNormalization() throws {
        let data = solidImage(color: .red)
        let tensor = try ImagePreprocessor().prepare(imageData: data)
        XCTAssertEqual(ImagePreprocessor.shape, [1, 3, 224, 224])
        XCTAssertEqual(tensor.count, 150528)
        XCTAssertEqual(tensor[0], (1 - 0.485) / 0.229, accuracy: 0.02)
        XCTAssertEqual(tensor[224 * 224], -0.456 / 0.224, accuracy: 0.02)
        XCTAssertEqual(tensor[2 * 224 * 224], -0.406 / 0.225, accuracy: 0.02)
    }

    func testEXIFOrientationAndPixelOrder() throws {
        let format = UIGraphicsImageRendererFormat(); format.scale = 1
        let image = UIGraphicsImageRenderer(size: CGSize(width: 256, height: 256), format: format).image { context in
            UIColor.red.setFill(); context.fill(CGRect(x: 0, y: 0, width: 256, height: 128))
            UIColor.blue.setFill(); context.fill(CGRect(x: 0, y: 128, width: 256, height: 128))
        }
        let buffer = NSMutableData()
        let destination = try XCTUnwrap(CGImageDestinationCreateWithData(buffer, UTType.jpeg.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, try XCTUnwrap(image.cgImage), [kCGImagePropertyOrientation: 6] as CFDictionary)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
        let tensor = try ImagePreprocessor().prepare(imageData: buffer as Data)
        // EXIF 6 rotates the red top half clockwise onto the right half.
        let left = 112 * 224 + 20, right = 112 * 224 + 200
        XCTAssertLessThan(tensor[left], -1.9)
        XCTAssertGreaterThan(tensor[right], 2.0)
        XCTAssertGreaterThan(tensor[2 * 224 * 224 + left], 2.4)
    }

    func testCenterCropExcludesOuterBorder() throws {
        let format = UIGraphicsImageRendererFormat(); format.scale = 1
        let data = UIGraphicsImageRenderer(size: CGSize(width: 512, height: 256), format: format).pngData { context in
            UIColor.blue.setFill(); context.fill(CGRect(x: 0, y: 0, width: 512, height: 256))
            UIColor.red.setFill(); context.fill(CGRect(x: 144, y: 0, width: 224, height: 256))
        }
        let tensor = try ImagePreprocessor().prepare(imageData: data)
        XCTAssertTrue(tensor.prefix(224 * 224).allSatisfy { abs($0 - (1 - 0.485) / 0.229) < 0.02 })
    }

    func testInvalidImageRejected() {
        XCTAssertThrowsError(try ImagePreprocessor().prepare(imageData: Data([0, 1, 2])))
    }

    func testStableSoftmaxAndSorting() throws {
        var logits = [Float](repeating: -1000, count: 1000)
        logits[7] = 1002; logits[2] = 1001; logits[99] = 1000
        let results = try ModelOutputProcessor().process(logits: logits, labels: labels)
        XCTAssertEqual(results.map(\.classIndex), [7, 2, 99])
        XCTAssertEqual(results[0].confidence, 0.66524094, accuracy: 0.00001)
        XCTAssertEqual(results.reduce(0) { $0 + $1.confidence }, 1, accuracy: 0.00001)
    }

    func testInvalidOutputsRejected() {
        XCTAssertThrowsError(try ModelOutputProcessor().process(logits: [], labels: labels))
        XCTAssertThrowsError(try ModelOutputProcessor().process(logits: Array(repeating: .nan, count: 1000), labels: labels))
        XCTAssertThrowsError(try ModelOutputProcessor().process(logits: Array(repeating: 0, count: 1000), labels: []))
    }

    func testTiesAreDeterministic() throws {
        let results = try ModelOutputProcessor().process(logits: Array(repeating: 0, count: 1000), labels: labels)
        XCTAssertEqual(results.map(\.classIndex), [0, 1, 2])
        XCTAssertEqual(results[0].confidence, 0.001, accuracy: 0.000001)
    }

    func testModelInitializationAndBundledResources() async throws {
        let service = ONNXModelService()
        try await service.initialize()
    }

    func testRealSamoyedInference() async throws {
        let url = try XCTUnwrap(Bundle(for: Self.self).url(forResource: "dog", withExtension: "jpg"))
        let data = try Data(contentsOf: url)
        let service = ONNXModelService()
        let results = try await service.classify(imageData: data)
        XCTAssertEqual(results.count, 3)
        XCTAssertEqual(results.first?.classIndex, 258)
        XCTAssertTrue(results.first?.label.contains("Samoyed") == true)
        XCTAssertGreaterThan(results[0].confidence, 0.5)
    }

    func testMissingModel() async {
        let service = ONNXModelService(bundle: Bundle(for: Self.self))
        do {
            try await service.initialize()
            XCTFail("Expected missing model error")
        } catch {
            XCTAssertTrue(error is ClassificationError)
        }
    }

    private var labels: [String] { (0..<1000).map { "Class \($0)" } }

    private func solidImage(color: UIColor) -> Data {
        let format = UIGraphicsImageRendererFormat(); format.scale = 1
        return UIGraphicsImageRenderer(size: CGSize(width: 320, height: 240), format: format).pngData { context in
            color.setFill(); context.fill(CGRect(x: 0, y: 0, width: 320, height: 240))
        }
    }
}
