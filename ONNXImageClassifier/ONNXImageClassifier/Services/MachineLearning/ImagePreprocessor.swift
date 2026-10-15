import CoreImage
import ImageIO
import Foundation

struct ImagePreprocessor: Sendable {
    static let shape: [Int] = [1, 3, 224, 224]
    static let mean: [Float] = [0.485, 0.456, 0.406]
    static let standardDeviation: [Float] = [0.229, 0.224, 0.225]

    func prepare(imageData: Data) throws -> [Float] {
        try Task.checkCancellation()
        // ImageIO applies EXIF orientation and bounds decoding memory for large photos.
        guard let source = CGImageSourceCreateWithData(imageData as CFData, nil),
              let thumbnail = CGImageSourceCreateThumbnailAtIndex(source, 0, [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceCreateThumbnailWithTransform: true,
                kCGImageSourceThumbnailMaxPixelSize: 2048,
                kCGImageSourceShouldCacheImmediately: true
              ] as CFDictionary) else { throw ClassificationError.invalidImage }
        let width = CGFloat(thumbnail.width), height = CGFloat(thumbnail.height)
        let scale = 256 / min(width, height)
        let resizedWidth = (width * scale).rounded(), resizedHeight = (height * scale).rounded()
        let image = CIImage(cgImage: thumbnail).transformed(by: CGAffineTransform(
            scaleX: resizedWidth / width, y: resizedHeight / height))
        let crop = CGRect(x: floor(resizedWidth / 2) - 112, y: floor(resizedHeight / 2) - 112,
                          width: 224, height: 224)
        let context = CIContext(options: [.cacheIntermediates: false])
        guard let cropped = context.createCGImage(image, from: crop),
              let rgb = CGColorSpace(name: CGColorSpace.sRGB) else { throw ClassificationError.invalidImage }
        var pixels = [UInt8](repeating: 0, count: 224 * 224 * 4)
        let rendered = pixels.withUnsafeMutableBytes { bytes -> Bool in
            guard let drawing = CGContext(data: bytes.baseAddress, width: 224, height: 224,
                bitsPerComponent: 8, bytesPerRow: 224 * 4, space: rgb,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue) else { return false }
            drawing.draw(cropped, in: CGRect(x: 0, y: 0, width: 224, height: 224))
            return true
        }
        guard rendered else { throw ClassificationError.invalidImage }
        var tensor = [Float](repeating: 0, count: 3 * 224 * 224)
        for channel in 0..<3 {
            try Task.checkCancellation()
            for pixel in 0..<(224 * 224) {
                tensor[channel * 224 * 224 + pixel] =
                    (Float(pixels[pixel * 4 + channel]) / 255 - Self.mean[channel]) / Self.standardDeviation[channel]
            }
        }
        return tensor
    }
}
