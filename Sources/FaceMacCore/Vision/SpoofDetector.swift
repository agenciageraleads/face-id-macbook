import CoreGraphics
import CoreML
import CoreVideo
import Foundation

/// Presentation-attack check using Silent-Face-Anti-Spoofing (MiniFASNet,
/// Apache-2.0). Tells a real face from a photo or a screen on a single frame.
///
/// Input: 80x80 face crop, BGR, float 0-255. Class 1 is "real face".
public final class SpoofDetector {
    public struct Score {
        public let real: Float
    }

    private let wide: MLModel
    private let tight: MLModel

    public init?() {
        guard
            let wideURL = Bundle.main.url(forResource: "FASV2", withExtension: "mlmodelc"),
            let tightURL = Bundle.main.url(forResource: "FASV1SE", withExtension: "mlmodelc"),
            let wide = try? MLModel(contentsOf: wideURL),
            let tight = try? MLModel(contentsOf: tightURL)
        else { return nil }
        self.wide = wide
        self.tight = tight
    }

    public func score(pixelBuffer: CVPixelBuffer, face: DetectedFace) -> Score? {
        guard
            let wideReal = realProbability(wide, pixelBuffer, face.boundingBox, scale: 2.7),
            let tightReal = realProbability(tight, pixelBuffer, face.boundingBox, scale: 4.0)
        else { return nil }
        return Score(real: (wideReal + tightReal) / 2)
    }

    private func realProbability(_ model: MLModel, _ buffer: CVPixelBuffer, _ box: CGRect, scale: CGFloat) -> Float? {
        guard
            let input = Self.crop(buffer, box: box, scale: scale),
            let features = try? MLDictionaryFeatureProvider(dictionary: ["image": input]),
            let output = try? model.prediction(from: features),
            let probs = output.featureValue(for: "probs")?.multiArrayValue,
            probs.count == 3
        else { return nil }
        return probs[1].floatValue
    }

    /// Same crop as the original `generate_patches.py`, with bilinear resampling.
    static func crop(_ buffer: CVPixelBuffer, box: CGRect, scale: CGFloat) -> MLMultiArray? {
        guard CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32BGRA,
              let array = try? MLMultiArray(shape: [1, 3, 80, 80], dataType: .float32) else { return nil }
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }
        let pixels = base.assumingMemoryBound(to: UInt8.self)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        let srcW = CGFloat(CVPixelBufferGetWidth(buffer))
        let srcH = CGFloat(CVPixelBufferGetHeight(buffer))

        // Vision's origin is bottom-left; the pixel buffer's is top-left.
        let boxW = box.width * srcW
        let boxH = box.height * srcH
        guard boxW > 1, boxH > 1 else { return nil }
        let x = box.minX * srcW
        let y = (1 - box.maxY) * srcH
        let s = min((srcH - 1) / boxH, min((srcW - 1) / boxW, scale))
        let newW = boxW * s
        let newH = boxH * s
        var left = x + boxW / 2 - newW / 2
        var top = y + boxH / 2 - newH / 2
        var right = left + newW
        var bottom = top + newH
        if left < 0 { right -= left; left = 0 }
        if top < 0 { bottom -= top; top = 0 }
        if right > srcW - 1 { left -= right - srcW + 1; right = srcW - 1 }
        if bottom > srcH - 1 { top -= bottom - srcH + 1; bottom = srcH - 1 }
        let l = max(0, Int(left)), t = max(0, Int(top))
        let cropW = CGFloat(Int(right) - l + 1), cropH = CGFloat(Int(bottom) - t + 1)
        let maxX = Int(srcW) - 1, maxY = Int(srcH) - 1

        let out = array.dataPointer.assumingMemoryBound(to: Float32.self)
        for row in 0..<80 {
            let fy = (CGFloat(row) + 0.5) * cropH / 80 - 0.5 + CGFloat(t)
            let y0 = min(max(Int(floor(fy)), 0), maxY), y1 = min(y0 + 1, maxY)
            let wy = Float32(max(0, fy - CGFloat(y0)))
            for col in 0..<80 {
                let fx = (CGFloat(col) + 0.5) * cropW / 80 - 0.5 + CGFloat(l)
                let x0 = min(max(Int(floor(fx)), 0), maxX), x1 = min(x0 + 1, maxX)
                let wx = Float32(max(0, fx - CGFloat(x0)))
                for channel in 0..<3 { // BGRA already stores B, G, R in order.
                    let a = Float32(pixels[y0 * bytesPerRow + x0 * 4 + channel])
                    let b = Float32(pixels[y0 * bytesPerRow + x1 * 4 + channel])
                    let c = Float32(pixels[y1 * bytesPerRow + x0 * 4 + channel])
                    let d = Float32(pixels[y1 * bytesPerRow + x1 * 4 + channel])
                    let upper = a + (b - a) * wx
                    let lower = c + (d - c) * wx
                    out[channel * 6400 + row * 80 + col] = upper + (lower - upper) * wy
                }
            }
        }
        return array
    }
}
