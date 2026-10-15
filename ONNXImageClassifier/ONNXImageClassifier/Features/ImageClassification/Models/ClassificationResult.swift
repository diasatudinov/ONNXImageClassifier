import Foundation

struct ClassificationResult: Identifiable, Sendable, Equatable {
    let classIndex: Int
    let label: String
    let confidence: Float
    var id: Int { classIndex }
}

enum ClassificationError: LocalizedError {
    case missingResource(String), invalidImage, invalidTensor, invalidOutput, incompatibleModel
    var errorDescription: String? {
        switch self {
        case .missingResource(let name): "The bundled \(name) is missing. Please reinstall the app."
        case .invalidImage: "This image could not be read. Please choose another photo."
        case .invalidTensor: "The image could not be converted into the model’s required input."
        case .invalidOutput: "The model returned an invalid classification."
        case .incompatibleModel: "The bundled model is incompatible with this version of the app."
        }
    }
}
