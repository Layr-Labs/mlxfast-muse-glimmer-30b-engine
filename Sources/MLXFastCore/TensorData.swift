import Foundation

public enum TensorDType: String, Codable, Equatable {
    case bool = "BOOL"
    case u8 = "U8"
    case i8 = "I8"
    case i16 = "I16"
    case u16 = "U16"
    case i32 = "I32"
    case u32 = "U32"
    case i64 = "I64"
    case u64 = "U64"
    case f16 = "F16"
    case bf16 = "BF16"
    case f32 = "F32"
    case f64 = "F64"
    /// OCP FP8 E4M3 ("FN": no infinities, NaN only at S.1111.111). The
    /// pinned Muse Glimmer checkpoint stores its calibrated float8 (W8A8)
    /// projection weights in this format.
    case f8e4m3 = "F8_E4M3"

    public var byteWidth: Int {
        switch self {
        case .bool, .u8, .i8, .f8e4m3:
            return 1
        case .i16, .u16, .f16, .bf16:
            return 2
        case .i32, .u32, .f32:
            return 4
        case .i64, .u64, .f64:
            return 8
        }
    }

    public static func parse(_ raw: String) throws -> TensorDType {
        guard let dtype = TensorDType(rawValue: raw) else {
            throw MLXFastError.invalidInput("unsupported safetensors dtype \(raw)")
        }
        return dtype
    }
}

public struct MaterializedTensor: Equatable {
    public let name: String
    public let dtype: TensorDType
    public let shape: [Int]
    public let bytes: Data

    public init(name: String, dtype: TensorDType, shape: [Int], bytes: Data) throws {
        let expectedByteCount = try expectedTensorByteCount(name: name, dtype: dtype, shape: shape)
        guard expectedByteCount == bytes.count else {
            throw MLXFastError.invalidInput(
                "tensor \(name) byte count \(bytes.count) does not match dtype \(dtype.rawValue) and shape \(shape) expected \(expectedByteCount)"
            )
        }
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.bytes = bytes
    }

    public var elementCount: Int {
        shape.reduce(1, *)
    }

    public func uint8Values() throws -> [UInt8] {
        guard dtype == .u8 || dtype == .bool else {
            throw MLXFastError.invalidInput("tensor \(name) has dtype \(dtype.rawValue), not U8/BOOL")
        }
        return Array(bytes)
    }

    public func uint32Values() throws -> [UInt32] {
        guard dtype == .u32 else {
            throw MLXFastError.invalidInput("tensor \(name) has dtype \(dtype.rawValue), not U32")
        }
        return bytes.withUnsafeBytes { raw in
            (0..<elementCount).map { index in
                raw.loadUnaligned(fromByteOffset: index * 4, as: UInt32.self).littleEndian
            }
        }
    }

    public func int32Values() throws -> [Int32] {
        guard dtype == .i32 else {
            throw MLXFastError.invalidInput("tensor \(name) has dtype \(dtype.rawValue), not I32")
        }
        return bytes.withUnsafeBytes { raw in
            (0..<elementCount).map { index in
                raw.loadUnaligned(fromByteOffset: index * 4, as: Int32.self).littleEndian
            }
        }
    }

    /// Decodes F8_E4M3 elements exactly: exponent bias 7, three mantissa
    /// bits, subnormals at exponent 0, finite range +-448, and NaN for the
    /// one all-ones exponent-and-mantissa pattern of either sign.
    public func float8E4M3Values() throws -> [Float] {
        guard dtype == .f8e4m3 else {
            throw MLXFastError.invalidInput("tensor \(name) has dtype \(dtype.rawValue), not F8_E4M3")
        }
        return bytes.map(decodeFloat8E4M3)
    }

    public func float32Values() throws -> [Float] {
        guard dtype == .f32 else {
            throw MLXFastError.invalidInput("tensor \(name) has dtype \(dtype.rawValue), not F32")
        }
        return bytes.withUnsafeBytes { raw in
            (0..<elementCount).map { index in
                let bits = raw.loadUnaligned(fromByteOffset: index * 4, as: UInt32.self).littleEndian
                return Float(bitPattern: bits)
            }
        }
    }
}

/// One OCP FP8 E4M3 ("FN") byte as a Float. Every finite value is exact in
/// Float, so the decode loses nothing.
public func decodeFloat8E4M3(_ byte: UInt8) -> Float {
    let negative = byte & 0x80 != 0
    let exponent = Int((byte >> 3) & 0x0F)
    let mantissa = Float(byte & 0x07)
    let magnitude: Float
    if exponent == 0x0F, byte & 0x07 == 0x07 {
        return .nan
    } else if exponent == 0 {
        magnitude = mantissa / 8 * 0x1p-6
    } else {
        magnitude = (1 + mantissa / 8) * Float(sign: .plus, exponent: exponent - 7, significand: 1)
    }
    return negative ? -magnitude : magnitude
}

public func expectedTensorByteCount(name: String, dtype: TensorDType, shape: [Int]) throws -> Int {
    guard shape.allSatisfy({ $0 >= 0 }) else {
        throw MLXFastError.invalidInput("tensor \(name) has negative dimension in shape \(shape)")
    }
    var elements = 1
    for dimension in shape {
        let result = elements.multipliedReportingOverflow(by: dimension)
        guard !result.overflow else {
            throw MLXFastError.invalidInput("tensor \(name) shape \(shape) element count overflows Int")
        }
        elements = result.partialValue
    }
    let bytes = elements.multipliedReportingOverflow(by: dtype.byteWidth)
    guard !bytes.overflow else {
        throw MLXFastError.invalidInput("tensor \(name) byte count overflows Int for dtype \(dtype.rawValue) and shape \(shape)")
    }
    return bytes.partialValue
}

public func materializeTensor(
    name: String,
    dtype rawDType: String,
    shape: [Int],
    bytes: Data
) throws -> MaterializedTensor {
    try MaterializedTensor(
        name: name,
        dtype: TensorDType.parse(rawDType),
        shape: shape,
        bytes: bytes
    )
}
