//
//  ContentView.swift
//  TestConcretMLX
//
//  Created by Dimitri Dupuis-Latour on 17/01/2025.
//

import SwiftUI

struct ContentView: View {
    var body: some View {
        VStack {
            Image(systemName: "globe")
                .imageScale(.large)
                .foregroundStyle(.tint)
            Text("Hello, world!")
            
            Button("Tap me !") {
                buttonTapped()
            }
        }
        .padding()
    }
    
    func buttonTapped() {
        sayHello()
        
        do {
            let jsonParams = defaultParams()

            let cryptoParams: MatmulCryptoParameters = try matmulCryptoParametersDeserialize(content: jsonParams)
            
            let keys = measure("Keygen") {
                cpuCreatePrivateKey(cryptoParams: cryptoParams) // 23 sec
            }
            
            let pk: PrivateKey = keys.privateKey()
            let ck: CpuCompressionKey = keys.cpuCompressionKey()
            
            let pk_serialized: Data = try measure("Serialize PK") {
                try pk.serialize() // instant
            }
            
            let ck_serialized: Data = try measure("Serialize CK") {
                try ck.serialize() // 2.5 sec
            }
            
            let input: [[UInt64]] = [[0, 0, 1]]
            let encrypted: EncryptedMatrix = try measure("Encrypt Data") {
                try encryptMatrix(pkey: pk,
                                  cryptoParams: cryptoParams,
                                  data: input) // instant
            }
            
            let serialized: Data = try measure("Serialize Data") {
                try encrypted.serialize() // instant
            }
            
            print(pk_serialized.formattedSize, // 33 KB
                  ck_serialized.formattedSize, // 67 MB
                  serialized.formattedSize) // 8 KB
            
            let serverData = Data() // TODO: needs to come from server…
            let compressedMatrix: CompressedResultEncryptedMatrix = try measure("Deserialize Matrix") {
                try compressedResultEncryptedMatrixDeserialize(content: serverData)
            }
            
            let clearResult: [[UInt64]] = try measure("Decrypt Matrix") {
                try decryptMatrix(compressedMatrix: compressedMatrix,
                                  privateKey: pk,
                                  cryptoParams: cryptoParams,
                                  numValidGlweValuesInLastCiphertext: 42) // What is this ?
            }
            
            print(clearResult)
            
        } catch {
            print(error)
        }
    }
}

func measure<T>(_ name: String, _ block: () -> T) -> T {
    print("\(name) in progress…")
    let start = Date()
    let result = block()
    let duration = Date().timeIntervalSince(start)
    print("\(name) done (\(duration) seconds).")
    
    return result
}

func measure<T>(_ name: String, _ block: () throws -> T) throws -> T {
    print("\(name) in progress…")
    let start = Date()
    do {
        let result = try block()
        let duration = Date().timeIntervalSince(start)
        print("\(name) done (\(duration) seconds).")
        return result
    } catch {
        let duration = Date().timeIntervalSince(start)
        print("\(name) exception (\(duration) seconds)")
        throw error
    }
}

extension Data {
    /// Returns a human-readable size format (e.g., "10.5 MB", "512 KB")
    var formattedSize: String {
        let byteCount = Double(self.count)
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useBytes, .useKB, .useMB, .useGB]
        formatter.countStyle = .file
        return formatter.string(fromByteCount: Int64(byteCount))
    }
}

#Preview {
    ContentView()
}
