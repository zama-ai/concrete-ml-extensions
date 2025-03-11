// Copyright © 2025 Zama. All rights reserved.

import SwiftUI

@main
struct TestConcretMLXApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

struct ContentView: View {
    // Male, 40, French/France, Automobiles => Top Ads: 160, 161, 188
    let french: [[UInt64]] = [[0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    // Female, 40, Italian/Italy, Food => Top Ads: 1279, 1280, 1281
    let italian: [[UInt64]] = [[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    var input: [[UInt64]] { french }
                
    @State var sizes: String = """
        PK:
        CK:
        input: 
        """

    @State var pk_: PrivateKey?
    @State var uid: String?
    @State var taskID: String?
    @State var status: String?
    @State var resultData: Data?
    @State var result: String?
    @State var ad1: Int?
    @State var ad2: Int?
    @State var ad3: Int?

    var body: some View {
        Text("Ad Targeting")
            .font(.title)

        ScrollView {
            LabeledContent("OneHot") {
                CopyableText("\(input)")
            }
            
            AsyncButton("Encrypt", action: encrypt)
            
            LabeledContent("Encryption") {
                Text(sizes)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            AsyncButton("Upload CK", action: uploadKey)
            LabeledContent("UID") {
                CopyableText(uid)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            
            AsyncButton("Start Task", action: startTask)
            LabeledContent("Task ID") {
                CopyableText(taskID)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            
            AsyncButton("Get Status", action: getStatus)
            LabeledContent("Status") {
                Text(status ?? "")
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            AsyncButton("Get Result", action: getResult)
            LabeledContent("Encrypted") {
                Text(resultData?.formattedSize ?? "")
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            
            AsyncButton("Decrypt", action: decrypt)
            LabeledContent("Clear") {
                let top = [ad1, ad2, ad3].compactMap { $0 }.map { String($0) }.joined(separator: ", ")
                // 25, 191, 489
                // 928 929 912
                
                VStack(alignment: .leading) {
                    CopyableText("Top Ads: \(top)")
                    CopyableText(result)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding()
        .buttonStyle(.borderedProminent)
        
        Spacer()
    }
    
    func cryptoParams() throws -> MatmulCryptoParameters {
        let jsonString = defaultParams()
        var asJSON = try JSONSerialization.jsonObject(with: jsonString.data(using: .utf8)!, options: []) as! [String: Any]
        asJSON["bits_reserved_for_computation"] = 8
        let asData = try JSONSerialization.data(withJSONObject: asJSON, options: [.prettyPrinted])
        let newParams = String(data: asData, encoding: .utf8)!
        
        return try matmulCryptoParametersDeserialize(content: newParams)
    }
    
    func encrypt() async throws {
        let cryptoParams = try cryptoParams()
        let keys = measure("Keygen") {
            cpuCreatePrivateKey(cryptoParams: cryptoParams) // 23 sec
        }
        
        let pk: PrivateKey = keys.privateKey()
        let ck: CpuCompressionKey = keys.cpuCompressionKey()
        self.pk_ = pk
        
        let pk_serialized: Data = try measure("Serialize PK") {
            try pk.serialize() // instant
        }
        
        let ck_serialized: Data = try measure("Serialize CK") {
            try ck.serialize() // 2.5 sec
        }
        
        let encrypted: EncryptedMatrix = try measure("Encrypt Data") {
            try encryptMatrix(pkey: pk,
                              cryptoParams: cryptoParams,
                              data: input) // instant
        }
        
        let input_serialized: Data = try measure("Serialize Data") {
            try encrypted.serialize() // instant
        }
        
        print(pk_serialized.formattedSize, // 33 KB
              ck_serialized.formattedSize, // 67 MB
              input_serialized.formattedSize) // 8 KB
        
        self.sizes = """
        PK: \(pk_serialized.formattedSize)
        CK: \(ck_serialized.formattedSize)
        input: \(input_serialized.formattedSize)
        """
        try await Storage.write(.matrixPrivateKey, data: pk_serialized)
        try await Storage.write(.matrixCPUCompressionKey, data: ck_serialized)
        try await Storage.write(.matrixEncryptedProfile, data: input_serialized)
    }
    
    func uploadKey() async throws {
        if let ck = await Storage.read(.matrixCPUCompressionKey) {
            try await measureAsync("Upload Server Key") {
                self.uid = try await Network.shared.uploadServerKey(ck, for: .ad_targeting)
            }
        }
    }

    func startTask() async throws {
        if let uid, let input = await Storage.read(.matrixEncryptedProfile) {
            self.taskID = try await Network.shared.startTask(.ad_targeting, uid: uid, encrypted_input: input)
        }
    }
    
    func getStatus() async throws {
        if let taskID, let uid {
            self.status = try await Network.shared.getStatus(for: .ad_targeting, id: taskID, uid: uid)
        }
    }
    
    func getResult() async throws {
        if let taskID, let uid {
            let data = try await Network.shared.getTaskResult(for: .ad_targeting, taskID: taskID, uid: uid)
            self.resultData = data
            try await Storage.write(.matrixEncryptedResult, data: data)
        }
    }
    
    func decrypt() async throws {
        guard let pk_, let resultData else {
            return
        }
        
        let compressedMatrix: CompressedResultEncryptedMatrix = try measure("Deserialize Matrix") {
            try compressedResultEncryptedMatrixDeserialize(content: resultData)
        }
        
        let rawResult: [[UInt64]] = try measure("Decrypt Matrix") {
            try decryptMatrix(compressedMatrix: compressedMatrix,
                              privateKey: pk_,
                              cryptoParams: cryptoParams(),
                              numValidGlweValuesInLastCiphertext: 42) // What is this ?
        }
        
        let clearResult: [Int64] = rawResult[0].compactMap {
            let raw = Int64(truncatingIfNeeded: $0)
            return raw <= 0 ? 0 : raw
        }

        print(clearResult)
        result = "\(clearResult)"
        ad1 = nthHighestScore(rank: 0, in: clearResult)
        ad2 = nthHighestScore(rank: 1, in: clearResult)
        ad3 = nthHighestScore(rank: 2, in: clearResult)
    }
}

/// Returns the index of the `rank`th highest score in the given list.
///
/// - Parameters:
///   - rank: The position (0-based) of the desired item, where `0` is the highest.
///   - scores: A list of numerical scores.
/// - Returns: The index of the `rank`th highest score.
///
/// - Example:
///   ```swift
///   let scores: [UInt64] = [10, 50, 12, 32]
///   nthHighestScore(rank: 0, in: scores) // → 1 (highest score: 50)
///   nthHighestScore(rank: 1, in: scores) // → 3 (second highest: 32)
///   nthHighestScore(rank: 2, in: scores) // → 2 (third highest: 12)
///   nthHighestScore(rank: 3, in: scores) // → 0 (fourth highest: 10)
///   ```
func nthHighestScore(rank: Int, in scores: [Int64]) -> Int {
    let positions = scores.enumerated()
        .sorted { $0.element > $1.element } // highest scores first
        .map { $0.offset }
    return positions[rank]
}

