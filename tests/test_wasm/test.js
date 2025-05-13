import initWasmModule, {
        keygen_radix_u64_wasm,
        encrypt_serialize_u64_radix_flat_wasm,
        decrypt_serialized_u64_radix_flat_wasm
    } from '../../rust/pkg-wasm/concrete_ml_extensions_wasm.js';
    
    let wasmModule = null;
    
    const runDemoButton = document.getElementById('runDemoButton');
    const loader = document.getElementById('loader');
    const resultsArea = document.getElementById('resultsArea');
    const errorArea = document.getElementById('errorArea');
    
    // Helper to update HTML elements
    function updateElement(id, value) {
        const el = document.getElementById(id);
        if (el) {
            el.textContent = value;
        } else {
            console.warn(`Element with ID ${id} not found.`);
        }
    }
    
    function showLoader(show) {
        loader.style.display = show ? 'block' : 'none';
    }
    
    function displayError(message) {
        resultsArea.style.display = 'none';
        errorArea.textContent = `Error: ${message}`;
        errorArea.style.display = 'block';
        console.error(message);
    }
    
    function clearResults() {
        const N_A = '-';
        updateElement('timeInit', N_A);
        updateElement('timeKeygen', N_A);
        updateElement('timeEncrypt', N_A);
        updateElement('timeDecrypt', N_A);
        updateElement('sizeKey', N_A);
        updateElement('sizeServerKey', N_A); // Clear server key size
        updateElement('sizeOriginal', N_A);
        updateElement('sizeEncrypted', N_A);
        updateElement('expansionFactor', N_A);
        updateElement('dataOriginal', N_A);
        updateElement('dataEncrypted', N_A);
        updateElement('dataDecrypted', N_A);
        updateElement('verificationStatus', N_A);
        const verificationEl = document.getElementById('verificationStatus');
        verificationEl.className = 'status'; // Reset class
        resultsArea.style.display = 'none';
        errorArea.style.display = 'none';
    }
    
    async function initializeWasm() {
        try {
            console.log('Starting WASM initialization...');
            if (wasmModule) {
                console.log('WASM module already initialized');
                return true;
            }
            console.log('Calling initWasmModule()...');
            wasmModule = await initWasmModule();
            console.log('initWasmModule() completed');
            if (!wasmModule) {
                throw new Error('WASM module initialization returned null');
            }
            console.log('Verifying WASM module...');
            if (typeof wasmModule.keygen_radix_u64_wasm !== 'function') {
                throw new Error('WASM module functions not properly initialized');
            }
            console.log('WASM module initialized successfully');
            return true;
        } catch (error) {
            console.error('WASM initialization error:', error);
            displayError(`Failed to initialize WASM module: ${error.message}`);
            return false;
        }
    }
    
    async function runDemo() {
        console.log('Demo started');
        runDemoButton.disabled = true;
        showLoader(true);
        clearResults();
    
        try {
            let startTime, endTime;
    
            if (!wasmModule) {
                startTime = performance.now();
                const initialized = await initializeWasm();
                if (!initialized) {
                    throw new Error('WASM module initialization failed');
                }
                endTime = performance.now();
                updateElement('timeInit', (endTime - startTime).toFixed(2));
            }
    
            resultsArea.style.display = 'block';
    
            console.log('Starting key generation...');
            startTime = performance.now();
            try {
                // keygen_radix_u64_wasm now returns a JsValue which is a JS object
                const keyPairResult = keygen_radix_u64_wasm();
                if (!keyPairResult || typeof keyPairResult !== 'object') {
                    throw new Error('Key generation did not return a valid object.');
                }
    
                const clientKeyBytes = keyPairResult.clientKey;
                const serverKeyBytes = keyPairResult.serverKey; // Get the server key bytes
    
                if (!clientKeyBytes || !(clientKeyBytes instanceof Uint8Array)) {
                    throw new Error('ClientKey not found or is not a Uint8Array.');
                }
                if (!serverKeyBytes || !(serverKeyBytes instanceof Uint8Array)) {
                    throw new Error('ServerKey not found or is not a Uint8Array.');
                }
    
                endTime = performance.now();
                console.log('Key generation completed');
                updateElement('timeKeygen', (endTime - startTime).toFixed(2));
                updateElement('sizeKey', clientKeyBytes.length);
                updateElement('sizeServerKey', serverKeyBytes.length); // Display server key size
    
                const rows = 2;
                const cols = 2;
                const originalDataFlat = new BigUint64Array(rows * cols);
                originalDataFlat[0] = BigInt(Math.floor(Math.random() * Number.MAX_SAFE_INTEGER));
                originalDataFlat[1] = BigInt(Math.floor(Math.random() * Number.MAX_SAFE_INTEGER));
                originalDataFlat[2] = BigInt(Math.floor(Math.random() * Number.MAX_SAFE_INTEGER));
                originalDataFlat[3] = BigInt(Math.floor(Math.random() * Number.MAX_SAFE_INTEGER));
                const originalDataSnapshot = Array.from(originalDataFlat.slice(0, 4)).map(String);
                updateElement('dataOriginal', `[${originalDataSnapshot.join(', ')}]`);
                updateElement('sizeOriginal', originalDataFlat.byteLength);
    
                startTime = performance.now();
                const encryptedDataBytes = encrypt_serialize_u64_radix_flat_wasm(
                    originalDataFlat,
                    clientKeyBytes
                );
                if (!encryptedDataBytes) {
                    throw new Error('Encryption failed');
                }
                endTime = performance.now();
                updateElement('timeEncrypt', (endTime - startTime).toFixed(2));
                updateElement('sizeEncrypted', encryptedDataBytes.length);
    
                const encryptedSnapshot = Array.from(encryptedDataBytes.slice(0, 16))
                    .map(byte => byte.toString(16).padStart(2, '0').toUpperCase())
                    .join(' ');
                updateElement('dataEncrypted', `0x${encryptedSnapshot}...`);
                
                const expansionFactor = (encryptedDataBytes.length / originalDataFlat.byteLength).toFixed(2);
                updateElement('expansionFactor', expansionFactor);
    
                startTime = performance.now();
                const decryptedDataFlat = decrypt_serialized_u64_radix_flat_wasm(
                    encryptedDataBytes,
                    clientKeyBytes
                );
                if (!decryptedDataFlat) {
                    throw new Error('Decryption failed');
                }
                endTime = performance.now();
                updateElement('timeDecrypt', (endTime - startTime).toFixed(2));
                
                const decryptedSnapshot = Array.from(decryptedDataFlat.slice(0, 4)).map(String);
                updateElement('dataDecrypted', `[${decryptedSnapshot.join(', ')}]`);
    
                let match = true;
                if (decryptedDataFlat.length !== originalDataFlat.length) {
                    match = false;
                } else {
                    for (let i = 0; i < originalDataFlat.length; i++) {
                        if (originalDataFlat[i] !== decryptedDataFlat[i]) {
                            match = false;
                            break;
                        }
                    }
                }
                const verificationEl = document.getElementById('verificationStatus');
                if (match) {
                    verificationEl.textContent = "SUCCESS: Decrypted data matches original.";
                    verificationEl.className = 'status success';
                } else {
                    verificationEl.textContent = "FAILURE: Decrypted data does NOT match original.";
                    verificationEl.className = 'status error';
                }
            } catch (error) {
                console.error('Operation error:', error);
                throw error;
            }
    
        } catch (error) {
            console.error("Demo Error:", error);
            console.error("Error stack:", error.stack);
            displayError(error.message || "An unknown error occurred.");
        } finally {
            showLoader(false);
            runDemoButton.disabled = false;
            console.log('Demo finished');
        }
    }
    
    window.addEventListener('load', async () => {
        console.log('Page loaded, initializing WASM...');
        await initializeWasm();
    });
    
    runDemoButton.addEventListener('click', () => {
        console.log('Button clicked');
        runDemo();
    });