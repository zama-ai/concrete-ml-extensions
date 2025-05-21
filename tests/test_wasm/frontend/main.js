import init, {
  keygen_radix_u64_wasm as keygen,
  encrypt_serialize_u64_radix_flat_wasm as enc,
  decrypt_serialized_u64_radix_flat_wasm as dec,
} from './pkg/concrete_ml_extensions_wasm.js';

const out = document.getElementById('out');
const log = (txt) => (out.textContent += `${txt}\n`);

let wasmInitialized = false;
let keys = null;

function toBase64(bytes) {
  let binary = '';
  const CHUNK = 0x8000;           // 32 768-byte slices avoid stack overflow
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  return btoa(binary);
}

const nextPaint = () => new Promise(r => requestAnimationFrame(() => r()));

async function initializeWasm() {
  if (wasmInitialized) return;
  
  log('🟡 Initialising WASM …');
  await nextPaint();
  await init();
  wasmInitialized = true;
  log('✅ WASM ready');
  await nextPaint();
}

async function ensureKeys() {
  if (keys) return keys;
  
  log('🟡 Generating keys …');
  await nextPaint();
  keys = await generateKeys();
  log('✅ Keys ready');
  await nextPaint();
  return keys;
}

async function generateKeys() {
  const { clientKey, serverKey } = keygen();
  return { clientKey, serverKey };
}

async function encrypt(keys, value) {
  const data = new BigUint64Array([BigInt(value)]);
  return enc(data, keys.clientKey);
}

async function add(ct1, ct2, serverKey) {
  const res = await fetch('/api/add', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      ct1_b64: toBase64(ct1),
      ct2_b64: toBase64(ct2),
      sk_b64: toBase64(serverKey),
    }),
  });

  if (!res.ok) {
    throw new Error(`Server error: ${await res.text()}`);
  }
  return new Uint8Array(await res.arrayBuffer());
}

async function decrypt(keys, ct) {
  const result = dec(ct, keys.clientKey);
  return Number(result[0]);
}

document.getElementById('run').onclick = async () => {
  out.textContent = '';
  
  const value1 = document.getElementById('value1').value;
  const value2 = document.getElementById('value2').value;
  
  if (!value1 || !value2) {
    log('❌ Please enter both numbers');
    return;
  }
  
  const num1 = parseInt(value1);
  const num2 = parseInt(value2);
  
  if (isNaN(num1) || isNaN(num2)) {
    log('❌ Please enter valid numbers');
    return;
  }

  await initializeWasm();
  const keys = await ensureKeys();

  log('🟡 Encrypting …');
  const ct1 = await encrypt(keys, num1);
  const ct2 = await encrypt(keys, num2);
  log('✅ Encrypted');
  await nextPaint();

  log('🟡 Computing …');
  const ctSum = await add(ct1, ct2, keys.serverKey);
  log('✅ Computed');
  await nextPaint();

  log('🟡 Decrypting …');
  const sum = await decrypt(keys, ctSum);
  log('✅ Decrypted');
  await nextPaint();

  log(`\n🎯 Result: ${sum}`);
  log(`Expected: ${num1 + num2}`);
}; 