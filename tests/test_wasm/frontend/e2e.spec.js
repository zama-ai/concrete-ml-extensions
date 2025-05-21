const { test, expect } = require('@playwright/test');

test.describe('FHE Addition Demo E2E Test', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('#out')).toContainText(/🟡 Initialising WASM|✅ WASM ready|click "Run demo" to begin/, { timeout: 20000 });
  });

  test('should perform FHE addition correctly', async ({ page }) => {
    const value1 = '250';
    const value2 = '175';
    const expectedSum = parseInt(value1) + parseInt(value2);

    await expect(page.locator('#out')).toContainText('✅ WASM ready', { timeout: 60000 });

    await page.fill('#value1', value1);
    await page.fill('#value2', value2);

    await page.click('#run');

    await expect(page.locator('#out')).toContainText('✅ Encrypted', { timeout: 20000 });
    await expect(page.locator('#out')).toContainText('✅ Computed', { timeout: 45000 });
    await expect(page.locator('#out')).toContainText('✅ Decrypted', { timeout: 20000 });

    await expect(page.locator('#out')).toContainText(`🎯 Result: ${expectedSum}`);
    await expect(page.locator('#out')).toContainText(`Expected: ${expectedSum}`);
  });

  test('should show error for empty inputs', async ({ page }) => {
    await expect(page.locator('#out')).toContainText('✅ WASM ready', { timeout: 60000 });
    
    await page.click('#run');
    await expect(page.locator('#out')).toContainText('❌ Please enter both numbers');
  });

  test('should show error for invalid number inputs', async ({ page }) => {
    await expect(page.locator('#out')).toContainText('✅ WASM ready', { timeout: 60000 });

    await page.evaluate(() => document.getElementById('value1').value = 'not_a_number');
    await page.fill('#value2', '175');
    await page.click('#run');
    await expect(page.locator('#out')).toContainText('❌ Please enter both numbers');
  });
}); 