// Real-browser smoke test: confirms the single-page terminal loads without
// console errors/failed requests, all four panels render, and the core
// interactions (ticker switch, OSINT headline -> event modal,
// header-triggered drawers, what-if simulate) work. Requires both dev
// servers running (see README.md):
//   ./.venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000   (repo root)
//   npm run dev -- --host 127.0.0.1 --port 5173                    (frontend/)
//
// Usage: node smoke_test.mjs [baseUrl]  (default http://127.0.0.1:5173)

import { chromium } from "playwright";

const BASE_URL = process.argv[2] ?? "http://127.0.0.1:5173";

const browser = await chromium.launch();
const page = await browser.newPage();

let hadErrors = false;
page.on("console", (msg) => {
  if (msg.type() === "error") {
    hadErrors = true;
    console.log(`[console.error] ${msg.text()}`);
  }
});
page.on("pageerror", (err) => {
  hadErrors = true;
  console.log(`[pageerror] ${err}`);
});
page.on("requestfailed", (req) => {
  hadErrors = true;
  console.log(`[requestfailed] ${req.url()} -> ${req.failure()?.errorText}`);
});

console.log(`=== ${BASE_URL}/ ===`);
const res = await page.goto(`${BASE_URL}/`, { waitUntil: "networkidle" });
console.log("status:", res.status());
await page.waitForTimeout(800); // let React state / lightweight-charts render flush

for (const selector of [".ticker-tape", ".terminal-left", ".terminal-center", ".terminal-right"]) {
  const count = await page.locator(selector).count();
  if (count === 0) {
    hadErrors = true;
    console.log(`[FAIL] missing panel: ${selector}`);
  } else {
    console.log(`panel present: ${selector}`);
  }
}

// Ticker switch: click JPM in the right-column ticker list, confirm the
// middle column's header updates to reflect the new selection.
await page.getByRole("button", { name: /JPM/ }).first().click();
await page.waitForTimeout(600);
const centerHeader = await page.locator(".terminal-center .panel-header").first().innerText();
if (!centerHeader.startsWith("JPM")) {
  hadErrors = true;
  console.log(`[FAIL] clicking JPM in the ticker list did not update the center panel: ${centerHeader}`);
} else {
  console.log("ticker list switch: OK");
}

// OSINT headline -> event modal.
const headline = page.locator(".osint-list .event-headline-link").first();
if ((await headline.count()) > 0) {
  await headline.click();
  await page.waitForTimeout(500);
  const modalCount = await page.locator(".modal").count();
  if (modalCount === 0) {
    hadErrors = true;
    console.log("[FAIL] clicking an OSINT headline did not open the event modal");
  } else {
    console.log("OSINT headline -> event modal: OK");
    await page.keyboard.press("Escape");
    await page.waitForTimeout(300);
    if ((await page.locator(".modal").count()) !== 0) {
      hadErrors = true;
      console.log("[FAIL] Esc did not close the event modal");
    } else {
      console.log("Esc closes modal: OK");
    }
  }
} else {
  hadErrors = true;
  console.log("[FAIL] no OSINT headlines found to click");
}

// Header-triggered drawers: What-if and Backtests, each open/close via Esc.
for (const [buttonName, drawerCheck] of [
  ["What-if", "Simulate"],
  ["Backtests", "Sector comparison"],
]) {
  await page.getByRole("button", { name: buttonName }).click();
  await page.waitForTimeout(500);
  const drawerCount = await page.locator(".drawer").count();
  if (drawerCount === 0) {
    hadErrors = true;
    console.log(`[FAIL] "${buttonName}" button did not open a drawer`);
    continue;
  }
  const hasContent = (await page.getByText(drawerCheck).count()) > 0;
  if (!hasContent) {
    hadErrors = true;
    console.log(`[FAIL] "${buttonName}" drawer missing expected content "${drawerCheck}"`);
  } else {
    console.log(`"${buttonName}" drawer: OK`);
  }
  await page.keyboard.press("Escape");
  await page.waitForTimeout(300);
}

// What-if simulate flow (re-open the drawer, run a simulation).
await page.getByRole("button", { name: "What-if" }).click();
await page.waitForTimeout(400);
await page.getByRole("button", { name: "Simulate" }).click();
await page.waitForTimeout(1000);
const whatIfResult = await page.locator(".whatif-result").count();
if (whatIfResult === 0) {
  hadErrors = true;
  console.log("[FAIL] what-if simulator produced no result panel");
} else {
  console.log("what-if simulator: OK");
}
await page.keyboard.press("Escape");

await browser.close();
console.log("\n=== DONE ===");
console.log(hadErrors ? "FAILED" : "PASSED");
process.exit(hadErrors ? 1 : 0);
