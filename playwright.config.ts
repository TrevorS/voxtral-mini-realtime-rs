import { defineConfig } from "@playwright/test";
import { join } from "node:path";
import { readdirSync } from "node:fs";

// Find the full Chromium binary (not headless_shell)
const cacheDir = join(process.env.HOME ?? "", ".cache/ms-playwright");
const chromiumDir = readdirSync(cacheDir).find(
  (d) => d.startsWith("chromium-") && !d.includes("headless")
);
const chromiumBase = chromiumDir
  ? join(cacheDir, chromiumDir, "chrome-linux")
  : "";

export default defineConfig({
  testDir: "./tests",
  testMatch: "**/*.spec.ts",
  timeout: 600_000, // 10 min
  projects: [
    {
      name: "chromium-webgpu",
      use: {
        browserName: "chromium",
        headless: false, // Let our --headless=new handle it
        launchOptions: {
          executablePath: join(chromiumBase, "chrome"),
          env: {
            ...process.env,
            // Use the real NVIDIA Vulkan driver, not SwiftShader
            VK_ICD_FILENAMES: "/usr/share/vulkan/icd.d/nvidia_icd.json",
          },
          args: [
            "--headless=new",
            "--ozone-platform=headless",
            "--no-sandbox",
            "--disable-gpu-sandbox",
            "--enable-unsafe-webgpu",
            "--enable-features=Vulkan,WebGPU",
            "--use-angle=vulkan",
            "--enable-gpu-rasterization",
            "--ignore-gpu-blocklist",
            "--js-flags=--max-old-space-size=8192",
          ],
        },
      },
    },
  ],
});
