<div align="center">
<img width="1200" height="300" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# Run and deploy your AI Studio app

This contains everything you need to run your app locally.

View your app in AI Studio: https://ai.studio/apps/drive/1uA68LGio1CJePNXz2emCcow0SASjOXf4

## Run Locally

**前提條件:**  Node.js


1. 安裝依賴項:
   `npm install` 或 `yarn install`  
2. 執行應用程式:  
   `npm run dev` 或 `yarn run dev`
3. 執行Chrome 或是 Edge 網址列輸入 `chrome://flags/`, 搜尋 `Insecure origins treated as secure` 輸入本地伺服器 IP 以繞過安全檢查並使用 WebGPU  
範例: http://192.168.31.92:3000,192.168.31.92:3000,http://localhost:3000  

# LocalAI-Translator

這是一個簡潔且現代的翻譯應用程式，支援 Google Gemini 與 OpenAI 相容 API；同時也支援離線開源模型以防止敏感資料外洩。它提供了一個乾淨、行動裝置優先的使用者介面，可在多種語言之間進行快速準確的文字翻譯。  
🚀[Live Demo](https://willo83417.github.io/LocalAI-Translator/ "LocalAI-Translator")

## 📸 截圖  


<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="https://github.com/willo83417/LocalAI-Translator/blob/gh-pages/screenshots/P1.png" width="200" alt="Screenshot 1"/>
      </td>
      <td align="center">
        <img src="https://github.com/willo83417/LocalAI-Translator/blob/gh-pages/screenshots/P2.png" width="200" alt="Screenshot 2"/>
      </td>
      <td align="center">
        <img src="https://github.com/willo83417/LocalAI-Translator/blob/gh-pages/screenshots/P3.png" width="200" alt="Screenshot 3"/>
      </td>
      <td align="center">
        <img src="https://github.com/willo83417/LocalAI-Translator/blob/gh-pages/screenshots/P4.png" width="200" alt="Screenshot 4"/>
      </td>
    </tr>
    <tr>
      <td align="center">
        <img src="https://github.com/willo83417/LocalAI-Translator/blob/gh-pages/screenshots/P5.png" width="200" alt="Screenshot 5"/>
      </td>
      <td align="center">
        <img src="https://github.com/willo83417/LocalAI-Translator/blob/gh-pages/screenshots/P6.png" width="200" alt="Screenshot 6"/>
      </td>
      <td align="center">
        <img src="https://github.com/willo83417/LocalAI-Translator/blob/gh-pages/screenshots/P7.png" width="200" alt="Screenshot 7"/>
      </td>
      <td align="center">
        <img src="https://github.com/willo83417/LocalAI-Translator/blob/gh-pages/screenshots/P8.png" width="200" alt="Screenshot 8"/>
      </td>
    </tr>
  </table>
</div> 


## 設計特點

### 核心翻譯功能💬
- **文字翻譯**: 支援即時串流翻譯，在輸入文字時即時顯示翻譯結果。
- **語音翻譯**:
  - **一般錄音**: 將使用者的語音轉換為文字並填入輸入框。
  - **即時語音翻譯 (AST)**: 直接將使用者的語音翻譯成目標語言，並以文字顯示，實現更流暢的對話式翻譯。
- **圖像翻譯**: 透過裝置相機或從相簿匯入圖片，應用程式能自動偵測並擷取圖片中的文字，然後進行翻譯。

### 線上與離線模式🌐
- **線上模式**: 利用強大的雲端 AI 模型（可選 Gemini 或 OpenAI）進行高品質翻譯。此模式需要網路連線和有效的 API 金鑰。
- **離線模式**: 在沒有網路連線的情況下，使用下載到裝置本地的 Gemma 模型（透過 MediaPipe 執行）進行翻譯。離線模式支援文字、語音轉文字以及圖像文字擷取。

### 多國語言支援🌟
應用程式支援多種主流語言，包括英文、繁體中文、簡體中文、日文、韓文、西班牙文等，並提供「自動偵測」來源語言的功能。

### 個人化設定🔒
- **API 金鑰與模型管理**: 使用者可以在設定中輸入自己的 Gemini 或 OpenAI API 金鑰，並自訂要使用的模型名稱。
- **離線模型下載**: 提供一個完整的介面來下載、管理和刪除用於離線模式的本地模型。
- **離線文字轉語音 (TTS)**: 使用者可以自訂離線 TTS 的語音、速度和音高，以獲得個人化的聆聽體驗。
- **高準確度日中翻譯模式**: 針對日文翻譯至中文的場景，提供一個特殊的「兩步翻譯」模式（日→英→中），以提升翻譯的準確性和流暢度。

### 歷史紀錄📊
自動儲存最近的 50 筆翻譯紀錄，方便使用者隨時查閱和重複使用。

## 文件架構🏗️

- `/` (根目錄): 包含專案的入口點和主要設定檔，如 `index.html`, `index.tsx`, `package.json`, `vite.config.ts`, `App.tsx`, `constants.ts`, `types.ts`, `i18n.ts`。
- `/components`: 存放所有 React UI 元件。每個元件負責一部分 UI 的渲染和互動。
  - `TranslationInput.tsx`: 來源語言輸入介面。
  - `TranslationOutput.tsx`: 目標語言輸出介面。
  - `SettingsModal.tsx`: 設定彈出視窗。
  - `HistoryModal.tsx`: 歷史紀錄彈出視窗。
  - `CameraView.tsx`: 相機翻譯介面。
  - `LanguageSelector.tsx`: 自訂語言選擇下拉選單。
- `/services`: 存放應用程式的核心業務邏輯，特別是與 API 和後端服務的互動。
  - `geminiService.ts`: 處理與 Google Gemini API 的所有通訊。
  - `openaiService.ts`: 處理與 OpenAI API 的所有通訊。
  - `asrService.ts`: 管理音訊處理與語音辨識服務。
  - `downloadManager.ts`: 負責離線模型的下載、暫停、續傳和刪除。
- `/workers`: 包含 Web Workers，用於背景處理以保持 UI 響應。
  - `litert-lm.worker.ts`: 離線 LLM 的主要推論 Worker (使用 LiteRT)。
  - `mediapipe.worker.ts`: 離線 LLM 的傳統推論 Worker (使用 MediaPipe)。
  - `offline.worker.ts`: 離線模型管理的背景 Worker。
  - `nemotron.worker.ts`: 使用 ONNX Runtime 進行即時語音辨識 (ASR) 的 Worker (Nemotron 模型)。
  - `transformersASR.worker.ts`: 使用 Hugging Face Transformers.js 進行語音辨識 (ASR) 的 Worker (Whisper 模型)。
  - `ocr.worker.ts`: 使用 esearch-ocr 與 ONNX Runtime 進行本地圖像文字擷取 (OCR) 的 Worker。
- `/hooks`: 用於特定邏輯的自訂 React Hook。
  - `usePaddleOcr.ts`: 使用 PaddleOCR 進行本地 OCR 的 Hook。
  - `useWebSpeech.ts`: Web Speech API 整合的 Hook。
- `/utils`: 工具函式與共享輔助工具。
  - `db.ts`: 用於本地儲存的 IndexedDB 管理。
- `/types.ts`: 定義整個應用程式中使用的 TypeScript 型別。
- `/constants.ts`: 存放應用程式的常數，如支援的語言列表、離線模型資訊等。
- `/i18n.ts`: i18next 國際化設定檔，管理多語言介面。

## UI 功能說明

### 主介面
主介面分為左右兩個區塊，分別用於輸入和輸出。

- **來源語言區塊 (`TranslationInput`)**:
  - **文字輸入區**: 用於輸入或貼上要翻譯的文字。
  - **語言選擇器**: 選擇來源語言，支援自動偵測。
  - **功能按鈕**:
    - `翻譯`: 執行翻譯。
    - `錄音`: 啟動麥克風進行語音輸入。
    - `相機`: 開啟相機進行圖像翻譯。
    - `設定`: 開啟設定視窗。
  - **狀態顯示**: 顯示字元數和目前的網路連線狀態（線上/離線）。

- **目標語言區塊 (`TranslationOutput`)**:
  - **翻譯結果顯示區**: 顯示翻譯後的文字。
  - **語言選擇器**: 選擇目標語言。
  - **功能按鈕**:
    - `交換語言`: 快速對調來源和目標語言。
    - `翻轉螢幕`: 將此區塊的文字鏡像翻轉，方便向對面的人展示。
    - `複製`: 複製翻譯結果到剪貼簿。
    - `歷史紀錄`: 開啟歷史紀錄視窗。
    - `清除`: 清除翻譯結果。
  - **文字轉語音 (TTS)**: 提供男聲和女聲按鈕（或自訂語音），朗讀翻譯結果。
  - **即時語音翻譯 (AST)**: 啟動後，直接將語音輸入翻譯成目標語言。

### 彈出視窗
- **設定 (`SettingsModal`)**:
  - **線上分頁**: 設定 API 金鑰、服務提供商（Gemini/OpenAI）和模型名稱。
  - **離線分頁**: 管理 Hugging Face API 金鑰、下載/刪除離線模型、啟用離線模式及高準確度日中模式。
  - **離線 TTS 分頁**: 設定自訂語音、速率和音高。
- **歷史紀錄 (`HistoryModal`)**: 顯示過去的翻譯紀錄列表，可點擊以重新載入。
- **相機 (`CameraView`)**:
  - 提供即時的相機預覽畫面。
  - 控制項包含：擷取照片、從相簿匯入、閃光燈開關、焦距縮放。

## 技術棧🚀

- **前端框架**: React 19
- **語言**: TypeScript
- **建構工具**: Vite
- **樣式**: Tailwind CSS
- **離線模型**: Google Gemma, Whisper, Nemotron, PaddleOCR
- **離線推論引擎**: LiteRT, MediaPipe, Transformers.js, ONNX Runtime Web
- **線上翻譯**: Google Gemini API (`@google/genai`) / OpenAI API
- **國際化**: i18next  

## 工作分配:📖  
[**LOGO - Design**]：NanoBanana  
[**AI Studio - Gemini 2.5Pro**]：80% (大部分UI、程式碼設計)  
[**Copilot -in Edge**]：10% (解決Gemini 2.5Pro無法處理得錯誤)  
[**人類**]：10% (提出概念、測試、Debug)  
## 注意事項⚠️
## GitHub Pages 設定  
1. 修改 package.json
   ```
   "homepage": "https://<your-github-username>.github.io/repository name", ← 新增這行
   "deploy": "gh-pages -d dist", ← 新增這行  
   ```
2. 修改 vite.config.ts
   ```
   base: '/repository name/', // ← 新增這行
   plugins: [],
   ```
   ```
           icons: [
          {
            "src": "images/icon-192.png", ←修改成 "images/icon-192.png" 而非 "/images/icon-192.png"
            "sizes": "192x192",
            "type": "image/png",
            "purpose": "any maskable"
          },
          {
            "src": "images/icon-512.png", ←修改成 "images/icon-512.png" 而非 "/images/icon-512.png"
            "sizes": "512x512",
            "type": "image/png"
          }
        ]
   ```
3. 安裝 gh-pages  
   `npm install gh-pages --save-dev` 或 `yarn add gh-pages -D`  
4. 輸出靜態網站檔案  
   `npm build` 或 `yarn build`
5. 推送到 GitHub Pages  
    `npm run deploy` 或 `yarn run deploy`
    