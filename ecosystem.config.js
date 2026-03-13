module.exports = {
  apps: [
    {
      name: "weapon-detect",
      script: "/Users/faith/Documents/codes/ai-imagealize/.venv/bin/uvicorn",
      args: "app.main:app --host 0.0.0.0 --port 8000",
      cwd: "/Users/faith/Documents/codes/ai-imagealize",
      interpreter: "none",
      env: {
        WEAPON_DETECT_STREAM_FPS: "15",
        WEAPON_DETECT_STREAM_INFERENCE_FPS: "10",
        WEAPON_DETECT_STREAM_PREVIEW_FPS: "15",
        WEAPON_DETECT_FRAME_SKIP: "1",
        WEAPON_DETECT_ENABLE_STREAM_RECORDING: "true",
        WEAPON_DETECT_STREAM_RECORD_FPS: "10"
      }
    }
  ]
};
