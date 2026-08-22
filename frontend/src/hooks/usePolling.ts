import { useEffect, useRef } from "react";

// Generic interval hook, ref-based to avoid stale closures over `callback`.
// Fires once immediately, then every `intervalMs`. Pauses while the tab is
// hidden (backgrounded) and re-fires immediately on becoming visible again,
// so an unattended background tab doesn't keep hitting the API.
export function usePolling(callback: () => void, intervalMs: number, enabled = true): void {
  const callbackRef = useRef(callback);
  callbackRef.current = callback;

  useEffect(() => {
    if (!enabled) return;

    let timer: ReturnType<typeof setInterval> | null = null;

    function start() {
      callbackRef.current();
      timer = setInterval(() => callbackRef.current(), intervalMs);
    }

    function stop() {
      if (timer !== null) {
        clearInterval(timer);
        timer = null;
      }
    }

    function onVisibilityChange() {
      if (document.hidden) {
        stop();
      } else {
        start();
      }
    }

    if (!document.hidden) start();
    document.addEventListener("visibilitychange", onVisibilityChange);

    return () => {
      stop();
      document.removeEventListener("visibilitychange", onVisibilityChange);
    };
  }, [intervalMs, enabled]);
}
