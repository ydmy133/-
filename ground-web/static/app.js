(() => {
  const HZ = 10;
  const CMD_TOPIC = "boat/boat01/cmd";

  const $ = (id) => document.getElementById(id);
  const portSel = $("port");
  const tRange = $("tRange");
  const yRange = $("yRange");
  const tVal = $("tVal");
  const yVal = $("yVal");
  const pad = $("pad");
  const knob = $("knob");
  const estopBtn = $("estop");

  let ws = null;
  let cmdTopic = CMD_TOPIC;
  let serialOpen = false;
  let stopped = true;
  let T = 0;
  let Y = 0;
  let lastSent = "";

  function clamp(n) {
    n = Math.round(Number(n) || 0);
    if (n > 100) return 100;
    if (n < -100) return -100;
    return n;
  }

  function setTY(t, y, fromPad, fromEstop) {
    T = clamp(t);
    Y = clamp(y);
    tRange.value = String(T);
    yRange.value = String(Y);
    tVal.textContent = String(T);
    yVal.textContent = String(Y);
    if (!fromPad) syncKnob();
    if (!fromEstop) {
      stopped = false;
      estopBtn.classList.remove("armed");
    }
    $("lastCmd").textContent = compactCmd();
  }

  function syncKnob() {
    const r = 82;
    knob.style.transform = `translate(calc(-50% + ${(Y / 100) * r}px), calc(-50% + ${(-T / 100) * r}px))`;
  }

  function compactCmd() {
    if (stopped) return '{"mode":"stop"}';
    return `{"mode":"speed","T":${T},"Y":${Y}}`;
  }

  function sendWs(obj) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify(obj));
  }

  function expectedAck(payload) {
    if (payload.indexOf('"stop"') >= 0) return "OK T=0 Y=0";
    const tm = payload.match(/"T":(-?\d+)/);
    const ym = payload.match(/"Y":(-?\d+)/);
    const t = tm ? Number(tm[1]) : 0;
    const y = ym ? Number(ym[1]) : 0;
    return `OK T=${t * 10} Y=${y * 10}`;
  }

  function publish(payload, flush) {
    lastSent = payload;
    $("lastCmd").textContent = payload;
    const msg = { op: "publish", topic: cmdTopic, payload };
    if (flush) msg.flush = true;
    sendWs(msg);
  }

  function emergencyStop() {
    stopped = true;
    setTY(0, 0, false, true);
    estopBtn.classList.add("armed");
    if (serialOpen) publish('{"mode":"stop"}', true);
    $("log").textContent = "急停中：拖动 T/Y 滑条或摇杆后才会发 speed";
  }

  function fillPorts(ports) {
    const cur = portSel.value;
    portSel.innerHTML = "";
    const list = ports && ports.length ? ports : [];
    if (!list.length) {
      const o = document.createElement("option");
      o.value = "";
      o.textContent = "未发现 COM 口";
      portSel.appendChild(o);
      return;
    }
    for (const p of list) {
      const o = document.createElement("option");
      if (typeof p === "string") {
        o.value = p;
        o.textContent = p;
      } else {
        o.value = p.device || "";
        o.textContent = p.label || p.device || "";
      }
      portSel.appendChild(o);
    }
    const values = [...portSel.options].map((x) => x.value);
    if (cur && values.includes(cur)) portSel.value = cur;
  }

  function setWsUi(ok, text) {
    $("wsDot").className = "dot " + (ok ? "on" : "off");
    $("wsLabel").textContent = text;
  }

  function setSerUi(ok, text) {
    $("serDot").className = "dot " + (ok ? "on" : ok === false ? "off" : "warn");
    $("serLabel").textContent = text;
  }

  function connectGateway() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.onopen = () => setWsUi(true, "网关已连接");
    ws.onclose = (ev) => {
      serialOpen = false;
      setWsUi(false, ev.code === 4000 ? "已有其它操控页" : "网关断开");
      setSerUi(false, "串口未开");
    };
    ws.onerror = () => $("log").textContent = "WebSocket 错误";
    ws.onmessage = (ev) => {
      let msg;
      try {
        msg = JSON.parse(ev.data);
      } catch {
        return;
      }
      if (msg.op === "hello") {
        if (msg.cmdTopic) cmdTopic = msg.cmdTopic;
        const ports = msg.ports ? msg.ports.slice() : [];
        if (msg.mock && !ports.includes("MOCK")) ports.unshift("MOCK");
        fillPorts(ports);
        if (msg.mode === "mqtt") {
          $("log").textContent = "MQTT 模式：点连接后网关发到 Broker（Retain 关）";
          $("connect").textContent = "连接 Broker";
        } else if (msg.mock) {
          $("log").textContent = "模拟串口模式，无需硬件";
        }
      } else if (msg.op === "ports") {
        const ports = msg.ports ? msg.ports.slice() : [];
        fillPorts(ports);
      } else if (msg.op === "status") {
        serialOpen = !!msg.serial;
        if (msg.error) {
          setSerUi(false, msg.error);
          $("connect").textContent = "连接";
        } else if (serialOpen) {
          setSerUi(true, msg.mock ? "模拟已开" : (msg.mode === "mqtt" ? `MQTT ${msg.port || ""}` : `已开 ${msg.port || ""}`));
          $("connect").textContent = "断开";
          stopped = true;
          publish('{"mode":"stop"}');
          $("log").textContent = "已连接，当前急停。请拖 T 滑条；摇杆请勾选「松手保持」再按住。";
          if (!msg.mock) {
            window.setTimeout(() => {
              if (!serialOpen) return;
              const ack = ($("lastAck").textContent || "").trim();
              const io = ($("serIo").textContent || "");
              if (ack === "—" || ack === "") {
                $("log").textContent = "电脑在发，但 MCU 没回数据。请选 USB-TTL（CH340/CP2102）那个口，不要选 ST-Link；TX/RX 必须交叉接 CN6；不要和 FS800 同时占 CN6；然后重新烧录 AT-COM。";
              }
              if (io.indexOf("RX 0") >= 0 || io.indexOf("回了 0") >= 0) {
                $("log").textContent = "RX=0：这个 COM 口上完全没收到 MCU 字节。多半接错口，或 TX/RX 没交叉，或 CN6 被 FS800 占用。";
              }
            }, 2000);
          }
        } else {
          setSerUi(false, "串口未开");
          $("connect").textContent = "连接";
        }
      } else if (msg.op === "sent") {
        $("lastCmd").textContent = msg.payload || lastSent;
        $("serIo").textContent = `电脑已写 ${msg.tx || 0} 字节，MCU 回了 ${msg.rx || 0} 字节`;
      } else if (msg.op === "message") {
        const text = (msg.payload || "").trim();
        $("lastAck").textContent = text;
        const exp = expectedAck(lastSent);
        if (exp && !text.startsWith(exp)) {
          $("log").textContent = `应答还停在旧值（当前应是 ${exp}）。请重新编译烧录 AT-COM 后再 Ctrl+F5 刷新本页`;
        } else if (text.startsWith("OK T=0 Y=0") && stopped) {
          $("log").textContent = "急停已生效";
        } else {
          $("log").textContent = `应答已跟上 ${exp}`;
        }
      } else if (msg.op === "log" || msg.op === "rejected") {
        $("log").textContent = msg.text || msg.reason || "";
      }
    };
  }

  $("refresh").onclick = () => sendWs({ op: "list_ports" });
  $("connect").onclick = () => {
    if (serialOpen) {
      sendWs({ op: "close" });
      return;
    }
    sendWs({ op: "open", port: portSel.value });
  };
  estopBtn.onclick = () => emergencyStop();
  document.addEventListener("keydown", (ev) => {
    if (ev.code !== "Space" && ev.key !== " ") return;
    const tag = (ev.target && ev.target.tagName) || "";
    if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
    if (tag === "BUTTON" && ev.target !== estopBtn) return;
    ev.preventDefault();
    emergencyStop();
  });

  function bindRange(el, axis) {
    const apply = () => {
      if (axis === "T") setTY(el.value, Y);
      else setTY(T, el.value);
    };
    el.addEventListener("input", apply);
    el.addEventListener("change", apply);
  }
  bindRange(tRange, "T");
  bindRange(yRange, "Y");
  $("tZero").onclick = () => setTY(0, Y);
  $("yZero").onclick = () => setTY(T, 0);
  document.querySelectorAll("button[data-axis]").forEach((btn) => {
    btn.onclick = () => {
      const d = Number(btn.dataset.d);
      if (btn.dataset.axis === "T") setTY(T + d, Y);
      else setTY(T, Y + d);
    };
  });

  function padPos(ev) {
    const rect = pad.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    const x = (ev.clientX - cx) / (rect.width / 2);
    const y = (cy - ev.clientY) / (rect.height / 2);
    const t = clamp(y * 100);
    const yy = clamp(x * 100);
    setTY(t, yy, true);
    syncKnob();
  }

  let dragging = false;
  pad.addEventListener("pointerdown", (ev) => {
    dragging = true;
    pad.setPointerCapture(ev.pointerId);
    padPos(ev);
  });
  pad.addEventListener("pointermove", (ev) => {
    if (dragging) padPos(ev);
  });
  function padUp() {
    if (!dragging) return;
    dragging = false;
    if (!$("padHold").checked) setTY(0, 0);
  }
  pad.addEventListener("pointerup", padUp);
  pad.addEventListener("pointercancel", padUp);

  document.addEventListener("visibilitychange", () => {
    if (document.hidden && serialOpen) emergencyStop();
  });
  window.addEventListener("beforeunload", () => {
    if (serialOpen) {
      sendWs({ op: "publish", topic: cmdTopic, payload: '{"mode":"stop"}' });
      sendWs({ op: "close" });
    }
  });

  setInterval(() => {
    if (!serialOpen) return;
    const payload = compactCmd();
    if (stopped) {
      publish('{"mode":"stop"}');
      return;
    }
    publish(payload);
  }, 1000 / HZ);

  setTY(0, 0);
  connectGateway();
})();
