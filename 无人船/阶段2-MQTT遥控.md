# 阶段 2：同一网页改走 MQTT

依据：[FS800-4G遥控与网页用户端方案.md](FS800-4G遥控与网页用户端方案.md) 第 6、7 章。  
UI 与 JSON 仍是阶段 1，只换传输：网关写 Broker，船上 FS800 透传到 AT-COM CN6。

**本阶段不做：** 公网 HTTPS 证书、Qt、图传。

---

## 数据流

装船（FS800 已配好 MQTT）：

```
浏览器  http://127.0.0.1:8080  （UI 不变）
    │  WebSocket /ws
    ▼
python server.py --mqtt
    │  MQTT TCP 1883  publish cmd / subscribe ack
    │  Retain=false  ClientID=boat01-webgw
    ▼
公网或本机 Broker
    │
    ▼
FS800 订阅 cmd、发布 ack  → UART → AT-COM CN6
```

实验室还没有 4G / 公网 Broker 时，用 **串口桥** 代替 FS800：

```
server.py --mqtt  →  Broker  →  mqtt_bridge.py --port COMx  →  USB-TTL  →  CN6
```

4G 模组**不能**访问 `127.0.0.1`。本机 Mosquitto 只适合「网页 + 桥接 + USB-TTL」。FS800 必须填**公网可达**的 Broker（云主机或内网穿透），见总方案 6.1。

---

## ClientID（必须互不相同）

| 角色 | 默认 ClientID |
|---|---|
| 网页网关 | `boat01-webgw` |
| 串口桥（实验室） | `boat01-bridge` |
| FS800 | 配置工具里另起，如 `boat01-fs800` |
| MQTTX | `boat01-mqttx` |

主题：`boat/boat01/cmd`、`boat/boat01/ack`。全部 **Retain 关闭**。

---

## 第 1 步：Broker

本机（仅 USB-TTL 桥接实验）：

```powershell
mosquitto -c E:\zhudian\ground-web\mosquitto.conf
```

`mosquitto.conf` 开了 1883。若提示找不到命令，先安装 Eclipse Mosquitto。

装船 / FS800：Broker 放到有公网 IP 的轻量云，防火墙放行 1883（若物联网卡封 1883 则改端口）。网关加：

```powershell
python server.py --mqtt --mqtt-host 你的公网IP --mqtt-port 1883
```

---

## 第 2 步：网页网关改 MQTT

```powershell
cd E:\zhudian\ground-web
pip install -r requirements.txt
python server.py --mqtt
```

浏览器仍打开 http://127.0.0.1:8080 ，点 **连接 Broker**。不必选 COM（COM 是桥接进程的事）。

| 参数 | 含义 |
|---|---|
| `--mqtt` | 网关发 MQTT，不再直接打开 COM |
| `--mqtt-host` | Broker 地址，默认 127.0.0.1 |
| `--mqtt-port` | 默认 1883 |
| `--mqtt-user` / `--mqtt-password` | 可选 |
| `--mqtt-client-id` | 默认 `boat01-webgw` |

未装上时：`pip install paho-mqtt`。本机若打不开 PyPI，可换镜像或 `conda install -c conda-forge paho-mqtt`。阶段 1 的 `--mock` / 直连串口不需要这个包。

---

## 第 3 步 A：实验室桥接（无 FS800）

USB-TTL 已接 CN6，Mosquitto 已开：

```powershell
python mqtt_bridge.py --port COM5
```

把 `COM5` 换成设备管理器里的口。桥接 ClientID 是 `boat01-bridge`。

无硬件试 MQTT 通路：

```powershell
python mqtt_bridge.py --mock
```

拖 T=20，应答应为 `OK T=200 Y=0`。

---

## 第 3 步 B：FS800 透传（有模组）

1. USB-TTL **5 V** 单独配模组（此时不要占 CN6）。
2. 配置工具：MQTT、公网 Broker、订阅 `boat/boat01/cmd`、发布 `boat/boat01/ack`、波特率 **115200**、Retain 关、ClientID=`boat01-fs800`。
3. MQTTX（第三个 ClientID）先发 `{"mode":"stop"}`，USB-TTL 串口应看到同一行。
4. 拔 USB-TTL，模组 TX/RX 交叉接 CN6，VIN 接 AT-COM DC5V。
5. `python server.py --mqtt --mqtt-host <公网>`，网页遥控。
6. **不要**再运行 `mqtt_bridge.py`（和 FS800 会抢同一主题）。

---

## 验收

| 项 | 通过标准 |
|---|---|
| 桥接 mock | `--mqtt` 网关 + `mqtt_bridge.py --mock`，T=20 → `OK T=200 Y=0` |
| 桥接真机 | 舵机随 T/Y；停网页约 5.2 s 停车 |
| FS800 | 同上，且拔天线约 5.2 s 停车 |
| Retain | 重连后船不应自己加速 |
| ClientID | 网关与模组同时在线，互不踢掉 |
| 单页 | 第二个浏览器标签仍提示已有操控页 |

---

## 和阶段 3 的衔接

阶段 3 把 Broker 换成 TLS/WSS、网页可改为直连 Broker 的 MQTT.js。现在仍由本机网关代发，便于沿用阶段 1 的单页锁和急停。
