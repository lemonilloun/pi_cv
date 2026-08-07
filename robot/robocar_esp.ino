/*
 * RoboCar — ESP8266 + TB6612FNG
 * Автопоиск сервера в сети + настройка WiFi без перепрошивки
 *
 * Плата: NodeMCU 1.0 (ESP-12E Module), Serial Monitor 115200
 *
 * КАК РАБОТАЕТ ПОДКЛЮЧЕНИЕ:
 *   1. Пробует сети: сначала сохранённую в EEPROM, потом из списка KNOWN[]
 *   2. Если ни одна не подошла — поднимает свою точку "RoboCar-Setup"
 *      (пароль robocar123). Заходишь с телефона на http://192.168.4.1,
 *      выбираешь сеть, вводишь пароль. Сохраняется навсегда.
 *   3. В сети шлёт UDP-broadcast "кто сервер?" и ждёт ответа.
 *      Найдя — подключается по TCP. IP сервера знать заранее не нужно.
 *
 * ВНИМАНИЕ: SECRET ниже — не шифрование, а только метка "свой сервер".
 * Любой в той же сети может подключиться к машинке.
 *
 * Распиновка TB6612FNG:
 *   PWMA -> D5   AIN1 -> D1   AIN2 -> D2
 *   PWMB -> D6   BIN1 -> D7   BIN2 -> D0
 *   STBY -> 3V3  VCC  -> 3V3  GND  -> GND
 */

#include <ESP8266WiFi.h>
#include <WiFiUdp.h>
#include <ESP8266WebServer.h>
#include <EEPROM.h>

// ==================== НАСТРОЙКИ ====================

// Общий с сервером токен. Поменяй на свой — но одинаково здесь и в server.py
const char* SECRET = "ROBOCAR-2026";

const uint16_t DISCOVERY_PORT = 5001;   // UDP, поиск сервера

// Сети, которые пробуются автоматически. Можно оставить пустым.
struct Net { const char* ssid; const char* pass; };
Net KNOWN[] = {
  { "",  "" },
  { "",  "" },
};
const uint8_t KNOWN_COUNT = sizeof(KNOWN) / sizeof(KNOWN[0]);

// Точка доступа для настройки
const char* AP_SSID = "RoboCar-Setup";
const char* AP_PASS = "robocar123";      // минимум 8 символов

// ==================== ПИНЫ ====================
// --- Camera pan servo (added 2026-08-03) -----------------------------------
// D4 / GPIO2 is the spare pin on this board. It is also the onboard LED, so
// the LED flickers while the servo is being driven; harmless.
//
// IMPORTANT, and the reason the servo is attached and detached rather than
// held: on the ESP8266 the Servo library and analogWrite() both use timer1,
// and the motors are driven by analogWrite. Holding a servo attached while
// driving degrades the motor PWM. So PAN refuses while the wheels are turning,
// moves the servo, waits for it to arrive, and detaches again. That matches
// how the robot is actually used — park, look around, drive on.
#include <Servo.h>
#define PIN_SERVO D4
Servo panServo;
int  panAngle = 90;          // servo degrees; 90 = straight ahead
const uint16_t PAN_TRAVEL_MS = 450;

#define PIN_PWMA  D5
#define PIN_AIN1  D1
#define PIN_AIN2  D2
#define PIN_PWMB  D6
#define PIN_BIN1  D7
#define PIN_BIN2  D0

// ==================== ДВИЖЕНИЕ ====================
int      MAX_PWM     = 150;   // сервер поднимает его при каждом коннекте
                              // (DEFAULT_MAX_PWM); 80 подбирали на голом
                              // шасси, оно едва везёт Pi 5 с камерой
int      MIN_PWM     = 40;
uint32_t FAILSAFE_MS = 400;
const int      RAMP_STEP = 8;
const uint32_t RAMP_MS   = 15;

// ==================== EEPROM ====================
#define CREDS_MAGIC 0xC0FFEE02
struct Creds {
  uint32_t magic;
  char ssid[33];
  char pass[65];
};
Creds creds;

// ==================== СОСТОЯНИЕ ====================
enum RunMode { MODE_STA, MODE_PORTAL };
RunMode runMode = MODE_STA;

WiFiClient      client;
WiFiUDP         udp;
ESP8266WebServer portal(80);

IPAddress serverIP;
uint16_t  serverPort = 0;
bool      serverFound = false;

int targetL = 0, targetR = 0;
int curL    = 0, curR    = 0;

uint32_t lastCmdMs = 0, lastRampMs = 0, lastTeleMs = 0;
uint32_t lastProbeMs = 0, lastRetryMs = 0;
bool     failsafeHit = false;

char    rxBuf[160];
uint8_t rxLen = 0;

bool     ledManual = false;
uint16_t blinkOn = 80, blinkOff = 80;
uint32_t blinkTs = 0;
bool     ledOn = false;

// ================= СВЕТОДИОД =================
void setLed(bool on) { digitalWrite(LED_BUILTIN, on ? LOW : HIGH); }
void setPattern(uint16_t on, uint16_t off) { blinkOn = on; blinkOff = off; }

void updateLed() {
  if (ledManual) return;
  uint32_t now = millis();
  if (now - blinkTs >= (uint32_t)(ledOn ? blinkOn : blinkOff)) {
    blinkTs = now;
    ledOn = !ledOn;
    setLed(ledOn);
  }
}

// ================= ОТПРАВКА =================
void send(const String& s) {
  if (client.connected()) { client.print(s); client.print('\n'); }
  Serial.print("TX: "); Serial.println(s);
}

// ================= МОТОРЫ =================
const char* dirName(int v) { return v > 0 ? "FWD" : (v < 0 ? "REV" : "STOP"); }

void applyMotor(bool chA, int v) {
  uint8_t in1 = chA ? PIN_AIN1 : PIN_BIN1;
  uint8_t in2 = chA ? PIN_AIN2 : PIN_BIN2;
  uint8_t pwm = chA ? PIN_PWMA : PIN_PWMB;

  if (v == 0) {
    digitalWrite(in1, LOW); digitalWrite(in2, LOW);
    analogWrite(pwm, 0);
    return;
  }
  if (v > 0) { digitalWrite(in1, HIGH); digitalWrite(in2, LOW);  }
  else       { digitalWrite(in1, LOW);  digitalWrite(in2, HIGH); }
  analogWrite(pwm, abs(v));
}

void applyBrake() {
  digitalWrite(PIN_AIN1, HIGH); digitalWrite(PIN_AIN2, HIGH);
  digitalWrite(PIN_BIN1, HIGH); digitalWrite(PIN_BIN2, HIGH);
  analogWrite(PIN_PWMA, 255);   analogWrite(PIN_PWMB, 255);
  curL = curR = targetL = targetR = 0;
}

// Ниже MIN_PWM мотор гудит и греется, но колесо не крутится: заклиненный
// коллекторный двигатель тянет СТОПОРНЫЙ ток -- самый большой, какой он вообще
// потребляет, и весь он уходит в тепло. Поэтому команда внутри мёртвой зоны
// хуже нуля: жрёт максимум батареи и не даёт движения.
int scaleCmd(int v) {
  if (abs(v) < 5) return 0;
  int m = map(abs(v), 5, 255, MIN_PWM, MAX_PWM);
  m = constrain(m, MIN_PWM, MAX_PWM);
  return v > 0 ? m : -m;
}

// Пара колёс масштабируется ОДНИМ множителем, а не каждое само по себе.
//
// Раньше каждое колесо гналось через scaleCmd отдельно, и это сжимало РАЗНИЦУ
// между ними -- а разница и есть поворот. Замер: пара (200, 86), отношение
// 0.43, выходила как (125, 75), отношение 0.60. Поворот слабел на ровном
// месте, ещё до того как его начинала душить просадка аккумулятора.
//
// Здесь через кривую MIN..MAX проводится только большее колесо, а меньшее
// получает тот же множитель. Отношение сохраняется точно; затем то, что
// попало в мёртвую зону, поднимается до MIN_PWM или обнуляется -- висеть
// внутри неё нельзя.
void scalePair(int l, int r, int *outL, int *outR) {
  int big = max(abs(l), abs(r));
  if (big < 5) { *outL = 0; *outR = 0; return; }

  int scaled = abs(scaleCmd(big));            // большее колесо на кривой
  float k = (float)scaled / (float)big;       // общий множитель для обоих

  int nl = (int)lroundf(l * k);
  int nr = (int)lroundf(r * k);

  // Меньшее колесо могло попасть в мёртвую зону. Поднимаем ВВЕРХ, а не
  // обнуляем: команда чуть ниже порога и так стоит полного тока, так что
  // выбросить это колесо значило бы потерять его вклад, уже оплаченный
  // батареей.
  if (nl != 0 && abs(nl) < MIN_PWM) nl = (nl > 0) ? MIN_PWM : -MIN_PWM;
  if (nr != 0 && abs(nr) < MIN_PWM) nr = (nr > 0) ? MIN_PWM : -MIN_PWM;

  *outL = constrain(nl, -255, 255);
  *outR = constrain(nr, -255, 255);
}

// Колёса разгоняются СИНХРОННО, одной долей пути, а не каждое своим шагом.
//
// Раньше каждое колесо шло к своей цели по RAMP_STEP за тик. Если одному
// колесу ехать дальше (а при повороте так всегда), то пока быстрое уже
// доехало, медленное ещё в пути -- и всё это время реальная разница между
// колёсами не та, что просили. Машина успевала «клюнуть» прямо, прежде чем
// начать поворот. Здесь обе цели достигаются за одно и то же число тиков,
// поэтому отношение колёс верное на всём разгоне.
void updateRamp() {
  if (millis() - lastRampMs < RAMP_MS) return;
  lastRampMs = millis();
  if (curL == targetL && curR == targetR) return;

  int dl = targetL - curL;
  int dr = targetR - curR;
  int biggest = max(abs(dl), abs(dr));
  if (biggest <= RAMP_STEP) {
    curL = targetL;
    curR = targetR;
  } else {
    // Общая доля пути: большее из расхождений проходит ровно RAMP_STEP,
    // меньшее -- пропорционально меньше.
    float frac = (float)RAMP_STEP / (float)biggest;
    curL += (int)lroundf(dl * frac);
    curR += (int)lroundf(dr * frac);
  }
  applyMotor(true, curL);
  applyMotor(false, curR);
}

void checkFailsafe() {
  if (targetL == 0 && targetR == 0) return;
  if (millis() - lastCmdMs > FAILSAFE_MS) {
    targetL = targetR = 0;
    if (!failsafeHit) {
      failsafeHit = true;
      send("FAILSAFE нет команд, стоп");
      Serial.println("!! FAILSAFE");
    }
  }
}

void motorsInit() {
  pinMode(PIN_PWMA, OUTPUT); pinMode(PIN_AIN1, OUTPUT); pinMode(PIN_AIN2, OUTPUT);
  pinMode(PIN_PWMB, OUTPUT); pinMode(PIN_BIN1, OUTPUT); pinMode(PIN_BIN2, OUTPUT);
  applyMotor(true, 0); applyMotor(false, 0);
  analogWriteRange(255);
  analogWriteFreq(1000);
}

// ================= EEPROM =================
void loadCreds() {
  EEPROM.begin(sizeof(Creds));
  EEPROM.get(0, creds);
  if (creds.magic != CREDS_MAGIC) {
    creds.magic = 0;
    creds.ssid[0] = 0;
    creds.pass[0] = 0;
  }
  creds.ssid[32] = 0;
  creds.pass[64] = 0;
}

void saveCreds(const char* ssid, const char* pass) {
  creds.magic = CREDS_MAGIC;
  strncpy(creds.ssid, ssid, 32); creds.ssid[32] = 0;
  strncpy(creds.pass, pass, 64); creds.pass[64] = 0;
  EEPROM.put(0, creds);
  EEPROM.commit();
  Serial.printf("Сохранено: \"%s\"\n", creds.ssid);
}

void forgetCreds() {
  creds.magic = 0;
  creds.ssid[0] = 0;
  creds.pass[0] = 0;
  EEPROM.put(0, creds);
  EEPROM.commit();
  Serial.println("Сохранённая сеть стёрта.");
}

// ================= ПОДКЛЮЧЕНИЕ К WiFi =================
bool tryNetwork(const char* ssid, const char* pass, uint32_t timeoutMs) {
  if (!ssid || !ssid[0]) return false;
  Serial.printf("Пробую \"%s\" ", ssid);
  WiFi.begin(ssid, pass);
  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < timeoutMs) {
    updateLed();
    delay(10);
    if ((millis() - t0) % 500 < 11) Serial.print(".");
  }
  Serial.println();
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("  OK, IP %s, RSSI %d dBm\n",
                  WiFi.localIP().toString().c_str(), WiFi.RSSI());
    return true;
  }
  WiFi.disconnect();
  return false;
}

bool connectWiFi() {
  WiFi.persistent(false);
  WiFi.mode(WIFI_STA);
  WiFi.setSleepMode(WIFI_NONE_SLEEP);
  setPattern(80, 80);

  if (creds.magic == CREDS_MAGIC && tryNetwork(creds.ssid, creds.pass, 15000))
    return true;

  for (uint8_t i = 0; i < KNOWN_COUNT; i++)
    if (tryNetwork(KNOWN[i].ssid, KNOWN[i].pass, 10000))
      return true;

  return false;
}

// ================= ПОРТАЛ НАСТРОЙКИ =================
String portalPage(const String& msg) {
  String h = F("<!DOCTYPE html><html lang='ru'><head><meta charset='utf-8'>"
               "<meta name='viewport' content='width=device-width,initial-scale=1'>"
               "<title>RoboCar</title><style>"
               "body{margin:0;padding:24px;background:#14161a;color:#e6e8ec;"
               "font:15px/1.5 system-ui,sans-serif}"
               "h1{font-size:18px;margin:0 0 18px}"
               "label{display:block;color:#8b929d;font-size:13px;margin:14px 0 5px}"
               "input,select{width:100%;padding:11px;background:#1e2229;color:#e6e8ec;"
               "border:1px solid #2c313a;border-radius:8px;font:inherit}"
               "button{width:100%;margin-top:20px;padding:13px;background:#2f6f4f;"
               "border:none;border-radius:8px;color:#eafff2;font:inherit;font-weight:600}"
               ".m{padding:10px;background:#1a1d23;border-radius:8px;color:#8b929d;"
               "font-size:13px;margin-bottom:14px}"
               "</style></head><body><h1>Настройка WiFi</h1>");
  if (msg.length()) h += "<div class='m'>" + msg + "</div>";

  h += F("<form action='/save' method='post'><label>Сеть</label><select name='ssid'>");
  int n = WiFi.scanNetworks();
  for (int i = 0; i < n; i++) {
    h += "<option value='" + WiFi.SSID(i) + "'>" + WiFi.SSID(i) +
         "  (" + WiFi.RSSI(i) + " dBm)</option>";
  }
  h += F("</select><label>Пароль</label><input name='pass' type='password'>"
         "<button type='submit'>Сохранить и перезагрузить</button></form>"
         "</body></html>");
  return h;
}

void startPortal() {
  runMode = MODE_PORTAL;
  WiFi.mode(WIFI_AP);
  WiFi.softAP(AP_SSID, AP_PASS);

  Serial.println();
  Serial.println("=== РЕЖИМ НАСТРОЙКИ ===");
  Serial.printf("Подключись с телефона к сети \"%s\"\n", AP_SSID);
  Serial.printf("Пароль: %s\n", AP_PASS);
  Serial.printf("Открой: http://%s\n", WiFi.softAPIP().toString().c_str());
  Serial.println();

  portal.on("/", []() { portal.send(200, "text/html", portalPage("")); });
  portal.on("/save", []() {
    String s = portal.arg("ssid");
    String p = portal.arg("pass");
    if (s.length() == 0) {
      portal.send(200, "text/html", portalPage("Выбери сеть"));
      return;
    }
    saveCreds(s.c_str(), p.c_str());
    portal.send(200, "text/html",
                "<meta charset='utf-8'><body style='background:#14161a;color:#e6e8ec;"
                "font:16px system-ui;padding:40px'>Сохранено. Перезагружаюсь...</body>");
    delay(800);
    ESP.restart();
  });
  portal.onNotFound([]() { portal.send(200, "text/html", portalPage("")); });
  portal.begin();
}

// ================= ПОИСК СЕРВЕРА =================
void probeServer() {
  String msg = String("ROBOCAR?") + SECRET;
  IPAddress bcast = WiFi.broadcastIP();
  udp.beginPacket(bcast, DISCOVERY_PORT);
  udp.write(msg.c_str());
  udp.endPacket();
}

void pollDiscovery() {
  int sz = udp.parsePacket();
  if (sz <= 0) return;

  char buf[96];
  int len = udp.read(buf, sizeof(buf) - 1);
  if (len <= 0) return;
  buf[len] = 0;

  String s(buf);
  s.trim();

  String expect = String("ROBOCAR!") + SECRET + ":";
  if (!s.startsWith(expect)) return;      // чужой пакет или собственный probe

  uint16_t port = s.substring(expect.length()).toInt();
  if (port == 0) return;

  IPAddress from = udp.remoteIP();
  if (!serverFound || from != serverIP || port != serverPort) {
    serverIP    = from;
    serverPort  = port;
    serverFound = true;
    Serial.printf("Сервер найден: %s:%u\n", serverIP.toString().c_str(), serverPort);
  }
}

// ================= КОМАНДЫ =================
void runSelfTest() {
  Serial.println();
  Serial.println("=== АВТОТЕСТ ===");
  struct Step { int l; int r; const char* what; };
  const Step steps[] = {
    {   0,   0, "стоп" },
    { 100,   0, "только левый вперёд" },
    {   0, 100, "только правый вперёд" },
    { 100, 100, "оба вперёд" },
    {-100,-100, "оба назад" },
    { 100,-100, "поворот вправо" },
    {-100, 100, "поворот влево" },
    {   0,   0, "стоп" },
  };
  for (uint8_t i = 0; i < sizeof(steps)/sizeof(steps[0]); i++) {
    scalePair(steps[i].l, steps[i].r, &targetL, &targetR);
    lastCmdMs = millis();
    failsafeHit = false;
    uint32_t t0 = millis();
    while (millis() - t0 < 700) { updateRamp(); updateLed(); delay(5); }
    Serial.printf("[%d] %s\n", i+1, steps[i].what);
    Serial.printf("    A: IN1=%d IN2=%d PWM=%3d (%s)   B: IN1=%d IN2=%d PWM=%3d (%s)\n",
      digitalRead(PIN_AIN1), digitalRead(PIN_AIN2), abs(curL), dirName(curL),
      digitalRead(PIN_BIN1), digitalRead(PIN_BIN2), abs(curR), dirName(curR));
  }
  targetL = targetR = curL = curR = 0;
  applyMotor(true, 0); applyMotor(false, 0);
  Serial.println("=== готово ===");
  send("OK TEST DONE");
}

void handleCommand(char* c) {
  for (char* p = c; *p; ++p) if (*p == '\r') *p = 0;
  if (strlen(c) == 0) return;

  // WIFI <ssid> <pass> — до перевода в верхний регистр!
  if (!strncmp(c, "WIFI ", 5) || !strncmp(c, "wifi ", 5)) {
    char ssid[33] = {0}, pass[65] = {0};
    int n = sscanf(c + 5, "%32s %64s", ssid, pass);
    if (n >= 1) {
      saveCreds(ssid, pass);
      send("OK сохранено, перезагружаюсь");
      delay(300);
      ESP.restart();
    } else send("ERR формат: WIFI <ssid> <pass>");
    return;
  }

  for (char* p = c; *p; ++p) *p = toupper(*p);
  Serial.print("RX: "); Serial.println(c);

  if (!strncmp(c, "M ", 2)) {
    int l = 0, r = 0;
    if (sscanf(c + 2, "%d %d", &l, &r) == 2) {
      scalePair(constrain(l, -255, 255), constrain(r, -255, 255),
                &targetL, &targetR);
      lastCmdMs = millis();
      failsafeHit = false;
      send(String("OK M ") + targetL + " " + targetR);
    } else send("ERR формат: M <-255..255> <-255..255>");
    return;
  }

  if (!strcmp(c, "STOP"))  { targetL = targetR = 0; lastCmdMs = millis();
                             send("OK STOP"); return; }
  if (!strcmp(c, "BRAKE")) { applyBrake(); lastCmdMs = millis();
                             send("OK BRAKE"); return; }

  if (!strncmp(c, "MAX ", 4)) { MAX_PWM = constrain(atoi(c+4), 20, 255);
                                send(String("OK MAX_PWM = ") + MAX_PWM); return; }
  if (!strncmp(c, "MIN ", 4)) { MIN_PWM = constrain(atoi(c+4), 0, 200);
                                send(String("OK MIN_PWM = ") + MIN_PWM); return; }
  if (!strncmp(c, "FS ", 3))  { FAILSAFE_MS = constrain(atol(c+3), 100, 5000);
                                send(String("OK FAILSAFE = ") + FAILSAFE_MS); return; }

  if (!strcmp(c, "TEST"))   { runSelfTest(); return; }
  if (!strncmp(c, "PAN ", 4)) {
    // Refuse while moving: see the timer note at PIN_SERVO. The server
    // enforces the same order, so this is a backstop, not the only guard.
    if (curL != 0 || curR != 0 || targetL != 0 || targetR != 0) {
      send("ERR PAN while driving"); return;
    }
    int deg = constrain(atoi(c + 4), -80, 80);
    panAngle = 90 + deg;
    panServo.attach(PIN_SERVO);
    panServo.write(panAngle);
    delay(PAN_TRAVEL_MS);
    panServo.detach();
    send(String("OK PAN ") + deg);
    return;
  }
  if (!strcmp(c, "PING"))   { send("PONG"); return; }
  if (!strcmp(c, "STATE"))  { send(String("STATE L=") + curL + " R=" + curR +
                                   " max=" + MAX_PWM + " min=" + MIN_PWM); return; }
  if (!strcmp(c, "ID"))     { send(String("ID ") + String(ESP.getChipId(), HEX) +
                                   " " + WiFi.macAddress()); return; }
  if (!strcmp(c, "IP"))     { send(String("IP ") + WiFi.localIP().toString() +
                                   " сервер " + serverIP.toString()); return; }
  if (!strcmp(c, "RSSI"))   { send(String("RSSI ") + WiFi.RSSI() + " dBm"); return; }
  if (!strcmp(c, "SSID"))   { send(String("SSID ") + WiFi.SSID()); return; }
  if (!strcmp(c, "UPTIME")) { send(String("UPTIME ") + (millis()/1000) + " s"); return; }
  if (!strcmp(c, "HEAP"))   { send(String("HEAP ") + ESP.getFreeHeap()); return; }
  if (!strcmp(c, "VER"))    { send(String("VER robocar-1.0 core ") +
                                   ESP.getCoreVersion()); return; }

  if (!strcmp(c, "FORGET")) { forgetCreds(); send("OK стёрто, перезагружаюсь");
                              delay(300); ESP.restart(); return; }
  if (!strcmp(c, "REBOOT")) { send("OK перезагрузка"); delay(300);
                              ESP.restart(); return; }

  if (!strcmp(c, "LED 1"))    { ledManual = true;  setLed(true);  send("OK LED ON");   return; }
  if (!strcmp(c, "LED 0"))    { ledManual = true;  setLed(false); send("OK LED OFF");  return; }
  if (!strcmp(c, "LED AUTO")) { ledManual = false;                send("OK LED AUTO"); return; }

  if (!strcmp(c, "HELP")) {
    send("M <l> <r> | STOP | BRAKE | TEST | STATE | MAX <n> | MIN <n> | FS <ms> | "
         "WIFI <ssid> <pass> | FORGET | REBOOT | PING ID IP SSID RSSI UPTIME HEAP VER");
    return;
  }

  send(String("ERR UNKNOWN ") + c);
}

void pollClient() {
  while (client.available()) {
    char ch = client.read();
    if (ch == '\n') { rxBuf[rxLen] = 0; handleCommand(rxBuf); rxLen = 0; }
    else if (rxLen < sizeof(rxBuf) - 1) rxBuf[rxLen++] = ch;
    else rxLen = 0;
  }
}

void pollSerial() {
  static char sBuf[160];
  static uint8_t sLen = 0;
  while (Serial.available()) {
    char ch = Serial.read();
    if (ch == '\n') { sBuf[sLen] = 0; handleCommand(sBuf); sLen = 0; }
    else if (sLen < sizeof(sBuf) - 1) sBuf[sLen++] = ch;
    else sLen = 0;
  }
}

// ================= SETUP =================
void setup() {
  motorsInit();
  pinMode(LED_BUILTIN, OUTPUT);
  setLed(false);

  Serial.begin(115200);
  delay(300);
  Serial.println();
  Serial.println();
  Serial.println("========================================");
  Serial.println("  RoboCar — ESP8266 + TB6612FNG");
  Serial.println("========================================");
  Serial.printf("Причина перезагрузки: %s\n", ESP.getResetReason().c_str());

  loadCreds();
  if (creds.magic == CREDS_MAGIC)
    Serial.printf("Сохранённая сеть: \"%s\"\n", creds.ssid);
  else
    Serial.println("Сохранённой сети нет.");

  if (!connectWiFi()) {
    Serial.println("Ни одна сеть не подошла.");
    startPortal();
    return;
  }

  udp.begin(DISCOVERY_PORT);
  Serial.printf("Ищу сервер по UDP broadcast, порт %u\n", DISCOVERY_PORT);
  lastCmdMs = millis();
}

// ================= LOOP =================
void loop() {
  updateLed();

  // --- режим настройки ---
  if (runMode == MODE_PORTAL) {
    setPattern(400, 400);
    portal.handleClient();
    pollSerial();
    return;
  }

  updateRamp();
  checkFailsafe();
  pollSerial();

  // --- потеряли WiFi ---
  if (WiFi.status() != WL_CONNECTED) {
    setPattern(80, 80);
    if (client.connected()) client.stop();
    serverFound = false;
    return;
  }

  pollDiscovery();

  // --- сервер ещё не найден ---
  if (!serverFound) {
    setPattern(150, 850);
    if (millis() - lastProbeMs > 1000) {
      lastProbeMs = millis();
      probeServer();
    }
    return;
  }

  // --- есть адрес, но нет соединения ---
  if (!client.connected()) {
    setPattern(60, 940);
    if (millis() - lastRetryMs > 1500) {
      lastRetryMs = millis();
      Serial.printf("Подключаюсь к %s:%u ... ",
                    serverIP.toString().c_str(), serverPort);
      if (client.connect(serverIP, serverPort)) {
        client.setNoDelay(true);
        Serial.println("есть");
        send(String("HELLO robocar ") + WiFi.localIP().toString() +
             " ssid " + WiFi.SSID());
      } else {
        Serial.println("нет ответа");
        serverFound = false;       // сервер мог переехать — ищем заново
      }
    }
    return;
  }

  // --- рабочий режим ---
  setPattern(curL || curR ? 150 : 1000, curL || curR ? 150 : 1000);
  pollClient();

  if (millis() - lastTeleMs > 2000) {
    lastTeleMs = millis();
    send(String("HB up=") + (millis()/1000) +
         " L=" + curL + " R=" + curR +
         " rssi=" + WiFi.RSSI() +
         " heap=" + ESP.getFreeHeap());
  }
}
