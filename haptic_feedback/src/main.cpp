#include <Arduino.h>
#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <ArduinoJson.h>

#define MOTOR_PIN 5

// const char* ssid = "Treehacks-2026";
// const char* password = "treehacks2026!";

const char* ssid = "Luke’s iPhone";
const char* password = "hihihihi";

extern "C" {
  #include "user_interface.h"
}

ESP8266WebServer server(80);

wl_status_t lastStatus = WL_DISCONNECTED;
unsigned long lastWifiCheck = 0;
unsigned long lastRSSIPrint = 0;

// --- NEW: drawing + LED blink state ---
volatile bool drawing = false;
unsigned long lastBlink = 0;
bool ledState = false;

String statusToStr(wl_status_t s) {
  switch (s) {
    case WL_IDLE_STATUS: return "IDLE";
    case WL_NO_SSID_AVAIL: return "NO_SSID_AVAIL";
    case WL_SCAN_COMPLETED: return "SCAN_COMPLETED";
    case WL_CONNECTED: return "CONNECTED";
    case WL_CONNECT_FAILED: return "CONNECT_FAILED";
    case WL_CONNECTION_LOST: return "CONNECTION_LOST";
    case WL_DISCONNECTED: return "DISCONNECTED";
    default: return "UNKNOWN";
  }
}

void handleDrawing() {
  if (server.method() != HTTP_POST) {
    server.send(405, "text/plain", "Use POST");
    return;
  }

  JsonDocument doc;
  if (deserializeJson(doc, server.arg("plain"))) {
    server.send(400, "text/plain", "Bad JSON");
    return;
  }

  drawing = doc["drawing"] | false;

  // Motor control (unchanged)
  digitalWrite(MOTOR_PIN, drawing ? HIGH : LOW);

  // Helpful debug
  Serial.print("drawing = ");
  Serial.println(drawing ? "true" : "false");

  // Optional but helps with ESP8266 HTTP stability
  server.sendHeader("Connection", "close");
  server.send(200, "text/plain", "OK");
}

void connectWifi() {
  Serial.println("\nAttempting WiFi connection...");
  WiFi.begin(ssid, password);
}

void setup() {
  Serial.begin(115200);
  delay(500);

  pinMode(MOTOR_PIN, OUTPUT);
  digitalWrite(MOTOR_PIN, LOW);

  // --- NEW: LED setup (active-low on most ESP8266 boards) ---
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, HIGH); // OFF initially

  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  delay(200);

  wifi_country_t country = {"US", 1, 11, 0};
  wifi_set_country(&country);

  WiFi.setAutoReconnect(true);
  WiFi.persistent(false);

  connectWifi();

  server.on("/drawing", handleDrawing);
  server.begin();

  Serial.println("HTTP server started on port 80");
}

void loop() {
  server.handleClient();

  // --- NEW: Blink built-in LED while drawing=true ---
  if (drawing) {
    if (millis() - lastBlink > 200) {
      lastBlink = millis();
      ledState = !ledState;
      // active-low LED: LOW = ON, HIGH = OFF
      digitalWrite(LED_BUILTIN, ledState ? LOW : HIGH);
    }
  } else {
    digitalWrite(LED_BUILTIN, HIGH); // keep OFF when not drawing
  }

  wl_status_t currentStatus = WiFi.status();

  // Detect status change
  if (currentStatus != lastStatus) {
    Serial.print("WiFi status changed: ");
    Serial.println(statusToStr(currentStatus));

    if (currentStatus == WL_CONNECTED) {
      Serial.print("IP: ");
      Serial.println(WiFi.localIP());
      Serial.print("RSSI: ");
      Serial.println(WiFi.RSSI());
    }

    lastStatus = currentStatus;
  }

  // Periodically check connection
  if (millis() - lastWifiCheck > 5000) {
    lastWifiCheck = millis();

    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("WiFi lost. Reconnecting...");
      connectWifi();
    }
  }

  // Periodic RSSI monitor
  if (WiFi.status() == WL_CONNECTED && millis() - lastRSSIPrint > 10000) {
    lastRSSIPrint = millis();
    Serial.print("Signal strength RSSI: ");
    Serial.println(WiFi.RSSI());
  }
}
