// #include <Arduino.h>
// #include <ESP8266WiFi.h>

// extern "C" {
//   #include "user_interface.h"
// }

// const char* ssid = "Luke’s iPhone";
// const char* password = "hihihihi";

// String statusToStr(wl_status_t s) {
//   switch (s) {
//     case WL_IDLE_STATUS: return "IDLE";
//     case WL_NO_SSID_AVAIL: return "NO_SSID_AVAIL";
//     case WL_SCAN_COMPLETED: return "SCAN_COMPLETED";
//     case WL_CONNECTED: return "CONNECTED";
//     case WL_CONNECT_FAILED: return "CONNECT_FAILED";
//     case WL_CONNECTION_LOST: return "CONNECTION_LOST";
//     case WL_DISCONNECTED: return "DISCONNECTED";
//     default: return "UNKNOWN";
//   }
// }

// void setup() {
//   Serial.begin(115200);
//   delay(500);

//   Serial.println();
//   Serial.println("Resetting WiFi...");

//   WiFi.mode(WIFI_STA);
//   WiFi.disconnect();
//   delay(100);

//   // Explicitly set US channel range
//   wifi_country_t country = {"US", 1, 11, 0};
//   wifi_set_country(&country);

//   Serial.print("Connecting to: ");
//   Serial.println(ssid);

//   WiFi.begin(ssid, password);

//   unsigned long startAttemptTime = millis();

//   while (WiFi.status() != WL_CONNECTED &&
//          millis() - startAttemptTime < 15000) {

//     Serial.print("Status: ");
//     Serial.println(statusToStr(WiFi.status()));

//     delay(1000);
//   }

//   Serial.println();
//   Serial.print("Final Status: ");
//   Serial.println(statusToStr(WiFi.status()));

//   if (WiFi.status() == WL_CONNECTED) {
//     Serial.println("Connected!");
//     Serial.print("IP address: ");
//     Serial.println(WiFi.localIP());
//   } else {
//     Serial.println("Failed to connect after 15 seconds.");
//   }
// }

// void loop() {
// }

#include <Arduino.h>
#include <ESP8266WiFi.h>

extern "C" {
  #include "user_interface.h"
}

const char* target = "Luke’s iPhone";
// const char* target = "Treehacks-2026";

void setup() {
  Serial.begin(115200);
  delay(500);

  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  delay(200);

  wifi_country_t country = {"US", 1, 11, 0};
  wifi_set_country(&country);

  Serial.println("\nScanning...");
  int n = WiFi.scanNetworks(false, true); // show hidden too
  Serial.printf("Found %d networks\n", n);

  bool found = false;
  for (int i = 0; i < n; i++) {
    String s = WiFi.SSID(i);
    int rssi = WiFi.RSSI(i);
    Serial.printf("%2d) '%s' RSSI=%d CH=%d\n", i+1, s.c_str(), rssi, WiFi.channel(i));
    if (s == target) found = true;
  }

  if (!found) {
    Serial.println("\nDid NOT see target SSID. If your laptop sees it, this points to hotspot advertising quirks or ESP RF/power.");
  } else {
    Serial.println("\nSaw target SSID!");
  }
}

void loop() {}
