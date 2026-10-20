#include <avr/io.h>
#include <avr/interrupt.h>

int const calibration_leds_pin = A5;  // LED para pupila brillante
int const ir_led_pin = A4;            // LED para pupila oscura
volatile bool led_state = false;      // Variable para alternar el estado de los LEDs
volatile bool sync_enabled = false;   // Controla si el temporizador está activo

void setup() {
  pinMode(calibration_leds_pin, OUTPUT);
  pinMode(ir_led_pin, OUTPUT);

  // Inicializa la comunicación serial
  Serial.begin(115200);

  // Configura Timer1 para interrupciones a 60 Hz (ajustable)
  cli(); // Desactiva las interrupciones globales
  TCCR1A = 0; // Configura el registro de control del temporizador
  TCCR1B = 0;
  TCNT1 = 0;
  
  // Configura el temporizador para 60 Hz (aproximadamente 16.67 ms por interrupción)
  OCR1A = 163;             // Ajustado para 60 Hz
  //OCR1A = 83;             // Ajustado para 187 Hz
  TCCR1B |= (1 << WGM12);    // Modo CTC (Clear Timer on Compare Match)
  TCCR1B |= (1 << CS12) | (1 << CS10); // Prescaler de 1024
  TIMSK1 |= (1 << OCIE1A);   // Habilita la interrupción por comparación
  sei(); // Activa las interrupciones globales
}

void loop() {
  // Activa o desactiva el temporizador con un comando serial
  if (Serial.available() > 0) {
    char command = Serial.read();
    if (command == '1') {
      sync_enabled = true;
      Serial.println("Sincronización activada.");
    } else if (command == '0') {
      sync_enabled = false;
      digitalWrite(calibration_leds_pin, LOW);
      digitalWrite(ir_led_pin, LOW);
      Serial.println("Sincronización desactivada.");
    }
  }
}

// Interrupción del Timer1
ISR(TIMER1_COMPA_vect) {
  if (sync_enabled) {
    // Alterna los LEDs
    if (led_state) {
      digitalWrite(calibration_leds_pin, HIGH); // Pupila brillante
      digitalWrite(ir_led_pin, LOW);
    } else {
      digitalWrite(calibration_leds_pin, LOW);
      digitalWrite(ir_led_pin, HIGH); // Pupila oscura
    }
    led_state = !led_state; // Cambia el estado para la próxima interrupción
  }
}
