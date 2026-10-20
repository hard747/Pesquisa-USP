

#define CLK_OUT_PIN  5

volatile unsigned long syncled_period = 5347U; // at 187fps it is equivalent to 1 frame period

volatile unsigned long periodtimer3; // 1/16000000 s
volatile unsigned long periodtimer3_final;
volatile unsigned long overflows;
volatile unsigned long totaloverflows;

// ================================================================
// ===               INTERRUPT DETECTION ROUTINE                ===
// ================================================================


ISR(TIMER3_COMPA_vect) {
  if (overflows == 1) {
    OCR3A = periodtimer3_final;
    overflows--;
  }
  else if (overflows == 0) {
    OCR3A = periodtimer3;
    overflows = totaloverflows;

    digitalWrite(CLK_OUT_PIN, HIGH);
    int i = 0;
    while (i < 64) i++;
    digitalWrite(CLK_OUT_PIN, LOW);
  }
  else {
    overflows--;
  }
}


// ================================================================
// ===                      INITIAL SETUP                       ===
// ================================================================

void setup() {

  if (syncled_period << 4 > 0xFFFE) {
    periodtimer3 = 0xFFFE;
    totaloverflows = syncled_period >> 12;
    periodtimer3_final = (syncled_period << 4) - ((totaloverflows << 16) - 1);
  } else {
    totaloverflows = 0;
    periodtimer3_final = periodtimer3 = (syncled_period << 4);
  }
  if (periodtimer3_final < 4000 && totaloverflows > 0) {
    long diff = 4000 - periodtimer3_final;
    periodtimer3 -= diff / totaloverflows;
    periodtimer3_final = 4000 - (diff - totaloverflows * (diff / totaloverflows));
  }
  overflows = 1;

  pinMode(CLK_OUT_PIN, OUTPUT);
  digitalWrite(CLK_OUT_PIN, LOW);

  cli(); //stop interrupts

  //----------- TIMER 3 SETUP -----------
  TCCR3A = 0;// set entire TCCR1A register to 0
  TCCR3B = 0;// same for TCCR1B
  TCNT3  = 0;//initialize counter value to 0

  OCR3A = periodtimer3;

  // turn on CTC mode
  TCCR3B |= (1 << WGM32);
  // Set CS30 bits for no prescaler
  TCCR3B |= (1 << CS30);/* no prescaler */

  TIMSK3 |= (1 << OCIE3A);
  //----------- TIMER 3 SETUP -----------

  sei();//allow interrupts

  // initialize serial communication
  Serial.begin(115200);
  while (!Serial); // wait for Leonardo enumeration, others continue immediately
}



// ================================================================
// ===                    MAIN PROGRAM LOOP                     ===
// ================================================================

void loop() {
  // read serial:
  while (Serial.available() > 0) {
    byte inByte = Serial.read();
    if (inByte == 'V') {
      Serial.println("CLKEL:");
    } else if (inByte == 'v') {
      long time = totaloverflows * periodtimer3 + periodtimer3_final;
      Serial.println(time, DEC);
    } else if (inByte == 'k') { // half of microsecond resolution
      long tmp = Serial.parseInt();
      if (tmp > 2000 && tmp <= 65535) {
        syncled_period = tmp;
        if (syncled_period << 4 > 0xFFFE) {
          periodtimer3 = 0xFFFE;
          totaloverflows = syncled_period >> 12;
          periodtimer3_final = (syncled_period << 4) - ((totaloverflows << 16) - 1);
        } else {
          totaloverflows = 0;
          periodtimer3_final = periodtimer3 = (syncled_period << 4);
        }
        if (periodtimer3_final < 4000 && totaloverflows > 0) {
          long diff = 4000 - periodtimer3_final;
          periodtimer3 -= diff / totaloverflows;
          periodtimer3_final = 4000 - (diff - totaloverflows * (diff / totaloverflows));
        }
        Serial.println(periodtimer3, DEC);
        Serial.println(totaloverflows, DEC);
        Serial.println(periodtimer3_final, DEC);
        cli(); //stop interrupts
        OCR3A = periodtimer3;
        overflows = totaloverflows;
        sei(); //enable interrupts
      }
    } else if (inByte == 'o') { // 62.5ns resolution
      long tmp = Serial.parseInt();
      if (tmp > 0 && tmp <= 65535) {
        // adjust only the offset, not the whole value
        // 32000 is zero = no adjust
        long ocrsave = periodtimer3_final;
        long diff = (tmp - 32000);
        Serial.println(diff, DEC);
        if ((diff + periodtimer3_final) < 0) {
          totaloverflows--;
          diff += periodtimer3_final;
          periodtimer3_final = 0xFFFE + diff;
        } else if ((diff + periodtimer3_final) > 0xFFFE) {
          totaloverflows++;
          diff -= periodtimer3_final;
          periodtimer3_final = diff;
        } else {
          periodtimer3_final += diff;
        }
        if (periodtimer3_final < 4000 && totaloverflows > 0) {
          diff = 4000 - periodtimer3_final;
          periodtimer3 -= diff / totaloverflows;
          periodtimer3_final = 4000 - (diff - totaloverflows * (diff / totaloverflows));
        }
        Serial.println(periodtimer3, DEC);
        Serial.println(totaloverflows, DEC);
        Serial.println(periodtimer3_final, DEC);
        if (OCR3A == ocrsave) {// if timer 3 is on the last part
          cli(); //stop interrupts
          OCR3A = periodtimer3_final;
          sei(); //enable interrupts
        }
      }
    }
  }
}

